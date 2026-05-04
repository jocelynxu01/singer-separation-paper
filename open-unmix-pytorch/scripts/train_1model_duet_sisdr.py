import argparse
import torch
import time
from pathlib import Path
from typing import Tuple
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio
import torch
# torch.set_num_threads(1)
import torch.nn as nn

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms
import Customized_NetSV as M 
from asteroid.losses import singlesrc_neg_sisdr

tqdm.monitor_interval = 0
SEGMENT_LENGTH = 3

def save_audio(tensor, filename, sample_rate=44100):
    # Ensure the tensor is on the CPU
    tensor = tensor.detach().cpu()

    # Convert the tensor to a waveform if it's in stereo (channels > 1)
    if tensor.ndimension() == 3:  # [batch, channels, time]
        tensor = tensor[0]  # Take the first batch, assuming single batch

    # Save the waveform
    torchaudio.save(filename, tensor, sample_rate)
    print(f"Audio saved to {filename}")
    
import os
import torchaudio

def save_wav(tensor, path, sr=44100):
    """
    tensor: [B, C, T] or [C, T] or [T]
    Saves first item in batch, downmixed to mono.
    """
    t = tensor.detach().cpu()

    if t.dim() == 3:        # [B, C, T]
        t = t[0]
    if t.dim() == 2 and t.size(0) > 1:
        t = t.mean(dim=0, keepdim=True)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    torchaudio.save(path, t, sr)

def listen_one_example(args, unmix, encoder, decoder, device, loader):
    unmix.eval()
    with torch.no_grad():
        x, y_v1, y_v2, bgm = next(iter(loader))

        x, y_v1, y_v2 = x.to(device), y_v1.to(device), y_v2.to(device)
        y_total = y_v1 + y_v2
        
        save_audio(x, "original mixture.wav")
        save_audio(y_v1, "original v1.wav")
        save_audio(y_v2, "original v2.wav")

        X = encoder(x)
        X_complex = torch.view_as_complex(X)
        X_mag = torch.abs(X_complex)
        X_phase = torch.angle(X_complex)

        Y_hat_mag = unmix(X_mag)
        Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
        Y_hat_real = torch.view_as_real(Y_hat_complex)
        y_hat_time = decoder(Y_hat_real)

        # save both (downmix to mono + align length so torchaudio saves cleanly)
        min_len = min(y_hat_time.size(-1), y_total.size(-1))
        pred = y_hat_time[..., :min_len].mean(dim=1, keepdim=True)   # [B,1,T]
        gt   = y_total[..., :min_len].mean(dim=1, keepdim=True)

        save_audio(pred, "predicted_audio.wav")
        save_audio(gt, "ground_truth_audio_clean.wav")
        save_audio(x, "ground_truth_audio_mixture.wav")
        print("pred:", pred.shape, "gt:", gt.shape)
        
        

def train(args, unmix, encoder, decoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    
    for x, y_v1, y_v2, bgm in pbar:
        pbar.set_description("Training batch")

        # --- move to device ---
        x, y_v1, y_v2 = x.to(device), y_v1.to(device), y_v2.to(device)
        
        # print(x.shape)
        # print(y_v1.shape)
        # print(y_v2.shape)
        
        if x.shape[-1] < args.nfft:
            print(f"Skipping short sample: len={x.shape[-1]} (nfft={args.nfft})")
            continue
        
        y_total = y_v1 + y_v2  # target = combined vocals
        
        optimizer.zero_grad()

        # --- STFT encoding ---
        X = encoder(x)
        X_complex = torch.view_as_complex(X)
        X_mag = torch.abs(X_complex)
        X_phase = torch.angle(X_complex)

        # --- model forward ---
        Y_hat_mag = unmix(X_mag)
        Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
        Y_hat_real = torch.view_as_real(Y_hat_complex)
        y_hat_time = decoder(Y_hat_real)
        
        # --- alignment ---
        min_length = min(y_hat_time.size(-1), y_total.size(-1))
        y_hat_time = y_hat_time[..., :min_length].mean(dim=1)
        y_total = y_total[..., :min_length].mean(dim=1)

        # --- loss ---
        loss = singlesrc_neg_sisdr(y_hat_time, y_total).mean()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), y_total.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.3f}")

    return losses.avg

def train_DEBUG(args, unmix, encoder, decoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)

    saved = False
    debug_dir = "TRAIN_DEBUG_AUDIO"
    os.makedirs(debug_dir, exist_ok=True)

    for step, (x, y_v1, y_v2, bgm) in enumerate(pbar):
        pbar.set_description("Training batch")

        # --- move to device ---
        x, y_v1, y_v2, bgm = (
            x.to(device),
            y_v1.to(device),
            y_v2.to(device),
            bgm.to(device),
        )

        if x.shape[-1] < args.nfft:
            continue

        # --- target ---
        y_total = y_v1 + y_v2

        optimizer.zero_grad()

        # --- STFT ---
        X = encoder(x)
        X_complex = torch.view_as_complex(X)
        X_mag = torch.abs(X_complex)
        X_phase = torch.angle(X_complex)

        # --- forward ---
        Y_hat_mag = unmix(X_mag)
        Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
        Y_hat_real = torch.view_as_real(Y_hat_complex)
        y_hat_time = decoder(Y_hat_real)

        # --- align ---
        min_length = min(y_hat_time.size(-1), y_total.size(-1), x.size(-1))
        y_hat_time = y_hat_time[..., :min_length].mean(dim=1)
        y_total = y_total[..., :min_length].mean(dim=1)
        mixture = x[..., :min_length].mean(dim=1)

        # --- SAVE FIRST BATCH ONLY ---
        if not saved:
            sr = 44100

            save_audio(mixture, f"{debug_dir}/00_mixture.wav", sr)
            save_audio(y_v1[..., :min_length], f"{debug_dir}/01_y1.wav", sr)
            save_audio(y_v2[..., :min_length], f"{debug_dir}/02_y2.wav", sr)
            save_audio(bgm[..., :min_length], f"{debug_dir}/03_bgm.wav", sr)
            save_audio(y_total, f"{debug_dir}/04_y1y2_target.wav", sr)

            save_audio(y_hat_time, f"{debug_dir}/05_pred_vocals.wav", sr)
            save_audio(mixture - y_hat_time, f"{debug_dir}/06_pred_residual.wav", sr)

            print(f"✅ Saved training debug audio to {debug_dir}/")
            saved = True

        # --- loss ---
        loss = singlesrc_neg_sisdr(y_hat_time, y_total).mean()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), y_total.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.3f}")

    return losses.avg

def valid(args, unmix, encoder, decoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    
    pbar = tqdm.tqdm(valid_sampler, disable=args.quiet)
    with torch.no_grad():
        for x, y_v1, y_v2, bgm in pbar:
            x, y_v1, y_v2 = x.to(device), y_v1.to(device), y_v2.to(device)
            y_total = y_v1 + y_v2  # target = combined vocals

            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag = torch.abs(X_complex)
            X_phase = torch.angle(X_complex)

            Y_hat_mag = unmix(X_mag)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            Y_hat_real = torch.view_as_real(Y_hat_complex)
            y_hat_time = decoder(Y_hat_real)

            min_length = min(y_hat_time.size(-1), y_total.size(-1))
            y_hat_time = y_hat_time[..., :min_length].mean(dim=1)
            y_total = y_total[..., :min_length].mean(dim=1)

            loss = singlesrc_neg_sisdr(y_hat_time, y_total).mean()
            losses.update(loss.item(), y_total.size(0))

    return losses.avg
    
def valid_for_test(args, unmix, encoder, decoder, device, valid_sampler):
    losses_V = utils.AverageMeter()
    losses_R = utils.AverageMeter()
    unmix.eval()
    
    pbar = tqdm.tqdm(valid_sampler, disable=args.quiet)
    with torch.no_grad():
        for x, y in pbar:
            if x.dim() == 3 and x.size(1) > 1:
                x = x.mean(dim=1, keepdim=True)
            if y.dim() == 3 and y.size(1) > 1:
                y = y.mean(dim=1, keepdim=True)
            
            x, y = x.to(device), y.to(device)
            
            if x.shape[-1] < args.nfft or y.shape[-1] < args.nfft:
                print(f"Skipping short sample: len={x.shape[-1]} (nfft={args.nfft})")
                continue
            
            X = encoder(x) # 16, 2, 2049, 255, 2]
            Y = encoder(y)
                    
            X_complex = torch.view_as_complex(X)  # Shape: (batch, channels, freq_bins, frames)
            X_mag = torch.abs(X_complex)  # Get magnitude, [1, 1, 2049, 259]
            X_phase = torch.angle(X_complex)
            
            Y_hat_mag = unmix(X_mag)
            
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase) # [16, 2, 2049, 255]
            Y_hat_real = torch.view_as_real(Y_hat_complex) # [16, 2, 2049, 255, 2]
                    
            y_hat_time = decoder(Y_hat_real)
            
            min_length = min(y_hat_time.size(-1), y.size(-1), x.size(-1)) 
            
            y_hat_time_singer1 = y_hat_time[..., :min_length].mean(dim=1)
            y_singer1 = y[..., :min_length].mean(dim=1)
            loss_V = singlesrc_neg_sisdr(y_hat_time_singer1, y_singer1)
            loss_V = loss_V.mean()
            losses_V.update(loss_V.item(), Y.size(1))
            
            residual = x[..., :min_length] - y_hat_time[..., :min_length]  # Mixture - Reconstructed Singer1
            background = x[..., :min_length] - y[..., :min_length]  # Mixture - Singer1 (ground truth)
            background_loss = singlesrc_neg_sisdr(residual.mean(dim=1), background.mean(dim=1))
            background_loss = background_loss.mean()
            losses_R.update(background_loss.item(), Y.size(1))
            
        return losses_V.avg, losses_R.avg

def valid_for_test_solo_fad(args, unmix, encoder, decoder, device, valid_sampler):
    """
    Evaluate baseline model on the solo test set using FAD.
    """
    unmix.eval()

    with tempfile.TemporaryDirectory() as tmp:
        pred_v1_dir = os.path.join(tmp, "pred_v1")
        ref_v1_dir = os.path.join(tmp, "ref_v1")
        pred_bgm_dir = os.path.join(tmp, "pred_bgm")
        ref_bgm_dir = os.path.join(tmp, "ref_bgm")
        mixture_dir = os.path.join(tmp, "mixture")

        os.makedirs(pred_v1_dir)
        os.makedirs(ref_v1_dir)
        os.makedirs(pred_bgm_dir)
        os.makedirs(ref_bgm_dir)
        os.makedirs(mixture_dir)

        with torch.no_grad():
            idx = 0
            skipped = 0

            for x, y in tqdm.tqdm(valid_sampler, disable=args.quiet, desc="Evaluating SOLO Test (FAD, baseline)"):

                x, y = x.to(device), y.to(device)

                if x.shape[-1] < args.nfft or y.shape[-1] < args.nfft:
                    print(f"Skipping short sample: len={x.shape[-1]} (nfft={args.nfft})")
                    skipped += 1
                    continue

                # forward
                X = encoder(x)
                X_complex = torch.view_as_complex(X)
                X_mag = torch.abs(X_complex)
                X_phase = torch.angle(X_complex)

                Y_hat_mag = unmix(X_mag)
                Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
                Y_hat_real = torch.view_as_real(Y_hat_complex)
                y_hat_time = decoder(Y_hat_real)

                # align lengths
                min_len = min(y_hat_time.size(-1), y.size(-1), x.size(-1))

                y1_hat = y_hat_time[..., :min_len].mean(dim=1)
                y_v1 = y[..., :min_len].mean(dim=1)
                x_mono = x[..., :min_len].mean(dim=1)

                residual = x_mono - y1_hat
                bgm = x_mono - y_v1

                try:
                    pred_v1 = y1_hat[0].detach().cpu().float().squeeze().numpy()
                    ref_v1 = y_v1[0].detach().cpu().float().squeeze().numpy()
                    pred_bgm = residual[0].detach().cpu().float().squeeze().numpy()
                    ref_bgm = bgm[0].detach().cpu().float().squeeze().numpy()
                    mixture = x_mono[0].detach().cpu().float().squeeze().numpy()

                    sf.write(os.path.join(pred_v1_dir, f"{idx}.wav"), pred_v1, 44100)
                    sf.write(os.path.join(ref_v1_dir, f"{idx}.wav"), ref_v1, 44100)
                    sf.write(os.path.join(pred_bgm_dir, f"{idx}.wav"), pred_bgm, 44100)
                    sf.write(os.path.join(ref_bgm_dir, f"{idx}.wav"), ref_bgm, 44100)
                    sf.write(os.path.join(mixture_dir, f"{idx}.wav"), mixture, 44100)

                    idx += 1

                except Exception as e:
                    skipped += 1
                    print(f"Skipping sample {idx} due to error: {e}")
                    continue

        print(f"FAD valid samples: {idx}, skipped: {skipped}")

        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            verbose=False,
        )

        vocal_fad = fad.score(ref_v1_dir, pred_v1_dir)
        bgm_fad = fad.score(ref_bgm_dir, pred_bgm_dir)
        mixture_vocal_fad = fad.score(ref_v1_dir, mixture_dir)
        vocal_fad_improvement = mixture_vocal_fad - vocal_fad

        print("\n--- SOLO Test FAD Summary (Baseline Model) ---")
        print(f"vocal_fad: {vocal_fad:.4f}")
        print(f"bgm_fad: {bgm_fad:.4f}")
        print(f"mixture_vocal_fad: {mixture_vocal_fad:.4f}")
        print(f"vocal_fad_improvement: {vocal_fad_improvement:.4f}")

        return {
            "vocal_fad": vocal_fad,
            "bgm_fad": bgm_fad,
            "mixture_vocal_fad": mixture_vocal_fad,
            "vocal_fad_improvement": vocal_fad_improvement,
        }
        
def valid_for_test_duet(args, unmix, encoder, decoder, device, duet_loader):
    losses = {k: utils.AverageMeter() for k in ["v1", "v2", "v1v2", "bgm", "bgmv2"]}
    unmix.eval()

    with torch.no_grad():
        for x, y_v1, y_v2, bgm in tqdm.tqdm(duet_loader, desc="Evaluating Duet Test"):
            x, y_v1, y_v2, bgm = [t.to(device) for t in (x, y_v1, y_v2, bgm)]

            # ---- Model forward ----
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)
            Y_hat_mag = unmix(X_mag)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            Y_hat_real = torch.view_as_real(Y_hat_complex)
            y1_hat = decoder(Y_hat_real)

            # Align lengths & downmix
            min_len = min(y1_hat.size(-1), y_v1.size(-1), y_v2.size(-1), bgm.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            y_v2 = y_v2[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            residual = x[..., :min_len].mean(dim=1) - y1_hat

            # ---- Compute losses ----
            losses["v1"].update(singlesrc_neg_sisdr(y1_hat, y_v1).mean().item())
            losses["v2"].update(singlesrc_neg_sisdr(y1_hat, y_v2).mean().item())
            losses["v1v2"].update(singlesrc_neg_sisdr(y1_hat, y_v1 + y_v2).mean().item())
            losses["bgm"].update(singlesrc_neg_sisdr(residual, bgm).mean().item())
            losses["bgmv2"].update(singlesrc_neg_sisdr(residual, bgm + y_v2).mean().item())

    print("\n--- Duet Test Loss Summary ---")
    for k, v in losses.items():
        print(f"{k}: {v.avg:.4f}")

    return tuple(v.avg for v in losses.values())

import os
import tempfile
import soundfile as sf
from frechet_audio_distance import FrechetAudioDistance

def valid_for_test_duet_fad(unmix, encoder, decoder, device, duet_loader):
    """
    Evaluate baseline model on the duet test set using FAD.
    """
    unmix.eval()

    with tempfile.TemporaryDirectory() as tmp:

        pred_v1_dir = os.path.join(tmp, "pred_v1")
        ref_v1_dir = os.path.join(tmp, "ref_v1")
        pred_residual_dir = os.path.join(tmp, "pred_residual")
        ref_bgmv2_dir = os.path.join(tmp, "ref_bgmv2")
        mixture_dir = os.path.join(tmp, "mixture")

        os.makedirs(pred_v1_dir)
        os.makedirs(ref_v1_dir)
        os.makedirs(pred_residual_dir)
        os.makedirs(ref_bgmv2_dir)
        os.makedirs(mixture_dir)

        with torch.no_grad():
            idx = 0
            skipped = 0

            for x, y_v1, y_v2, bgm in tqdm.tqdm(
                duet_loader, desc="Evaluating DUET Test (FAD, baseline)"
            ):
                x = x.to(device)
                y_v1 = y_v1.to(device)
                y_v2 = y_v2.to(device)
                bgm = bgm.to(device)

                # forward
                X = encoder(x)
                X_complex = torch.view_as_complex(X)
                X_mag = torch.abs(X_complex)
                X_phase = torch.angle(X_complex)

                Y_hat_mag = unmix(X_mag)
                Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
                y1_hat = decoder(torch.view_as_real(Y_hat_complex))

                # align lengths and downmix to mono
                min_len = min(
                    y1_hat.size(-1),
                    y_v1.size(-1),
                    y_v2.size(-1),
                    bgm.size(-1),
                    x.size(-1),
                )

                y1_hat = y1_hat[..., :min_len].mean(dim=1)
                y_v1 = y_v1[..., :min_len].mean(dim=1)
                y_v2 = y_v2[..., :min_len].mean(dim=1)
                bgm = bgm[..., :min_len].mean(dim=1)
                x_mono = x[..., :min_len].mean(dim=1)

                residual = x_mono - y1_hat
                ref_bgmv2 = bgm + y_v2

                try:
                    pred_v1 = y1_hat[0].detach().cpu().float().squeeze().numpy()
                    ref_v1 = y_v1[0].detach().cpu().float().squeeze().numpy()
                    pred_residual = residual[0].detach().cpu().float().squeeze().numpy()
                    ref_residual = ref_bgmv2[0].detach().cpu().float().squeeze().numpy()
                    mixture = x_mono[0].detach().cpu().float().squeeze().numpy()

                    sf.write(os.path.join(pred_v1_dir, f"{idx}.wav"), pred_v1, 44100)
                    sf.write(os.path.join(ref_v1_dir, f"{idx}.wav"), ref_v1, 44100)
                    sf.write(os.path.join(pred_residual_dir, f"{idx}.wav"), pred_residual, 44100)
                    sf.write(os.path.join(ref_bgmv2_dir, f"{idx}.wav"), ref_residual, 44100)
                    sf.write(os.path.join(mixture_dir, f"{idx}.wav"), mixture, 44100)

                    idx += 1

                except Exception as e:
                    skipped += 1
                    print(f"Skipping sample {idx} due to error: {e}")
                    continue
                
                # if idx >= 200:  # Limit to first 200 valid samples for FAD evaluation
                #     print("Reached 200 valid samples, stopping further processing for FAD.")
                #     break

        print(f"FAD valid samples: {idx}, skipped: {skipped}")

        # fad = FrechetAudioDistance(
        #     model_name="vggish",
        #     sample_rate=16000,
        #     verbose=False,
        # )
        
        fad = FrechetAudioDistance(
            model_name="encodec",
            sample_rate=48000,
            channels=2,
            verbose=False,
        )

        vocal_fad = fad.score(ref_v1_dir, pred_v1_dir)
        residual_fad = fad.score(ref_bgmv2_dir, pred_residual_dir)
        mixture_vocal_fad = fad.score(ref_v1_dir, mixture_dir)
        vocal_fad_improvement = mixture_vocal_fad - vocal_fad

        print("\n--- DUET Test FAD Summary (Baseline Model) ---")
        print(f"vocal_fad: {vocal_fad:.4f}")
        print(f"residual_fad: {residual_fad:.4f}")
        print(f"mixture_vocal_fad: {mixture_vocal_fad:.4f}")
        print(f"vocal_fad_improvement: {vocal_fad_improvement:.4f}")

        return {
            "vocal_fad": vocal_fad,
            "residual_fad": residual_fad,
            "mixture_vocal_fad": mixture_vocal_fad,
            "vocal_fad_improvement": vocal_fad_improvement,
        }


def save_first_test_outputs(unmix, encoder, decoder, device, test_loader, save_dir="TEST_AUDIO_DEBUG_DUET_V2"):
    os.makedirs(save_dir, exist_ok=True)
    unmix.eval()

    with torch.no_grad():

        example_idx = 0

        for i, (x, y_v1, y_v2, bgm) in enumerate(tqdm.tqdm(test_loader, desc="Evaluating Duet Test")):
            x, y_v1, y_v2, bgm = [t.to(device) for t in (x, y_v1, y_v2, bgm)]
            
            # print(x.shape)
            # print(y_v1.shape)
            # print(y_v2.shape)

            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)
            Y_hat_mag = unmix(X_mag)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            Y_hat_real = torch.view_as_real(Y_hat_complex)
            y1_hat = decoder(Y_hat_real)

            # Align lengths
            min_len = min(y1_hat.size(-1), y_v1.size(-1), y_v2.size(-1), bgm.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            y_v2 = y_v2[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            mixture = x[..., :min_len].mean(dim=1)

            residual = mixture - y1_hat

            # --- Save audio examples ---
            if example_idx < 20:
                prefix = f"{save_dir}/example_{example_idx:02d}"
                torchaudio.save(f"{prefix}_mixture.wav", mixture.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_vocal1_hat.wav", y1_hat.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_vocal1_true.wav", y_v1.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_vocal2_true.wav", y_v2.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_bgm_true.wav", bgm.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_residual.wav", residual.cpu(), sample_rate=44100)
            else:
                break
            example_idx += 1

    print(f"✅ Saved mixture + sources to: {save_dir}")
    

def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y, e, b = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std

def valid_for_testing(args, unmix, encoder, decoder, device, valid_sampler, epoch=0):
    """
    Validation loop with optional audio output (first 10 samples only on epoch 1)
    """
    losses = utils.AverageMeter()
    unmix.eval()
    
    save_dir = f"VALID_AUDIO_DEBUG_EPOCH{epoch:03d}"
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm.tqdm(valid_sampler, disable=args.quiet)
    with torch.no_grad():
        example_idx = 0

        for i, (x, y_v1, y_v2, bgm) in enumerate(pbar):
            x, y_v1, y_v2, bgm = [t.to(device) for t in (x, y_v1, y_v2, bgm)]
            y_total = y_v1 + y_v2  # target = combined vocals

            # --- Encode ---
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag = torch.abs(X_complex)
            X_phase = torch.angle(X_complex)

            # --- Predict ---
            Y_hat_mag = unmix(X_mag)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            Y_hat_real = torch.view_as_real(Y_hat_complex)
            y_hat_time = decoder(Y_hat_real)

            # --- Align lengths ---
            min_length = min(y_hat_time.size(-1), y_total.size(-1), bgm.size(-1))
            y_hat_time = y_hat_time[..., :min_length].mean(dim=1)
            y_total = y_total[..., :min_length].mean(dim=1)
            y_v1 = y_v1[..., :min_length].mean(dim=1)
            y_v2 = y_v2[..., :min_length].mean(dim=1)
            bgm = bgm[..., :min_length].mean(dim=1)
            mixture = x[..., :min_length].mean(dim=1)

            # --- Loss ---
            loss = singlesrc_neg_sisdr(y_hat_time, y_total).mean()
            losses.update(loss.item(), y_total.size(0))

            # --- Save first 10 examples for epoch 1 ---
            if example_idx < 10:
                prefix = f"{save_dir}/example_{example_idx:02d}"
                torchaudio.save(f"{prefix}_mixture.wav", mixture.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_pred_vocals.wav", y_hat_time.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_true_v1.wav", y_v1.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_true_v2.wav", y_v2.cpu(), sample_rate=44100)
                torchaudio.save(f"{prefix}_true_bgm.wav", bgm.cpu(), sample_rate=44100)
            else:
                break
            example_idx += 1

    print(f"✅ Validation audio saved (epoch {epoch}) to: {save_dir}")
    return losses.avg

def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # which target do we want to train?
    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned", # we choose this one
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )
    # HEREEEE, pre-trained model, umx
    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="vocals_NEW.wav",
        help="Embedding choice (clean vs. noisy)",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=1,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=8, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}
    # dataloader_kwargs = {}

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print('loading datasets')
    # train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    from openunmix.data import DuetAlignedDataset
    train_dataset = DuetAlignedDataset(root=args.root, split="train", seq_duration=args.seq_dur, random_chunks=True)
    valid_dataset = DuetAlignedDataset(root=args.root, split="valid", seq_duration=args.seq_dur)

    # create output dir if not exist
    target_path = Path(args.output) 
    target_path.mkdir(parents=True, exist_ok=True)
    print('Creating checkpoint directory', target_path)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    stft, istft = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate, center=True
    )
    encoder_for_statistics = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)
    encoder = torch.nn.Sequential(stft).to(device) 
    decoder = torch.nn.Sequential(istft).to(device)

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        # scaler_mean = None
        # scaler_std = None
        scaler_mean, scaler_std = get_statistics(args, encoder_for_statistics, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, args.nfft, args.bandwidth)

    if args.model:
        # HEREEEE
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        unmix = utils.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        unmix = unmix.to(device)
    else:
        unmix = model.OpenUnmix(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            max_bin=max_bin,
            unidirectional=args.unidirectional
        ).to(device)
        
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        unmix = torch.nn.DataParallel(unmix)

    # look into this
    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a checkpoint is specified: resume training
    if args.checkpoint:
        print('RESUME TRAINING')
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(args, unmix, encoder, decoder, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, encoder, decoder, device, valid_sampler)
        # valid_loss = valid_for_testing(args, unmix, encoder, decoder, device, valid_sampler) # if i wanna listen to valid samples
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break
        
def main_test():
    import argparse, torch
    from pathlib import Path
    from git import Repo
    from openunmix import model, transforms, utils
    from openunmix.data import AlignedDataset, DuetAlignedDataset

    parser = argparse.ArgumentParser(description="OpenUnmix Testing")

    parser.add_argument("--root", type=str, required=True, help="Root path of dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--target", type=str, default="vocals", help="Target source")
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nhop", type=int, default=1024)
    parser.add_argument("--nb-channels", type=int, default=2)
    parser.add_argument("--bandwidth", type=int, default=16000)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    # --- Device setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {use_cuda}")

    # --- Dataset ---
    print("Loading test dataset...")
    # test_dataset = AlignedDataset(root=args.root, split="test", seq_duration=None) # this is for solo testing
    test_dataset = DuetAlignedDataset(root=args.root, split="test", seq_duration=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # --- Encoder / Decoder ---
    stft, istft = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=test_dataset.sample_rate, center=True
    )
    encoder = torch.nn.Sequential(stft).to(device)
    decoder = torch.nn.Sequential(istft).to(device)

    # --- Model loading ---
    print("Loading checkpoint...")
    model_path = Path(args.checkpoint).expanduser()
    target_model_path = Path(model_path, args.target + ".pth")
    checkpoint = torch.load(target_model_path, map_location=device)

    unmix = model.OpenUnmix(
        nb_bins=args.nfft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=512,
        max_bin=utils.bandwidth_to_max_bin(test_dataset.sample_rate, args.nfft, args.bandwidth),
    ).to(device)
    unmix.load_state_dict(checkpoint, strict=False)
    
    # print("Evaluating on duet test set...")
    # (
    #     test_loss_v1,
    #     test_loss_v2,
    #     test_loss_v1v2,
    #     test_loss_bgm,
    #     test_loss_bgmv2
    # ) = valid_for_test_duet(args, unmix, encoder, decoder, device, test_loader)
    
    v, r, m, improv  = valid_for_test_duet_fad(unmix, encoder, decoder, device, test_loader)


    # print("\n✅ Duet Test Loss Summary:")
    # print(f"🎤 Vocals 1 (v1):       {test_loss_v1:.4f}")
    # print(f"🎤 Vocals 2 (v2):       {test_loss_v2:.4f}")
    # print(f"🎵 Vocals 1+2 (v1v2):   {test_loss_v1v2:.4f}")
    # print(f"🎧 Background (bgm):    {test_loss_bgm:.4f}")
    # print(f"🎶 Background+v2:        {test_loss_bgmv2:.4f}")
    
    # listen_one_example(args, unmix, encoder, decoder, device, test_loader)
    # return
    
    # save_first_test_outputs(unmix, encoder, decoder, device, test_loader)
 
def main_test_solo():
    import argparse, torch
    from pathlib import Path
    from openunmix import model, transforms, utils
    from openunmix.data import AlignedDataset
    import json

    parser = argparse.ArgumentParser(description="OpenUnmix Testing")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    # --- Load separator.json to recover training config ---
    sep_config_path = Path(args.checkpoint) / "separator.json"
    with open(sep_config_path, "r") as f:
        sep_conf = json.load(f)
    print("Loaded separator config:", sep_conf)

    nfft = sep_conf["nfft"]
    args.nfft = nfft
    nhop = sep_conf["nhop"]
    nb_channels = sep_conf["nb_channels"]
    sample_rate = sep_conf["sample_rate"]

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {use_cuda}")

    print(args.root)
    test_dataset = AlignedDataset(root=args.root, split="test", seq_duration=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    stft, istft = transforms.make_filterbanks(
        n_fft=nfft, n_hop=nhop, sample_rate=sample_rate, center=True
    )
    encoder = torch.nn.Sequential(stft).to(device)
    decoder = torch.nn.Sequential(istft).to(device)

    # --- Model loading ---
    target_model_path = Path(args.checkpoint, args.target + ".pth")
    checkpoint = torch.load(target_model_path, map_location=device)
    max_bin = utils.bandwidth_to_max_bin(sample_rate, nfft, 16000)

    unmix = model.OpenUnmix(
        nb_bins=nfft // 2 + 1,
        nb_channels=nb_channels,
        hidden_size=512,
        max_bin=max_bin,
    ).to(device)
    unmix.load_state_dict(checkpoint, strict=False)

    # --- Evaluation ---
    # test_loss_V, test_loss_R = valid_for_test(args, unmix, encoder, decoder, device, test_loader)
    # print(f"\n✅ Test Vocals Loss (V): {test_loss_V:.4f}")
    # print(f"✅ Test Residual Loss (R): {test_loss_R:.4f}")
    
    v, r, m, improv  = valid_for_test_solo_fad(args, unmix, encoder, decoder, device, test_loader)
    
# Decide which settings I am using
if __name__ == "__main__":
    main_test()
    # main_test_solo()

