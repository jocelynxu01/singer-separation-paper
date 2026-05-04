import argparse
import torch
import time
import json
import random
import copy
import tqdm
import numpy as np
from pathlib import Path
from git import Repo
from openunmix import data, model, utils, transforms
from asteroid.losses import singlesrc_neg_sisdr
from sklearn.preprocessing import StandardScaler
import os
import sklearn.preprocessing
import torchaudio
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
import torchaudio.functional as AF

def load_precalculated_embeddings(file_paths, embeddings_mapping, device):
    embeddings_batch = []
    valid_file_paths = []

    for file_path in file_paths:
        entry = embeddings_mapping.get(file_path)
        if entry is None:
            print(f"⚠️ Skipping {file_path}: no embedding found.")
            continue

        # Handle both dict and string cases
        if isinstance(entry, dict):
            embedding_path = entry.get("embedding_path")
        else:
            embedding_path = entry

        if embedding_path is None or not os.path.exists(embedding_path):
            print(f"⚠️ Skipping {file_path}: missing embedding file {embedding_path}")
            continue

        try:
            embedding = torch.load(embedding_path, map_location=device).to(device)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: failed to load embedding ({e})")
            continue

        embeddings_batch.append(embedding)
        valid_file_paths.append(file_path)

    if len(embeddings_batch) == 0:
        raise RuntimeError("❌ No valid embeddings found for this batch!")

    # Concatenate along batch dimension
    return torch.cat(embeddings_batch, dim=0)
    # return torch.cat(embeddings_batch, dim=0).unsqueeze(1).expand(-1, 1, -1)



# -------------------------------------------------------------------------
# Training / Validation loops
# -------------------------------------------------------------------------
def train(args, unmix, encoder, decoder, device, train_loader, optimizer, emb_map):
    losses_total = utils.AverageMeter()
    losses_vocal = utils.AverageMeter()
    losses_resid = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_loader, desc="Training", disable=args.quiet)
    
    for x, y_v1, y_v2, bgm, v1_paths in pbar:
    
        x, y_v1, y_v2 = x.to(device), y_v1.to(device), y_v2.to(device)
        y_target = y_v1  # only v1 is target now
        optimizer.zero_grad()

        embeddings = load_precalculated_embeddings(v1_paths, emb_map, device)
        X = encoder(x)
        X_mag = torch.abs(torch.view_as_complex(X))
        X_phase = torch.angle(torch.view_as_complex(X))

        Y_hat_mag = unmix(X_mag, embeddings)
        Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
        y_hat_time = decoder(torch.view_as_real(Y_hat_complex))
        
        min_len = min(y_hat_time.size(-1), y_target.size(-1))
        y_hat_time = y_hat_time[..., :min_len].mean(dim=1)
        y_target = y_target[..., :min_len].mean(dim=1)
        r_target = x[..., :min_len].mean(dim=1) - y_target
        r_hat = x[..., :min_len].mean(dim=1) - y_hat_time

        loss_vocal = singlesrc_neg_sisdr(y_hat_time, y_target).mean()
        loss_residual = singlesrc_neg_sisdr(r_hat, r_target).mean()
        
        lam = args.lam
        loss = loss_vocal + lam * loss_residual
        
        loss.backward()
        optimizer.step()

        losses_total.update(loss.item(), y_target.size(0))
        losses_vocal.update(loss_vocal.item(), y_target.size(0))
        losses_resid.update(loss_residual.item(), y_target.size(0))
        pbar.set_postfix(
            total=f"{losses_total.avg:.3f}",
            v=f"{losses_vocal.avg:.3f}",
            r=f"{losses_resid.avg:.3f}",
        )
    # return losses_total.avg
    return losses_total.avg, losses_vocal.avg, losses_resid.avg


def validate(args, unmix, encoder, decoder, device, valid_loader, emb_map):
    losses_total = utils.AverageMeter()
    losses_vocal = utils.AverageMeter()
    losses_resid = utils.AverageMeter()
    unmix.eval()
    pbar = tqdm.tqdm(valid_loader, desc="Validating", disable=args.quiet)

    with torch.no_grad():
        for x, y_v1, y_v2, bgm, v1_paths in pbar:
            x, y_v1, y_v2 = x.to(device), y_v1.to(device), y_v2.to(device)
            y_target = y_v1

            embeddings = load_precalculated_embeddings(v1_paths, emb_map, device)
            X = encoder(x)
            X_mag = torch.abs(torch.view_as_complex(X))
            X_phase = torch.angle(torch.view_as_complex(X))

            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y_hat_time = decoder(torch.view_as_real(Y_hat_complex))

            min_len = min(y_hat_time.size(-1), y_target.size(-1))
            y_hat_time = y_hat_time[..., :min_len].mean(dim=1)
            y_target = y_target[..., :min_len].mean(dim=1)
            r_target = x[..., :min_len].mean(dim=1) - y_target
            r_hat = x[..., :min_len].mean(dim=1) - y_hat_time

            loss_vocal = singlesrc_neg_sisdr(y_hat_time, y_target).mean()
            loss_residual = singlesrc_neg_sisdr(r_hat, r_target).mean()
            
            lam = args.lam
            loss = loss_vocal + lam * loss_residual

            losses_total.update(loss.item(), y_target.size(0))
            losses_vocal.update(loss_vocal.item(), y_target.size(0))
            losses_resid.update(loss_residual.item(), y_target.size(0))
            pbar.set_postfix(
                total=f"{losses_total.avg:.3f}",
                v=f"{losses_vocal.avg:.3f}",
                r=f"{losses_resid.avg:.3f}",
            )
    # return losses_total.avg
    return losses_total.avg, losses_vocal.avg, losses_resid.avg


def valid_for_test_duet_emb(unmix, encoder, decoder, device, duet_loader, emb_map_test):
    """
    Evaluate embedding-conditioned model on the duet test set and save loss summary.
    """
    losses = {k: utils.AverageMeter() for k in ["v1", "v2", "v1v2", "bgm", "bgmv2"]}
    unmix.eval()

    with torch.no_grad():
        for x, y_v1, y_v2, bgm, v1_paths in tqdm.tqdm(duet_loader, desc="Evaluating Duet Test (with embeddings)"):
            x, y_v1, y_v2, bgm = [t.to(device) for t in (x, y_v1, y_v2, bgm)]

            # --- Load embeddings for this batch ---
            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # --- Forward pass ---
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)
            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))

            # --- Align and downmix ---
            min_len = min(y1_hat.size(-1), y_v1.size(-1), y_v2.size(-1), bgm.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            y_v2 = y_v2[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            residual = x[..., :min_len].mean(dim=1) - y1_hat

            # --- Compute SI-SDR losses ---
            losses["v1"].update(singlesrc_neg_sisdr(y1_hat, y_v1).mean().item())
            losses["v2"].update(singlesrc_neg_sisdr(y1_hat, y_v2).mean().item())
            losses["v1v2"].update(singlesrc_neg_sisdr(y1_hat, y_v1 + y_v2).mean().item())
            losses["bgm"].update(singlesrc_neg_sisdr(residual, bgm).mean().item())
            losses["bgmv2"].update(singlesrc_neg_sisdr(residual, bgm + y_v2).mean().item())

    # --- Print summary ---
    print("\n--- Duet Test Loss Summary (Embedding Model) ---")
    for k, v in losses.items():
        print(f"{k}: {v.avg:.4f}")

    return tuple(v.avg for v in losses.values())

def valid_for_test_duet_emb_fad(unmix, encoder, decoder, device, duet_loader, emb_map_test):
    """
    Evaluate embedding-conditioned model on the duet test set using FAD.
    """
    unmix.eval()

    tmp = tempfile.mkdtemp()
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

    try:
        with torch.no_grad():
            idx = 0
            skipped = 0

            for x, y_v1, y_v2, bgm, v1_paths in tqdm.tqdm(
                duet_loader, desc="Evaluating DUET Test (FAD, with film embedding model)"
            ):
                x = x.to(device)
                y_v1 = y_v1.to(device)
                y_v2 = y_v2.to(device)
                bgm = bgm.to(device)

                embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

                # forward
                X = encoder(x)
                X_complex = torch.view_as_complex(X)
                X_mag = torch.abs(X_complex)
                X_phase = torch.angle(X_complex)

                Y_hat_mag = unmix(X_mag, embeddings)
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
                    # convert to numpy
                    pred_v1 = y1_hat[0].detach().cpu().float().squeeze().numpy()
                    ref_v1 = y_v1[0].detach().cpu().float().squeeze().numpy()
                    pred_residual = residual[0].detach().cpu().float().squeeze().numpy()
                    ref_residual = ref_bgmv2[0].detach().cpu().float().squeeze().numpy()
                    mixture = x_mono[0].detach().cpu().float().squeeze().numpy()

                    # write wavs
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

        # compute FAD
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

        print("\n--- DUET Test FAD Summary (film Embedding Model) ---")
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

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def valid_for_test_solo_emb(unmix, encoder, decoder, device, solo_loader, emb_map_test):
    """
    Evaluate embedding-conditioned model on SOLO test set.
    Expects batches: (x, y_v1, bgm, v1_paths)
      - x: mixture
      - y_v1: vocal1 target
      - bgm: background target (for residual metric)
      - v1_paths: list[str] keys into emb_map_test (usually vocal1.wav paths)
    """
    losses = {k: utils.AverageMeter() for k in ["v1", "bgm"]}
    unmix.eval()

    with torch.no_grad():
        for x, y_v1, v1_paths, bgm in tqdm.tqdm(solo_loader, desc="Evaluating SOLO Test (with embeddings)"):
            x = x.to(device)
            y_v1 = y_v1.to(device)
            bgm = bgm.to(device)

            # embeddings keyed by whatever you pass in v1_paths
            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # Forward
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)

            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))
            
            # Align lengths + downmix channel dim
            min_len = min(y1_hat.size(-1), y_v1.size(-1), bgm.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            residual = x[..., :min_len].mean(dim=1) - y1_hat

            losses["v1"].update(singlesrc_neg_sisdr(y1_hat, y_v1).mean().item())
            losses["bgm"].update(singlesrc_neg_sisdr(residual, bgm).mean().item())
            

    print("\n--- SOLO Test Loss Summary (Embedding Model) ---")
    for k, v in losses.items():
        print(f"{k}: {v.avg:.4f}")

    return losses["v1"].avg, losses["bgm"].avg


def valid_for_test_solo_emb_dnsmos(unmix, encoder, decoder, device, solo_loader, emb_map_test):
    """
    Evaluate embedding-conditioned model on SOLO test set using DNSMOS
    """
    scores = {
        "v1": [utils.AverageMeter() for _ in range(4)],
        "bgm": [utils.AverageMeter() for _ in range(4)],
    }
    unmix.eval()

    with torch.no_grad():
        for x, y_v1, v1_paths, bgm in tqdm.tqdm(
            solo_loader, desc="Evaluating SOLO Test (DNSMOS)"
        ):
            x = x.to(device)
            y_v1 = y_v1.to(device)
            bgm = bgm.to(device)

            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # Forward
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)

            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))

            # Align lengths + downmix
            min_len = min(y1_hat.size(-1), y_v1.size(-1), bgm.size(-1), x.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            x_mono = x[..., :min_len].mean(dim=1)
            residual = x_mono - y1_hat

            # ---- DNSMOS ----
            v1_score = deep_noise_suppression_mean_opinion_score(
                y1_hat, 44100, False, device=device
            )
            bgm_score = deep_noise_suppression_mean_opinion_score(
                residual, 44100, False, device=device
            )

            v1_vals = v1_score.squeeze(0)   # shape [4]
            bgm_vals = bgm_score.squeeze(0)

            for i in range(4):
                scores["v1"][i].update(v1_vals[i].item())
                scores["bgm"][i].update(bgm_vals[i].item())
                
    labels = ["p808", "sig", "bak", "ovr"]

    print("\n--- SOLO Test DNSMOS Summary ---")
    for i, name in enumerate(labels):
        print(f"v1_{name}: {scores['v1'][i].avg:.4f}")
    for i, name in enumerate(labels):
        print(f"bgm_{name}: {scores['bgm'][i].avg:.4f}")

    v1_avg = [m.avg for m in scores["v1"]]
    bgm_avg = [m.avg for m in scores["bgm"]]

    return v1_avg, bgm_avg

def valid_for_test_solo_emb_dnsmos_with_originals(unmix, encoder, decoder, device, solo_loader, emb_map_test):
    """
    Evaluate embedding-conditioned model on SOLO test set using DNSMOS
    """
    scores = {
        "v1": [utils.AverageMeter() for _ in range(4)],
        "bgm": [utils.AverageMeter() for _ in range(4)],
        "v1_orig": [utils.AverageMeter() for _ in range(4)],
        "bgm_orig": [utils.AverageMeter() for _ in range(4)],
    }
    unmix.eval()

    with torch.no_grad():
        for x, y_v1, v1_paths, bgm in tqdm.tqdm(
            solo_loader, desc="Evaluating SOLO Test (DNSMOS)"
        ):
            x = x.to(device)
            y_v1 = y_v1.to(device)
            bgm = bgm.to(device)

            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # Forward
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)

            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))

            # Align lengths + downmix
            min_len = min(y1_hat.size(-1), y_v1.size(-1), bgm.size(-1), x.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            x_mono = x[..., :min_len].mean(dim=1)
            residual = x_mono - y1_hat

            # ---- DNSMOS ----
            v1_score = deep_noise_suppression_mean_opinion_score(
                y1_hat, 44100, False, device=device
            )
            bgm_score = deep_noise_suppression_mean_opinion_score(
                residual, 44100, False, device=device
            )
            v1_orig_score = deep_noise_suppression_mean_opinion_score(
                y_v1, 44100, False, device=device
            )
            bgm_orig_score = deep_noise_suppression_mean_opinion_score(
                bgm, 44100, False, device=device
            )

            v1_vals = v1_score.squeeze(0)         # shape [4]
            bgm_vals = bgm_score.squeeze(0)
            v1_orig_vals = v1_orig_score.squeeze(0)
            bgm_orig_vals = bgm_orig_score.squeeze(0)

            for i in range(4):
                scores["v1"][i].update(v1_vals[i].item())
                scores["bgm"][i].update(bgm_vals[i].item())
                scores["v1_orig"][i].update(v1_orig_vals[i].item())
                scores["bgm_orig"][i].update(bgm_orig_vals[i].item())

    labels = ["p808", "sig", "bak", "ovr"]

    print("\n--- SOLO Test DNSMOS Summary ---")
    for i, name in enumerate(labels):
        print(f"v1_{name}: {scores['v1'][i].avg:.4f}")
    for i, name in enumerate(labels):
        print(f"bgm_{name}: {scores['bgm'][i].avg:.4f}")
    for i, name in enumerate(labels):
        print(f"v1_orig_{name}: {scores['v1_orig'][i].avg:.4f}")
    for i, name in enumerate(labels):
        print(f"bgm_orig_{name}: {scores['bgm_orig'][i].avg:.4f}")

    v1_avg = [m.avg for m in scores["v1"]]
    bgm_avg = [m.avg for m in scores["bgm"]]
    v1_orig_avg = [m.avg for m in scores["v1_orig"]]
    bgm_orig_avg = [m.avg for m in scores["bgm_orig"]]

    return v1_avg, bgm_avg, v1_orig_avg, bgm_orig_avg


import os
import tempfile
import shutil
import soundfile as sf
from frechet_audio_distance import FrechetAudioDistance

def valid_for_test_solo_emb_fad(unmix, encoder, decoder, device, solo_loader, emb_map_test):
    unmix.eval()

    tmp = tempfile.mkdtemp()
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

    try:
        with torch.no_grad():
            idx = 0

            for x, y_v1, v1_paths, bgm in tqdm.tqdm(
                solo_loader, desc="Evaluating SOLO Test (FAD)"
            ):
                x = x.to(device)
                y_v1 = y_v1.to(device)
                bgm = bgm.to(device)

                embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

                # forward
                X = encoder(x)
                X_complex = torch.view_as_complex(X)
                X_mag = torch.abs(X_complex)
                X_phase = torch.angle(X_complex)

                Y_hat_mag = unmix(X_mag, embeddings)
                Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
                y1_hat = decoder(torch.view_as_real(Y_hat_complex))

                # align lengths, then downmix to mono
                min_len = min(y1_hat.size(-1), y_v1.size(-1), bgm.size(-1), x.size(-1))
                y1_hat = y1_hat[..., :min_len].mean(dim=1)
                y_v1 = y_v1[..., :min_len].mean(dim=1)
                bgm = bgm[..., :min_len].mean(dim=1)
                x_mono = x[..., :min_len].mean(dim=1)

                residual = x_mono - y1_hat

                # batch size = 1
                pred_v1 = y1_hat[0].cpu().numpy()
                ref_v1 = y_v1[0].cpu().numpy()
                pred_bgm = residual[0].cpu().numpy()
                ref_bgm = bgm[0].cpu().numpy()
                mixture = x_mono[0].cpu().numpy()

                sf.write(os.path.join(pred_v1_dir, f"{idx}.wav"), pred_v1, 44100)
                sf.write(os.path.join(ref_v1_dir, f"{idx}.wav"), ref_v1, 44100)
                sf.write(os.path.join(pred_bgm_dir, f"{idx}.wav"), pred_bgm, 44100)
                sf.write(os.path.join(ref_bgm_dir, f"{idx}.wav"), ref_bgm, 44100)
                sf.write(os.path.join(mixture_dir, f"{idx}.wav"), mixture, 44100)

                idx += 1

                # if idx >= 20:
                #     break

        # compute FAD
        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            verbose=False,
        )

        # original metrics
        vocal_fad = fad.score(ref_v1_dir, pred_v1_dir)
        bgm_fad = fad.score(ref_bgm_dir, pred_bgm_dir)
        # new baseline comparison
        mixture_vocal_fad = fad.score(ref_v1_dir, mixture_dir)
        vocal_fad_improvement = mixture_vocal_fad - vocal_fad

        print("\n--- SOLO Test FAD Summary ---")
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

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_first_test_outputs(unmix, encoder, decoder, device, test_loader, emb_map_test, save_dir="TEST_AUDIO_DEBUG_EMB_FILM_DUELLOSSES0.2"):
    os.makedirs(save_dir, exist_ok=True)
    unmix.eval()

    with torch.no_grad():

        example_idx = 0

        for i, (x, y_v1, y_v2, bgm, v1_paths) in enumerate(tqdm.tqdm(test_loader, desc="Evaluating Duet Test")):
            x, y_v1, y_v2, bgm = [t.to(device) for t in (x, y_v1, y_v2, bgm)]
            
            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # --- Forward pass ---
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)
            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))

            # --- Align and downmix ---
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
    
def save_first_test_outputs_solo(
    unmix, encoder, decoder, device, test_loader, emb_map_test,
    save_dir="TEST_AUDIO_DEBUG_SOLO_FILM_EMB_0.2", max_examples=10, sample_rate=44100
):
    import os
    import tqdm
    import torchaudio

    os.makedirs(save_dir, exist_ok=True)
    unmix.eval()

    with torch.no_grad():
        example_idx = 0

        for i, batch in enumerate(tqdm.tqdm(test_loader, desc="Evaluating SOLO Test")):
            # Your SOLO dataset currently returns: X, Y, emb_path_str, B
            x, y_v1, v1_paths, bgm = batch

            x = x.to(device)
            y_v1 = y_v1.to(device)
            bgm = bgm.to(device)

            embeddings = load_precalculated_embeddings(v1_paths, emb_map_test, device)

            # --- Forward pass ---
            X = encoder(x)
            X_complex = torch.view_as_complex(X)
            X_mag, X_phase = torch.abs(X_complex), torch.angle(X_complex)
            Y_hat_mag = unmix(X_mag, embeddings)
            Y_hat_complex = Y_hat_mag * torch.exp(1j * X_phase)
            y1_hat = decoder(torch.view_as_real(Y_hat_complex))

            # --- Align and downmix ---
            min_len = min(y1_hat.size(-1), y_v1.size(-1), bgm.size(-1), x.size(-1))
            y1_hat = y1_hat[..., :min_len].mean(dim=1)
            y_v1 = y_v1[..., :min_len].mean(dim=1)
            bgm = bgm[..., :min_len].mean(dim=1)
            mixture = x[..., :min_len].mean(dim=1)

            residual = mixture - y1_hat

            # --- Save audio examples ---
            if example_idx < max_examples:
                prefix = f"{save_dir}/example_{example_idx:02d}"
                torchaudio.save(f"{prefix}_mixture.wav", mixture.cpu(), sample_rate=sample_rate)
                torchaudio.save(f"{prefix}_vocal1_hat.wav", y1_hat.cpu(), sample_rate=sample_rate)
                torchaudio.save(f"{prefix}_vocal1_true.wav", y_v1.cpu(), sample_rate=sample_rate)
                torchaudio.save(f"{prefix}_bgm_true.wav", bgm.cpu(), sample_rate=sample_rate)
                torchaudio.save(f"{prefix}_residual.wav", residual.cpu(), sample_rate=sample_rate)
            else:
                break

            example_idx += 1

    print(f"✅ Saved SOLO mixture + sources to: {save_dir}")

# -------------------------------------------------------------------------
# Compute normalization statistics (optional)
# -------------------------------------------------------------------------
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
        x, y, e, b, _ = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std

# -------------------------------------------------------------------------
# Main training entry
# -------------------------------------------------------------------------
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
    
    parser.add_argument("--lam", type=float, default=0.1)

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {use_cuda}")

    # Dataset
    from openunmix.data import DuetAlignedDatasetWithEmbeddings
    
    # Load embeddings map
    save_dir_train = "DUET_clean_embeddings/train"
    with open(os.path.join(save_dir_train, "embeddings_mapping_train.json"), "r") as f:
        emb_map_train = json.load(f)
    
    save_dir_valid = "DUET_clean_embeddings/valid"
    with open(os.path.join(save_dir_valid, "embeddings_mapping_valid.json"), "r") as f:
        emb_map_valid = json.load(f)
        
    print("Loading datasets...")
    train_dataset = DuetAlignedDatasetWithEmbeddings(
        root=args.root, split="train", seq_duration=args.seq_dur, emb_map=emb_map_train
    )
    valid_dataset = DuetAlignedDatasetWithEmbeddings(
        root=args.root, split="valid", seq_duration=args.seq_dur, emb_map=emb_map_valid
    )

    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Encoder/decoder
    stft, istft = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate, center=True
    )
    encoder = torch.nn.Sequential(stft).to(device)
    decoder = torch.nn.Sequential(istft).to(device)
    encoder_for_statistics = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)
    
    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    # Output dir
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {target_path}")
    
    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    # Compute scaler stats
    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        # scaler_mean = None
        # scaler_std = None
        scaler_mean, scaler_std = get_statistics(args, encoder_for_statistics, train_dataset)

    # Model
    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, args.nfft, args.bandwidth)
    unmix = model.OpenUnmix_FiLM(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_bins=args.nfft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        max_bin=max_bin,
        unidirectional=args.unidirectional,
        emb_dim=32
    ).to(device)
    

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, cooldown=10
    )
    es = utils.EarlyStopping(patience=args.patience)

    # Training loop
    repo = Repo(Path(__file__).resolve().parents[1])
    commit = repo.head.commit.hexsha[:7]
    train_losses, valid_losses, train_times = [], [], []
    train_losses_vocal = []
    train_losses_resid = []
    valid_losses_vocal = []
    valid_losses_resid = []
    best_epoch = 0

    for epoch in tqdm.trange(1, args.epochs + 1, desc="Training epochs"):
        start_time = time.time()
        train_loss, train_loss_vocal, train_loss_residual = train(args, unmix, encoder, decoder, device, train_loader, optimizer, emb_map_train)
        valid_loss, valid_loss_vocal, valid_loss_residual = validate(args, unmix, encoder, decoder, device, valid_loader, emb_map_valid)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_losses_vocal.append(train_loss_vocal)
        train_losses_resid.append(train_loss_residual)
        valid_losses_vocal.append(valid_loss_vocal)
        valid_losses_resid.append(valid_loss_residual)
        train_times.append(time.time() - start_time)

        stop = es.step(valid_loss)
        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target="vocals",
        )

        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "train_loss_vocal_history": train_losses_vocal,
            "train_loss_resid_history": train_losses_resid,
            "valid_loss_history": valid_losses,
            "valid_loss_vocal_history": valid_losses_vocal,
            "valid_loss_resid_history": valid_losses_resid,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }
        
        with open(target_path / "vocals.json", "w") as f:
            json.dump(params, f, indent=4, sort_keys=True)

        # print(f"Epoch {epoch}: Train {train_loss:.3f} | Valid {valid_loss:.3f}")
        print(
            f"Epoch {epoch}: "
            f"Train total={train_loss:.3f} v={train_loss_vocal:.3f} r={train_loss_residual:.3f} | "
            f"Valid total={valid_loss:.3f} v={valid_loss_vocal:.3f} r={valid_loss_residual:.3f}"
        )
        if stop:
            print("Early stopping triggered.")
            break
        
        
def main_test_with_embeddings():
    import argparse, json
    from pathlib import Path
    from openunmix import model, transforms, utils
    from openunmix.data import DuetAlignedDatasetWithEmbeddings

    parser = argparse.ArgumentParser(description="Test OpenUnmix with embeddings")
    parser.add_argument("--root", type=str, required=True, help="Root path of dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nhop", type=int, default=1024)
    parser.add_argument("--bandwidth", type=int, default=16000)
    parser.add_argument("--nb-channels", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {use_cuda}")

    # --- Load dataset & embeddings ---
    save_dir_test = "DUET_clean_embeddings/test"
    with open(os.path.join(save_dir_test, "embeddings_mapping_test.json"), "r") as f:
        emb_map_test = json.load(f)
        
    print("Loading test dataset...")
    test_dataset = DuetAlignedDatasetWithEmbeddings(
        root=args.root, split="test", seq_duration=None, emb_map=emb_map_test
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Encoder / Decoder ---
    stft, istft = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=test_dataset.sample_rate, center=True
    )
    encoder = torch.nn.Sequential(stft).to(device)
    decoder = torch.nn.Sequential(istft).to(device)

    # --- Model loading ---
    print("Loading model checkpoint...")
    model_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(Path(model_path, f"{args.target}.pth"), map_location=device)

    unmix = model.OpenUnmix_FiLM(
        nb_bins=args.nfft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=512,
        max_bin=utils.bandwidth_to_max_bin(test_dataset.sample_rate, args.nfft, args.bandwidth),
        emb_dim=32,
    ).to(device)
    unmix.load_state_dict(checkpoint, strict=False) #["state_dict"]
    unmix.eval()

    # --- Evaluate ---
    print("Evaluating on test set with embeddings...")
    # test_loss = valid_for_test_duet_emb(unmix, encoder, decoder, device, test_loader, emb_map_test)
    # print(f"\n🎯 Final Test Loss: {test_loss:.4f}")
    
    # v1, v2, v1v2, bgm, bgmv2 = valid_for_test_duet_emb(
    # unmix, encoder, decoder, device, test_loader, emb_map_test
    # )
    v, r, m, improv = valid_for_test_duet_emb_fad(
    unmix, encoder, decoder, device, test_loader, emb_map_test
    )
    

    # print(
    #     f"\n🎯 Test negSI-SDR (lower is better): "
    #     f"v1={v1:.4f} v2={v2:.4f} v1v2={v1v2:.4f} bgm={bgm:.4f} bgmv2={bgmv2:.4f}"
    # )
    
    # save_first_test_outputs(unmix, encoder, decoder, device, test_loader, emb_map_test)

def main_test_with_solo_embeddings():
    import argparse, json
    from pathlib import Path
    from openunmix import model, transforms, utils
    from openunmix.data import AlignedDataset_Embeddings_Background

    parser = argparse.ArgumentParser(description="Test OpenUnmix with embeddings")
    parser.add_argument("--root", type=str, required=True, help="Root path of dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nhop", type=int, default=1024)
    parser.add_argument("--bandwidth", type=int, default=16000)
    parser.add_argument("--nb-channels", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {use_cuda}")

    # --- Load dataset & embeddings ---
    save_dir_test = "SOLO_clean_embeddings/test"
    with open(os.path.join(save_dir_test, "embeddings_mapping_test.json"), "r") as f:
        emb_map_test = json.load(f)
        
    print("Loading test dataset...")
    test_dataset = AlignedDataset_Embeddings_Background(
        root=args.root, split="test", seq_duration=None, emb_map=emb_map_test
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Encoder / Decoder ---
    stft, istft = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=test_dataset.sample_rate, center=True
    )
    encoder = torch.nn.Sequential(stft).to(device)
    decoder = torch.nn.Sequential(istft).to(device)

    # --- Model loading ---
    print("Loading model checkpoint...")
    model_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(Path(model_path, f"{args.target}.pth"), map_location=device)

    unmix = model.OpenUnmix_FiLM(
        nb_bins=args.nfft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=512,
        max_bin=utils.bandwidth_to_max_bin(test_dataset.sample_rate, args.nfft, args.bandwidth),
        emb_dim=32,
    ).to(device)
    unmix.load_state_dict(checkpoint, strict=False) #["state_dict"]
    unmix.eval()

    # --- Evaluate ---
    print("Evaluating on test set with embeddings...")
    # test_loss = valid_for_test_solo_emb(unmix, encoder, decoder, device, test_loader, emb_map_test)
    # test_loss = valid_for_test_solo_emb_dnsmos(unmix, encoder, decoder, device, test_loader, emb_map_test)
    test_loss = valid_for_test_solo_emb_fad(unmix, encoder, decoder, device, test_loader, emb_map_test)

    # print(f"\n🎯 Final Test Loss - v1: {test_loss[0]:.4f} | bgm: {test_loss[1]:.4f}")
    
    # save_first_test_outputs_solo(unmix, encoder, decoder, device, test_loader, emb_map_test)



if __name__ == "__main__":
    main_test_with_embeddings()