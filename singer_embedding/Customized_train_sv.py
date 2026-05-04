import os
import pathlib
import json
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import wavfile
from pytorch_lightning import seed_everything
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from torch.utils.data import DataLoader

import Customized_DatasetSV as D
import Customized_NetSV as M

def ray_train_sv(config, use_ray: bool = True):
    try:
        # fix seed
        seed_everything(0)

        # select device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
            
        print('Device using:', device)

        network = M.NetSV(config['hidden_size'], config['num_layers']).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        print('START LOADING DATA')
        
        data_tr = D.DatasetCustomSV(
            audio_files=D.speaker_ids_tr, 
            segments_dict=D.segments_dict_tr,  
            utterance_duration=config['utterance_duration']
        )
        
        dl_tr = DataLoader(data_tr, batch_size=config['batch_size'])
        
        data_vl = D.DatasetCustomSV(
            audio_files=D.speaker_ids_vl, 
            segments_dict=D.segments_dict_vl, 
            utterance_duration=config['utterance_duration']
        )
        
        print('FINISH LOADING DATA')
        
        (vx_1, vx_2, vy) = next(iter(DataLoader(data_vl, batch_size=config['batch_size'])))
        vx_1 = vx_1.to(device)
        vx_2 = vx_2.to(device)
        vy = vy.to(device)

        # setup metrics + state dict
        (num_batches, current_epoch, best_epoch) = (0, 0, 0)
        best_result: float = 1e10
        best_accuracy: float = 0.
        best_state_dict = None
        errors, accuracies = [], []

        # loop indefinitely
        for (x_1, x_2, y) in dl_tr:

            current_epoch += 1
            num_batches += config['batch_size']

            optimizer.zero_grad()
            x_1 = x_1.to(device)
            x_2 = x_2.to(device)
            y = y.to(device)
            y_hat = network(x_1, x_2)
            
            train_loss = criterion(y_hat, y.squeeze())
            train_acc = (torch.round(torch.sigmoid(y_hat)) == y.squeeze()).float().mean().item()
            train_loss.backward()
            
            # Gradient clipping
            # max_norm = 1.0 
            # torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
            
            optimizer.step()
            
            # only validate every few epochs
            if current_epoch % 10:
                continue

            # validate
            with torch.no_grad():
                vy_hat = network(vx_1, vx_2)
                vl_loss = float(criterion(vy_hat, vy.squeeze()).item())
                vl_acc = float((torch.round(torch.sigmoid(vy_hat)) == vy.squeeze()).float().mean().item())
                errors.append(vl_loss)
                accuracies.append(vl_acc)

            if (current_epoch == 10) or (vl_loss < best_result):
                best_epoch = current_epoch
                best_result = vl_loss
                best_accuracy = vl_acc
                best_state_dict = network.state_dict()

                # if use_ray:
                # # save checkpoint using Ray Tune
                #     with tune.checkpoint_dir(current_epoch) as checkpoint_dir:
                #         path = os.path.join(checkpoint_dir, 'checkpoint')
                #         torch.save(network.state_dict(), path)
                #         path = os.path.join(checkpoint_dir, 'errors.json')
                #         with open(path, 'w') as fp:
                #             json.dump(errors, fp)
                #         path = os.path.join(checkpoint_dir, 'accuracies.json')
                #         with open(path, 'w') as fp:
                #             json.dump(accuracies, fp)
                
                # i feel like I do not need to do this..           
                # checkpoint_dir = f"./checkpoints/epoch_{current_epoch}"
                # os.makedirs(checkpoint_dir, exist_ok=True)

                # try:
                #     # Save the model
                #     torch.save(network.state_dict(), os.path.join(checkpoint_dir, 'checkpoint.pth'))

                #     # Save metrics
                #     with open(os.path.join(checkpoint_dir, 'errors.json'), 'w') as fp:
                #         json.dump(errors, fp)
                #     with open(os.path.join(checkpoint_dir, 'accuracies.json'), 'w') as fp:
                #         json.dump(accuracies, fp)

                # except Exception as e:
                #     print(f"Error saving checkpoint: {e}")
                    
            # ray.train.report(dict(num_batches=int(num_batches), vl_loss=float(best_result), vl_acc=float(best_accuracy)))
            # tune.report(num_batches=num_batches, vl_loss=best_result, vl_acc=best_accuracy)
            ray.train.report(dict(
                num_batches=int(num_batches),
                vl_loss=float(best_result),
                vl_acc=float(best_accuracy),
                tr_loss=float(train_loss.item()),  # Average train loss
                tr_acc=float(train_acc)  # Average train accuracy
            ))

            # check for convergence
            if current_epoch - best_epoch > 1000:
                break

        print('exited train_sv with args = {{' + \
            ', '.join([f'{k}={v}' for (k,v) in config.items()]) + '}}')
        
        # After the training loop and before returning
        final_model_path = f"model_{best_result}.pth"  # Specify the path where you want to save
        torch.save(best_state_dict, final_model_path)
        print(f"Best model saved to {final_model_path}")

        return {
            'state_dict': best_state_dict,
            'num_batches': best_epoch * int(config['batch_size']),
            'vl_loss': float(best_result),
            'vl_acc': float(best_accuracy),
        }
        
    except Exception as e:
        print(f"Trial failed due to: {str(e)}")
        raise e

def main(num_gpus: float = .25):
    config = {
        'utterance_duration': 3.0,
        'hidden_size': 32,
        'learning_rate': tune.qloguniform(1e-4, 1e-2, 1e-6), #tune.qloguniform(1e-4, 1e-3, 1e-6), #, 1e-2,1e-5,1e-6
        'num_layers': 2,
        'batch_size': 128, #128
    }
    scheduler = ASHAScheduler(
        metric='vl_loss',
        mode='min',
        max_t=10000,
        grace_period=100,
        reduction_factor=2
    )
    # scheduler = None
    result = tune.run(
        ray_train_sv,
        config=config,
        name='ray_train_sv_find_lr',
        keep_checkpoints_num=1,
        log_to_file='log.txt',
        progress_reporter=CLIReporter(
            max_report_frequency=50,
            metric_columns=['num_batches', 'vl_loss', 'vl_acc', 'tr_loss', 'tr_acc'],
            parameter_columns=['learning_rate'],
        ),
        num_samples=16, #16
        scheduler=scheduler,
        resources_per_trial={'gpu': num_gpus},
        verbose=1
    )
    best_trial = result.get_best_trial(
        metric='vl_loss',
        mode='min',
        scope='last'
    )
    print('Best trial config: {}'.format(
        best_trial.config))
    print('Best trial final test results: {}'.format(
        best_trial.last_result['vl_acc']))
    print('Best trial checkpoint path: {}'.format(
        os.path.join(best_trial.checkpoint.value, "checkpoint")))


if __name__ == '__main__':
    main()