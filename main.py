import os
import copy
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml


from os.path import join

from omegaconf import OmegaConf

from dataloaders import initialize_data_functions, get_evaluation_loaders
from utils.logging import print_header, print_args, print_config
from optimizer import get_optimizer, get_scheduler
from loss import get_loss
from data_transforms import get_data_transforms
from train import train_model, evaluate_model, plot_forecasts

from setup import format_arg, seed_everything
from setup import initialize_args
from setup import load_model_config, load_main_config
from setup import initialize_experiment
from setup.configs.model import update_output_config_from_args  # For multivariate feature prediction

from model.network import SpaceTime


def visualize_and_log(key, context, predicted_output, ground_truth, context_length, prediction_length, plot_count,
                      logger=None):
    # Concatenate predicted output and ground truth across batches
    # predicted_output = np.concatenate(self.data[key]['predictions'], axis=0)
    # ground_truth = np.concatenate(self.data[key]['ground_truths'], axis=0)
    # context = np.concatenate(self.data[key]['contexts'], axis=0)[:, :context_length] # check dim for traffic set.. this should be 2 dimesnional not 3

    # Create a plot that displays both the predicted values and the ground truth values for the feature

    # specific for robot data when obs is not part of target
    dims = [0, 4]
    for dim in dims:
        for i in np.random.randint(0, predicted_output.shape[0], plot_count):
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Plot the first array on the left side
            ax.plot(range(context_length), context[i, :context_length, dim], label='Context')

            # Plot the other two arrays on the right side
            begin_double_plot = (context_length - 1)
            end_double_plot = (context_length + prediction_length)
            ax.plot(range(begin_double_plot, end_double_plot),
                    np.concatenate((context[i, :, dim], predicted_output[i, :, dim]))[
                    begin_double_plot:end_double_plot],
                    label='Predicted')
            ax.plot(range(begin_double_plot, end_double_plot),
                    np.concatenate((context[i, :, dim], ground_truth[i, :, dim]))[begin_double_plot:end_double_plot],
                    label='Ground Truth')

            # Set the x-axis range to be 0 to 150
            ax.set_xlim(0, end_double_plot)

            # Add a legend
            ax.legend()

            # Title and Axis
            plt.title('[' + key + ']' + 'Predicted vs. Ground Truth: ' + str(i))
            plt.xlabel('Time')
            plt.ylabel('Feature Value')

            # Log the plot to WandB
            logger.log(
                {'[' + key + ':' + str(dim) + ']' + 'Predicted vs. Ground Truth: ' + str(i): plt})  # wandb.Image(plt)


def update_args(args, updates):
    if updates is not None:
        for key, value in updates.items():
            setattr(args, key, value)


def main(updates):
    print_header('*** EXPERIMENT ARGS ***')
    # give main the slurm dict and update the args with the slurm dict
    for key in updates:
        if updates[key] == 'None':
            updates[key] = None
    args = argparse.Namespace(**updates)  # initialize_args()
    #update_args(args, updates)

    seed_everything(args.seed)
    experiment_configs = load_main_config(args, config_dir='./configs')

    load_data, visualize_data = initialize_data_functions(args)
    print_header('*** DATASET ***')
    print_config(experiment_configs['dataset'])

    print_header('*** LOADER ***')
    print_config(experiment_configs['loader'])

    print_header('*** OPTIMIZER ***')
    print_config(experiment_configs['optimizer'])

    print_header('*** SCHEDULER ***')
    print_config(experiment_configs['scheduler'])

    # Loading Data
    dataloaders = load_data(experiment_configs['dataset'],
                            experiment_configs['loader'])
    train_loader, val_loader, test_loader = dataloaders
    splits = ['train', 'val', 'test']
    dataloaders_by_split = {split: dataloaders[ix]
                            for ix, split in enumerate(splits)}
    eval_loaders = get_evaluation_loaders(dataloaders, batch_size=args.batch_size)

    # Setup input_dim based on features
    x, y, *z = train_loader.dataset.__getitem__(0)
    args.input_dim = x.shape[1]  # L x D
    output_dim = y.shape[1]

    # Initialize Model
    args.device = (torch.device('cuda:0')
                   if torch.cuda.is_available() and not args.no_cuda
                   else torch.device('cpu'))
    model_configs = {'embedding_config': args.embedding_config,
                     'encoder_config': args.encoder_config,
                     'decoder_config': args.decoder_config,
                     'output_config': args.output_config}
    model_configs = OmegaConf.create(model_configs)
    model_configs = load_model_config(model_configs, config_dir='./configs/model',
                                      args=args)

    model_configs['inference_only'] = False
    model_configs['lag'] = args.lag
    model_configs['horizon'] = args.horizon

    if args.features == 'M':  # Update output
        update_output_config_from_args(model_configs['output_config'], args,
                                       update_output_dim=True, output_dim=output_dim)
        model_configs['output_config'].input_dim = model_configs['output_config'].kwargs.input_dim
        model_configs['output_config'].output_dim = model_configs['output_config'].kwargs.output_dim
        print(model_configs['output_config'])

    model = SpaceTime(**model_configs)
    model.replicate = args.replicate  # Only used for testing specific things indicated by replicate
    model.set_lag(args.lag)
    model.set_horizon(args.horizon)

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, experiment_configs['optimizer'])
    scheduler = get_scheduler(model, optimizer,
                              experiment_configs['scheduler'])

    # Save some model stats
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    args.model_parameter_count = params
    arg_dict = print_args(args, return_dict=True, verbose=args.verbose)

    # Setup logging
    wandb = initialize_experiment(args, experiment_name_id='',
                                  best_train_metric=1e10,
                                  best_val_metric=1e10)
    try:
        pd.DataFrame.from_dict(arg_dict).to_csv(args.log_configs_path)
    except:
        pd.DataFrame.from_dict([arg_dict]).to_csv(args.log_configs_path)

    if args.verbose:
        print_header('*** MODEL ***')
        print(model)
        print_config(model_configs)

        from einops import rearrange
        _k = model.encoder.blocks[0].pre.get_kernel(rearrange(x, '(o l) d -> o d l', o=1))
        _k_diff = model.encoder.blocks[0].pre.diff_kernel
        _k_ma_r = model.encoder.blocks[0].pre.ma_r_kernel
        print_header(f'──> Preprocessing kernels (full: {_k.shape}, diff: {_k_diff.shape}, ma: {_k_ma_r.shape})')
        print(_k[:16, :_k_ma_r.shape[-1]])

    print_header(f'*** TRAINING ***')
    print(f'├── Lag: {args.lag}')
    print(f'├── Horizon: {args.horizon}')
    print(f'├── Criterion: {args.loss}, weights: {args.criterion_weights}')
    print(f'├── Dims: input={args.input_dim}, model={args.model_dim}')
    print(f'├── Number trainable parameters: {params}')  # └── 
    print(f'├── Experiment name: {args.experiment_name}')
    print(f'├── Logging to: {args.log_results_path}')

    # Loss objectives
    criterions = {name: get_loss(name) for name in ['rmse', 'mse', 'mae', 'rse']}
    eval_criterions = criterions
    for name in ['rmse', 'mse', 'mae']:
        eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')

    input_transform, output_transform = get_data_transforms(args.data_transform,
                                                            args.lag)

    model = train_model(model, optimizer, scheduler, dataloaders_by_split,
                        criterions, max_epochs=args.max_epochs, config=args,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        val_metric=args.val_metric, wandb=wandb,
                        return_best=True, early_stopping_epochs=args.early_stopping_epochs)

    # Eval best val checkpoint
    eval_splits = ['eval_train', 'val', 'test']
    eval_loaders_by_split = {split: eval_loaders[ix] for ix, split in
                             enumerate(eval_splits)}
    model, log_metrics, total_y, total_x = evaluate_model(model, dataloaders=eval_loaders_by_split,
                                                          optimizer=optimizer, scheduler=scheduler,
                                                          criterions=eval_criterions, config=args,
                                                          epoch=args.best_val_metric_epoch,
                                                          input_transform=input_transform,
                                                          output_transform=output_transform,
                                                          val_metric=args.val_metric, wandb=wandb,
                                                          train=False)
    n_plots = len(splits)  # train, val, test
    fig, axes = plt.subplots(1, n_plots, figsize=(6.4 * n_plots, 4.8))

    plot_forecasts(total_y, splits=eval_splits, axes=axes)

    if not args.no_wandb:
        wandb.log({"forecast_plot": fig})
        wandb.log(log_metrics)

    context = total_x['test'].numpy()
    ground_truth = total_y['test']['true'].numpy()
    predicted_output = total_y['test']['pred'].numpy()



    file_path = Path(__file__)
    file_path = file_path.parent / 'tmp' / (args.experiment_name + '.npz')

    scale_path = file_path.parent / 'tmp' / 'scaler.npz'

    with np.load(scale_path) as data:
        mean = data['mean']
        std = data['std']

    os.remove(scale_path)

    np.savez(file=file_path.absolute(),
             context=context,
             ground_truth=ground_truth,
             predicted_output=predicted_output,
             mean=mean,
             std=std,
             )
    print('filez saved')

    visualize_and_log(key='test',
                      context=context,
                      ground_truth=ground_truth,
                      predicted_output=predicted_output,
                      context_length=int(args.lag),  # args.lag[0]
                      prediction_length=predicted_output.shape[1],
                      plot_count=3,
                      logger=wandb,
                      )



if __name__ == '__main__':
    args = initialize_args()
    with open('testcheetah.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    main(cfg)
