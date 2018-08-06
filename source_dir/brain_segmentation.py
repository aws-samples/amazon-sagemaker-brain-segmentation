import os
import tarfile
import mxnet as mx
import numpy as np
from iterator import DataLoaderIter
from losses_and_metrics import avg_dice_coef_metric
from models import build_unet, build_enet
import logging

logging.getLogger().setLevel(logging.DEBUG)

###############################
###     Training Loop       ###
###############################

def train(current_host, channel_input_dirs, hyperparameters, hosts, num_cpus, num_gpus):
    
    logging.info(mx.__version__)
    
    # Set context for compute based on instance environment
    if num_gpus > 0:
        ctx = [mx.gpu(i) for i in range(num_gpus)]
    else:
        ctx = mx.cpu()

    # Set location of key-value store based on training config.
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    
    # Get hyperparameters
    batch_size = hyperparameters.get('batch_size', 16)        
    learning_rate = hyperparameters.get('lr', 1E-3)
    beta1 = hyperparameters.get('beta1', 0.9)
    beta2 = hyperparameters.get('beta2', 0.99)
    epochs = hyperparameters.get('epochs', 100)
    num_workers = hyperparameters.get('num_workers', 6)
    num_classes = hyperparameters.get('num_classes', 4)
    class_weights = hyperparameters.get(
        'class_weights', [[1.35, 17.18, 8.29, 12.42]])
    class_weights = np.array(class_weights)
    network = hyperparameters.get('network', 'unet')
    assert network == 'unet' or network == 'enet', '"network" hyperparameter must be one of ["unet", "enet"]'
    
    # Locate compressed training/validation data
    train_dir = channel_input_dirs['train']
    validation_dir = channel_input_dirs['test']
    train_tars = os.listdir(train_dir)
    validation_tars = os.listdir(validation_dir)
    # Extract compressed image / mask pairs locally
    for train_tar in train_tars:
        with tarfile.open(os.path.join(train_dir, train_tar), 'r:gz') as f:
            f.extractall(train_dir)
    for validation_tar in validation_tars:
        with tarfile.open(os.path.join(validation_dir, validation_tar), 'r:gz') as f:
            f.extractall(validation_dir)
    
    # Define custom iterators on extracted data locations.
    train_iter = DataLoaderIter(
        train_dir,
        num_classes,
        batch_size,
        True,
        num_workers)
    validation_iter = DataLoaderIter(
        validation_dir,
        num_classes,
        batch_size,
        False,
        num_workers)
    
    # Build network symbolic graph
    if network == 'unet':
        sym = build_unet(num_classes, class_weights=class_weights)
    else:
        sym = build_enet(inp_dims=train_iter.provide_data[0][1][1:], num_classes=num_classes, class_weights=class_weights)
    logging.info("Sym loaded")
    
    # Load graph into Module
    net = mx.mod.Module(sym, context=ctx, data_names=('data',), label_names=('label',))
    
    # Initialize Custom Metric
    dice_metric = mx.metric.CustomMetric(feval=avg_dice_coef_metric, allow_extra_outputs=True)
    logging.info("Starting model fit")
    
    # Start training the model
    net.fit(
        train_data=train_iter,
        eval_data=validation_iter,
        eval_metric=dice_metric,
        initializer=mx.initializer.Xavier(magnitude=6),
        optimizer='adam',
        optimizer_params={
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2},
        num_epoch=epochs)
    
    # Save Parameters
    net.save_params('params')
    
    # Build inference-only graphs, set parameters from training models
    if network == 'unet':
        sym = build_unet(num_classes, inference=True)
    else:
        sym = build_enet(
            inp_dims=train_iter.provide_data[0][1][1:], num_classes=num_classes, inference=True)
    net = mx.mod.Module(
        sym, context=ctx, data_names=(
            'data',), label_names=None)
    
    # Re-binding model for a batch-size of one
    net.bind(data_shapes=[('data', (1,) + train_iter.provide_data[0][1][1:])])
    net.load_params('params')
    return net
