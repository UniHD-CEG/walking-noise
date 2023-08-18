import copy
import filecmp
import hashlib
import json
import os
import os.path
import shutil
import subprocess
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import seml
import torch
import torch.nn as nn
from sacred import Experiment

from data.CIFAR10 import CIFAR10
from data.FashionMNIST import FashionMNIST
from data.GSC2 import GSC2_PyTorch, GSC2_TF
from data.MNIST import MNIST
from data.SVHN import SVHN
from models_noisy.cnn_HE import CNN_HE
from models_noisy.lenet import LeNet
from models_noisy.mlp import MLP
from models_noisy.util import WeightClamper
from models_noisy.vgg import VGG
from noise_operator import config as cfg
from util import cluster
from util.console_logging import print_status

sacred_exp = Experiment(save_git_info=False)
seml.setup_logger(sacred_exp)


@sacred_exp.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@sacred_exp.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        sacred_exp.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))
        # sacred_exp.observers.append(seml.create_neptune_observer('uhdcsg/' + db_collection, api_token=None))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self,
                 sacred_exp,
                 init_all=True,
                 nfs_artifact_root=None,
                 device=None,
                 ):
        # Setup internal variables
        self._sacred_exp = sacred_exp
        self._seml_return_data = dict()
        self._seml_return_data['artifacts'] = dict()
        if nfs_artifact_root is None:
            self._nfs_artifact_root = cluster.get_artifact_root()
        elif nfs_artifact_root:
            self._nfs_artifact_root = nfs_artifact_root
        try:
            self.num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            self.num_avail_cpus = min(3, self.num_avail_cpus)
        except KeyError:
            self.num_avail_cpus = 1
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # Generate the artifact UUID, we trust that the config-hash and UUID4 together are sufficiently unique here
        config_hash = self.get_config_hash()
        self._artifact_uuid = f'{config_hash}_{str(uuid.uuid4())}'
        # Do main initialization
        if init_all:
            self.init_all()

    def get_seml_return_data(self):
        return self._seml_return_data

    def log_scalar(self, metric_name, value, log_to_sacred_observers=False):
        """
        Wrapper function, which logs to sacred and to the seml return argument at the same time.
        """
        # Log to sacred observers
        if log_to_sacred_observers:
            self._sacred_exp.log_scalar(metric_name, value)
        # Log to seml
        log_time = time.time()
        if metric_name not in self._seml_return_data.keys():
            self._seml_return_data[metric_name] = list()
            self._seml_return_data[metric_name + "_time"] = list()
        self._seml_return_data[metric_name].append(value)
        self._seml_return_data[metric_name + "_time"].append(log_time)

    @sacred_exp.capture
    def get_config_hash(self, general, data, model, optimizer, noise_settings, db_collection):
        """Calculates the sha1 hash value of the current experiment configuration
        Inspired from: https://github.com/TUM-DAML/seml/blob/7d9352e51c9a83b77aa30617e8926863f0f48797/seml/utils.py#L191"""
        config_hash = hashlib.sha1()
        config_dict = {'general': general, 'data': data, 'model': model, 'optimizer': optimizer,
                       'noise_settings': noise_settings, 'db_collection': db_collection}
        config_hash.update(json.dumps(config_dict, sort_keys=True).encode("utf-8"))
        return config_hash.hexdigest()

    def add_artifact(self, filename: Union[str, Path],
                     name: Optional[str] = None,
                     metadata: Optional[dict] = None,
                     content_type: Optional[str] = None,
                     ) -> None:
        """
        Copies the artifact to the network file system and stores its path in the seml result data.
        """
        # Create a new artifact folder
        artifact_folder = self._nfs_artifact_root / self._artifact_uuid
        artifact_folder.mkdir()

        # Copy over the file
        src_file = Path(filename)
        dst_file = artifact_folder / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"Copied artifact {src_file} to {dst_file}")
        # Check that the file was copied correctly and overwrite otherwise
        for i in range(5):
            time.sleep(1.)
            filecmp.clear_cache()
            if not filecmp.cmp(src_file, dst_file, shallow=False):
                print(f"Artifact copy mismatches the original, retrying ({i}-th try).")
                shutil.copy2(src_file, dst_file)
            else:
                break
        else:  # No break executed
            print("Copy operation of the artifact failed after too many retries.")
        # Save the artifact name in the seml data
        if name is None:
            name = src_file.name
        self._seml_return_data['artifacts'][name] = str(dst_file)

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @sacred_exp.capture()
    def init_dataset(self, data):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        # Find the dataset
        dataset = data['dataset']
        if dataset == "MNIST":
            self.data = MNIST
        elif dataset == "CIFAR10":
            self.data = CIFAR10
        elif dataset == "GSC2_PyTorch":
            self.data = GSC2_PyTorch
        elif dataset == "GSC2_TF":
            self.data = GSC2_TF
        elif dataset == "SVHN":
            self.data = SVHN
        elif dataset == "FashionMNIST":
            self.data = FashionMNIST
        else:
            raise ValueError(f"Dataset with name {dataset} is not supported.")
        # Get the batch_size
        if 'batch_size' in data:
            batch_size = data['batch_size']
        else:
            batch_size = 128
        # Init the dataset
        self.data = self.data(num_workers=self.num_avail_cpus,
                              data_root=str(cluster.get_artifact_root() / Path('datasets/')),
                              batch_size=batch_size,
                              )
        # Download the data
        self.data.prepare_data()
        # Setup the torch datasets
        self.data.setup()

        # Init train and val dataloaders
        self.train_loader = self.data.train_dataloader()
        self.val_loader = self.data.val_dataloader()
        if self.data.has_test_dataset:
            self.test_loader = self.data.test_dataloader()

    @staticmethod
    def assemble_layer_noise_config(experiment_noise_settings):
        '''
        Create the layer wise configuration for the noise factory from the experiment configuration.
        '''
        layer_wise_noise_config = dict()

        # Check for a layer wise setting, this addresses only one layer
        # Check that we have a sub dictionary available to us
        # Otherwise this might just be none and we can continue.
        try:
            layer_wise_setting = experiment_noise_settings['layer_wise']
        except KeyError:
            layer_wise_setting = None
        if isinstance(layer_wise_setting, dict):
            # For now only configuring one single layer is supported
            ex_layer_settings = experiment_noise_settings['layer_wise']
            single_layer_config = cfg.resolve_config_from_name(
                ex_layer_settings['noise_type'],
                **ex_layer_settings
            )
            index = int(ex_layer_settings['layer_index'])
            layer_wise_noise_config[index] = single_layer_config
            return layer_wise_noise_config

        # Check for a layer mapped setting, this addresses multiple layers
        try:
            layer_mapped_setting = experiment_noise_settings['layer_mapped']
        except KeyError:
            layer_mapped_setting = None
        if isinstance(layer_mapped_setting, dict):
            mapped_settings = copy.deepcopy(layer_mapped_setting)
            std_map = mapped_settings['std_map']
            # Do mapping rescaling
            std_multiplication_factor = mapped_settings['std_multiplication_factor']
            if mapped_settings['re_normalize_mapping']:
                std_sum = sum(std_map.values())
                std_multiplication_factor /= std_sum
            for key in std_map.keys():
                std_map[key] *= std_multiplication_factor
            # Create mapping
            for index in std_map.keys():
                kwarg_settings = copy.deepcopy(mapped_settings['noise_op_kwargs'])
                kwarg_settings[mapped_settings['std_key_name']] = std_map[index]
                single_layer_config = cfg.resolve_config_from_name(
                    mapped_settings['noise_type'],
                    **kwarg_settings
                )
                index = int(index)
                layer_wise_noise_config[index] = single_layer_config
            # Done
            return layer_wise_noise_config

        return layer_wise_noise_config

    def create_model(self, model, default_noise_config, layer_wise_noise_config):
        # Setup the nn
        model_class = model['model_class']
        if model_class == "MLP":
            # Here we can pass the "model_params" dict to the constructor directly, which can be very useful in
            # practice, since we don't have to do any model-specific processing of the config dictionary.
            internal_model = MLP(**model['MLP'],
                             num_classes=self.data.num_classes,
                             input_shape=self.data.dims,
                             default_noise_config=default_noise_config,
                             layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'VGG':
            internal_model = VGG(**model['VGG'],
                             num_classes=self.data.num_classes,
                             input_shape=self.data.dims,
                             default_noise_config=default_noise_config,
                             layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'LeNet':
            internal_model = LeNet(**model['LeNet'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == "CNN_HE":
            internal_model = CNN_HE(**model['CNN_HE'],
                                num_classes=self.data.num_classes,
                                input_shape=self.data.dims,
                                default_noise_config=default_noise_config,
                                layer_wise_noise_config=layer_wise_noise_config)
        else:
            raise ValueError(f"Model with name {model_class} is not supported.")
        # Move the model to the correct device and set the datatype to float32
        internal_model.to(self.device, dtype=torch.float32)
        # Do a dry run to initialize the lazy-init layers
        init_tensor = torch.randn(2, *self.data.dims, dtype=torch.float32, device=self.device)
        internal_model(init_tensor)
        return internal_model

    @sacred_exp.capture()
    def init_model(self, model, noise_settings, data, _log):
        # Setup the noise, quantized, pruned, whatever default model
        # Get default and layer wise noise settings
        default_noise_config = cfg.resolve_config_from_name(
            noise_settings['default']['noise_type'],
            **noise_settings['default']
        )
        layer_wise_noise_config = self.assemble_layer_noise_config(noise_settings)

        self.model = self.create_model(model, default_noise_config, layer_wise_noise_config)
        # Print check
        _log.info(f"Created model of the following configuration: {self.model}")

        # Setup the criterion
        crit_name = model['criterion']
        if crit_name == "CrossEntropyLoss":
            loss_weight = None
            # Re-weighting for GSC2_TF
            if 'weight_criterion' in model:
                if model['weight_criterion']:
                    if data['dataset'] == "GSC2_TF":
                        # Compute weights to balance the training dataset
                        bins, edges = np.histogram(self.data._ds_train._label_array, 12)
                        class_density = bins / bins.sum()
                        inverse_class_density = 1 - class_density
                        # Further suppress the "unknown" label, since it is severely overrepresented in training data
                        # The exact value is taken from here: https://github.com/mlcommons/tiny_results_v0.7/blob/691f8b26aa9dffa09b1761645d4a35ad35a4f095/open/hls4ml-finn/code/kws/KWS-W3A3/training/const_QMLP.yaml#L32
                        label_suppression_unknown = 3.6
                        inverse_class_density[-1] /= label_suppression_unknown
                        inverse_class_density /= inverse_class_density.sum()
                        inverse_class_density = inverse_class_density.astype(np.float32)
                        loss_weight = torch.from_numpy(inverse_class_density).to(self.device)
                    else:
                        ValueError("Weighting is currently not implemented for other datasets than GSC2_TF.")
            self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            raise ValueError(f"Criterion with name {crit_name} is not supported.")

    @sacred_exp.capture()
    def init_optimizer(self, general, optimizer):
        # Set the optimizer
        optim_name = optimizer['optimizer_type']
        if optim_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer['lr'])
        else:
            raise ValueError(f"Optimizer with name {optim_name} is not supported.")

        # Set the scheduler
        sched_name = optimizer['lr_scheduler']
        if sched_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=general['num_epochs'])
        else:
            raise ValueError(f"Scheduler with name {sched_name} is not supported.")

        # Figure out the clamping and put in none if it wasn't specified
        try:
            weight_clamping_params = [general['weight_clamping']['min'], general['weight_clamping']['max']]
        except KeyError:
            weight_clamping_params = [None, None]
        self._weightClamper = WeightClamper(*weight_clamping_params)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    def training_step(self):
        # Training step
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy

    def validation_step(self, run_on_test_dataset_instead=False):
        # Validation step
        self.model.eval()
        summed_loss = 0.
        correct = 0
        total = 0

        with torch.no_grad():
            curr_ds = self.val_loader
            if run_on_test_dataset_instead:
                curr_ds = self.test_loader
            for batch_idx, (inputs, targets) in enumerate(curr_ds):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                summed_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        if run_on_test_dataset_instead:
            self.log_scalar("test.loss", summed_loss)
            self.log_scalar("test.accuracy", accuracy)
        else:
            self.log_scalar("validation.loss", summed_loss)
            self.log_scalar("validation.accuracy", accuracy)

        return summed_loss, accuracy


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@sacred_exp.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(sacred_exp, init_all=init_all)
    return experiment


@sacred_exp.capture
def check_if_curr_exp_has_noise_during_training(noise_settings):
    # First check if no noise is applied during training
    noise_during_training = False
    # Check if global noise is applied during training
    if noise_settings['default']['noise_type'] != 'NoNoise':
        if noise_settings['default']['enable_in_training']:
            noise_during_training = True
    # Check if any layer wise noise is applied during training
    if noise_settings['layer_wise'] is not None:
        if noise_settings['layer_wise']['enable_in_training']:
            noise_during_training = True
    return noise_during_training

@sacred_exp.capture
def get_no_noise_db_entry_equivalent_to_current_exp(general, data, model, optimizer, db_collection, _log):
    # Load already computed seml results, which have no noise and check if an equivalent one exists
    # Select experiments, which have no default noise AND which have no layer noise
    filter_dict = {
        'config.noise_settings.default.noise_type': 'NoNoise',
        'config.noise_settings.layer_wise': None,
        'config.noise_settings.layer_mapped': None,
    }
    fields_to_get = ['config', 'result']
    seml_res = seml.get_results(db_collection, filter_dict=filter_dict, fields=fields_to_get, to_data_frame=False)
    pre_trained_db_entry = None
    for completed_exp in seml_res:
        completed_exp_cfg = completed_exp['config']
        # Check if the experiment is actually without noise (these two operations should already be done by
        # MongoDB through the use of the filter_dict variable)
        if not completed_exp_cfg['noise_settings']['default']['noise_type'] == 'NoNoise':
            continue
        if completed_exp_cfg['noise_settings']['layer_wise'] is not None:
            continue
        # Check if the experiment matches our current config, excluding the noise
        # ToDo: Make this stricter again, i.e. completed_exp_cfg['model']['conf_name'] should check the whole model again, not just the conf
        # This was changed to make sure it dosen't take the repetition config into account, but it should take other stuff into account.
        # So it should do: completed_exp_cfg['model'] == model
        # But excluting the repetition_config key.
        internal_sub_model = model[list(model.keys())[0]]['conf_name']
        gotten_sub_model = completed_exp_cfg['model'][list(completed_exp_cfg['model'].keys())[0]]['conf_name']
        # Same for the number of epochs, or rather the general key, here we wated to cut out the 'experiment_name' key.
        internal_num_epochs = general['num_epochs']
        gotten_num_epochs = completed_exp_cfg['general']['num_epochs']
        internal_repeat_number = general['repeat_number']
        gotten__repeat_number = completed_exp_cfg['general']['repeat_number']
        if completed_exp_cfg['data'] == data and gotten_sub_model == internal_sub_model and \
                completed_exp_cfg['optimizer'] == optimizer and gotten_num_epochs == internal_num_epochs  and gotten__repeat_number == internal_repeat_number:
            _log.info('Found an equivalent pretrained model')
            pre_trained_db_entry = completed_exp
            break
    return pre_trained_db_entry

@sacred_exp.capture
def load_checkpoint_from_exp(pre_trained_db_entry, _log, device='cpu'):
    _log.info(f'Loading model, with _id: {pre_trained_db_entry["_id"]}')
    # Load the checkpoint of the pre-trained model
    checkpoint_path = Path(pre_trained_db_entry['result']['artifacts']['Trained model checkpoint'])
    local_checkpoint_path = cluster.convert_artifact_path_to_local_path(checkpoint_path, logger=_log)
    if local_checkpoint_path.exists():
        pre_trained_checkpoint = torch.load(local_checkpoint_path, map_location=device)
        found_pre_trained_model = True
    else:
        _log.warning("Could not load the model, "
                     "because the checkpoint file doesn't exist on the local artifact storage.")
    return found_pre_trained_model, pre_trained_checkpoint


# In some cases the model being trained doesn't inject any noise during training, only during evaluation.
# In these cases the training is equivalent to noise less training, so we try to load the model from an equivalent
# training run.
@sacred_exp.capture
def get_pre_trained_checkpoint(general, data, model, optimizer, noise_settings, db_collection, _log, device='cpu'):
    # Check if the model can be loaded from a pre-existing checkpoint
    found_pre_trained_model = False
    pre_trained_checkpoint = None
    noise_during_training = check_if_curr_exp_has_noise_during_training(noise_settings)

    if not noise_during_training:
        _log.info('Found that this experiment contains no noise at training time.')
        _log.info('Searching for a pretrained model without noise.')
        pre_trained_db_entry = get_no_noise_db_entry_equivalent_to_current_exp(general, data, model, optimizer, db_collection, _log)
        if pre_trained_db_entry is None:
            _log.warning('No equivalent pre-trained model was found, '
                         'evaluating this experiment may take longer than what would otherwise be required.')
        else:
            found_pre_trained_model, pre_trained_checkpoint = load_checkpoint_from_exp(pre_trained_db_entry, _log, device='cpu')

    return found_pre_trained_model, pre_trained_checkpoint


def get_free_gpus(_log):
    """
    Checks nvidia-smi for available GPUs and returns those with the most available memory.
    Inspired by: https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/2
    """
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode()
    gpu_stats = gpu_stats.replace(' MiB', '')
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    _log.info('GPU usage [MiB]:\n{}'.format(gpu_df))
    valid_gpus = gpu_df.loc[gpu_df['memory.free'] == gpu_df['memory.free'].max()].index.values
    _log.info(f'GPUs with the most available memory: {list(valid_gpus)}')
    return valid_gpus


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@sacred_exp.automain
def train(general, _log, experiment=None):
    # If we are running on a GPU, then select the device with the most available memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Wait for a random amount of time before initializing
        # Only sleep if there is more than one GPU available
        avail_devices = get_free_gpus(_log)
        if len(avail_devices) > 1:
            sleep_delay = np.random.randint(0, 60)
            _log.info(f'Multiple GPUs available: Sleeping for {sleep_delay} seconds before starting.')
            time.sleep(sleep_delay)
        else:
            _log.info("Only one GPU available: Skipping sleep")
        # Get the GPUs with the most memory and select one randomly
        avail_devices = get_free_gpus(_log)
        device = int(np.random.choice(avail_devices))
        _log.info(f'Selected GPU: {device}')
    # Exception for octane005: Force CPU usage
    if os.uname().nodename == 'octane005.ziti.uni-heidelberg.de':
        _log.info("Detected host as octane005. "
                  "Forcing 'cpu' as torch device to avoid issues with the unregistered GPU on this system.")
        device = 'cpu'
    device = torch.device(device)

    # Create the experiment wrapper
    if experiment is None:
        experiment = ExperimentWrapper(sacred_exp, device=device)

    # Check if we can get a pre-trained checkpoint for this particular experiment
    # This usually reduces the compute time immensely
    # Note that all the required arguments get inserted by sacred automatically
    found_pre_trained_model, pre_trained_checkpoint = get_pre_trained_checkpoint(device=device)
    if found_pre_trained_model:
        experiment.model.load_state_dict(pre_trained_checkpoint['state_dict'])

    # Training
    start_time = time.time()
    if found_pre_trained_model:
        # The model is already trained, so we simply do one validation step to compute the accuracy with noise
        val_loss, val_acc = experiment.validation_step()
        # Commandline logging
        status_dict = {
            "Validation: Loss": val_loss,
            "v_Accuracy": val_acc,
        }

        # Optional test dataset for some dataset
        if experiment.data.has_test_dataset:
            test_loss, test_acc = experiment.validation_step(run_on_test_dataset_instead=True)
            status_dict['Test: Loss'] = test_loss
            status_dict['ts_Accuracy'] = test_acc

        print_status(_log, start_time, 0, 1, **status_dict)
    else:
        # Do some actual training
        for epoch in range(general['num_epochs']):
            # Training and validation
            train_loss, train_acc = experiment.training_step()
            val_loss, val_acc = experiment.validation_step()
            experiment.scheduler.step()

            # Commandline logging
            status_dict = {
                "Training: Loss": train_loss,
                "tr_Accuracy:": train_acc,
                "Validation: Loss": val_loss,
                "v_Accuracy": val_acc,
            }

            # Optional test dataset for some dataset
            if experiment.data.has_test_dataset:
                test_loss, test_acc = experiment.validation_step(run_on_test_dataset_instead=True)
                status_dict['Test: Loss'] = test_loss
                status_dict['ts_Accuracy'] = test_acc

            print_status(_log, start_time, epoch, general['num_epochs'], **status_dict)

        # Export and save model data
        config_hash = experiment.get_config_hash()
        export_path = f"{config_hash}_{str(uuid.uuid4())}_trained_model.pt"
        torch.save({
            'state_dict': experiment.model.state_dict(),
            'optim_dict': experiment.optimizer.state_dict(),
            'scheduler_dict': experiment.scheduler.state_dict(),
        }, export_path)
        # Wait a bit to make sure the file is completely written before it gets uploaded
        time.sleep(10.)
        experiment.add_artifact(export_path, 'Trained model checkpoint')
        # Again, wait a bit to make sure the upload completes
        time.sleep(10.)
        # Make sure the artifact gets removed from the local disk
        os.remove(export_path)

    # Save the result data with seml
    return experiment.get_seml_return_data()
