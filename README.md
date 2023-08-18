# Walking Noise

## Purpose of this repository
This repository contains the acompaning code for the paper: `Walking Noise: On Layer-Specific Robustness of Neural Architectures against Noisy Computations and Associated Characteristic Learning Dynamics`

It is meant for documentation and reproduction purposes.
On one side the code serves as a way to show how particular aspects of the Walking Noise paper were implemented.
On the other side it is meant to provide interested parties with the ability to reproduces results or extend 
upon results from the paper.

## Prerequisites for running the experiments
### Cluster environment
The experiments are designed to be run on a cluster managed by SLURM. 
To keep track of individual experiments we use a library called [seml](https://github.com/TUM-DAML/seml).
Seml provides the ability to easily define grid searches and also to manage failed or killed jobs.

As such one requires the following:
- A SLURM cluster
- A mongodb instance for seml

### Python environment
The python environment is easily installed with conda.
Make sure conda is installed and accessible, then run the following:
```
conda env create -f seml_pyt-tf_environment.yml
```
Activate the environment with:
```
conda activate seml_pyt-tf
```
Then setup seml by running the following:
```
seml configure  # provide your MongoDB credentials
```
Depending on the nodes in your cluster it may be beneficial to run multiple experiments on one GPU in parallel.
the `seml_settings_example.py` shows an example configuration for 10 jobs in parallel.
If you want to use this configuration overwrite the default seml configuration like such:
```
cp seml_settings_example.py ~/.config/seml/settings.py
```

### Potentially optional additions to the code
Due to storage constraints on our database server, we opted to save all experiment artifacts on a network storage.
To make the code aware of your local cluster infrastructure you will need to add the name of your SLURM cluster and the 
network path you'd like to use for artifacts to the `_artifact_root_per_cluster_name` variable in `util\cluster.py`.

If you would like to instead save the artifact to your MongoDB instance modify the code to use the corresponding
[sacred function](https://sacred.readthedocs.io/en/stable/apidoc.html?highlight=add_artifact#sacred.Experiment.add_artifact).
This should be possible by converting all calls to the `add_artifact` member function to calls to `_sacred_exp.add_artifact`,
which calls a function with the same signature from the sacred experiment instance.

## Running an experiment
As an example we will be running the layer-wise noise experiments for LeNet-5/CIFAR-10.
Decide on a collection in your database to use for this experiment. We'll be going with `LeNet-CIFAR`.

To speed up the process of running the inference only experiments contained in the layer-wise experiments, we'll first be running the experiments without noise.
This way the script can later on load these pre-trained models.

Add the experiments:
```
seml LeNet-CIFAR add experiment_configs/CIFAR10/LeNet5_no_noise_experiment.yaml
```
Check that the experiments have been added successfully:
```
seml LeNet-CIFAR status
```
Start the experiments:
```
seml LeNet-CIFAR start
```

After the no-noise experiments have completed it is time to add the layer wise experiments with and without Batch Norm:
```
seml LeNet-CIFAR add experiment_configs/CIFAR10/LeNet5_BN_layer_wise_noise_experiment.yaml add experiment_configs/CIFAR10/LeNet5_layer_wise_noise_experiment.yaml
```
Check that the experiments have been added successfully:
```
seml LeNet-CIFAR status
```
Start the experiments:
```
seml LeNet-CIFAR start
```

And that's it for one of the layer wise experiments, now one simply needs to wait until they complete or run other experiments.

For more information on how to use seml, please refer to the [seml examples](https://github.com/TUM-DAML/seml/tree/master/examples).

For analyzing the experiments the result data can be loaded from the MongoDB. For an example refer to the 
seml [notebook example](https://github.com/TUM-DAML/seml/blob/master/examples/notebooks/experiment_results.ipynb).
Note that converting the so produced results to dataframes is not recommended.

## Credit
Apart from their own contribution the authors would like to make note of the following open source and knowledge repositories:
- Many of the network implementations were inspired by kuangliu's `pytorch-cifar` repository: https://github.com/kuangliu/pytorch-cifar
- The GSC2 dataloader implementation is directly taken from the FINN-hls4ml submission to the tiny MLPerf 0.7 benchmark: https://github.com/mlcommons/tiny_results_v0.7/tree/main/open/hls4ml-finn/code/kws/KWS-W3A3/training/data
- The implementation of the noisy operator is inspired by the following discussion on the PyTorch forums: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745




