import os
from pathlib import Path


_artifact_root_per_cluster_name = {
    'helix': Path('/home/hd/hd_hd/hd_sc429/gpfs/hd_sc429-seml_artifact_storage_1/'),
    'octane': Path('/share/hborras/seml_artifacts/'),
}


def get_cluster_name():
    """
    Access the SLURM_CLUSTER_NAME environment variable to determine the cluster name.
    If ta variable is not set None will be returned. This may happen when the program is executed
    outside of a SLRUM job.
    """
    # Get info from SLURM
    cluster_name = os.getenv("SLURM_CLUSTER_NAME")
    # If this fails try to guess from the username
    if cluster_name is None:
        if os.getlogin() == 'hd_sc429':
            cluster_name = 'helix'
        elif os.getlogin() == 'hborras':
            cluster_name = 'octane'
        else:
            cluster_name = None
    # Return result
    return cluster_name


def get_artifact_root(cluster_name=get_cluster_name()):
    return _artifact_root_per_cluster_name[cluster_name]


def convert_artifact_path_to_local_path(artifact_path: Path, local_cluster=get_cluster_name(), logger=None):
    """
    Converts between artifact paths for different clusters.
    It will return a version of the artifact path that works for the current cluster, that the job is running on.
    """
    local_artifact_root = get_artifact_root(local_cluster)
    # Find out from which cluster this artifact is originally
    for c_name, root_path in _artifact_root_per_cluster_name.items():
        if artifact_path.is_relative_to(root_path):
            if logger is not None:
                logger.info(f'Found original artifact location to be on {c_name}')
            break
    else:  # No break executed
        raise RuntimeError(f'Could not determine original cluster for artifact: {artifact_path}')

    converted_path = Path(str(artifact_path).replace(str(root_path), str(local_artifact_root)))
    if logger is not None:
        logger.info(f'Converted artifact path from {artifact_path} to {converted_path}')

    return converted_path

