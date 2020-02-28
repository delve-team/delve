from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import warnings
from delve.writers import STATMAP
from delve.metrics import compute_intrinsic_dimensionality, compute_saturation
from os.path import join, basename, dirname, curdir, splitext
from pathlib import Path
import torch


def _get_files(path: str, filext: str) -> List[str]:
    paths: List[str] = [str(x) for x in Path(path).rglob(f'*.{filext}')]
    return paths


def _check_stat_available(paths: List[str], stat: str) -> bool:
    return any([stat in basename(path) for path in paths])


def _filter_by_stat(paths: List[str], stat: str, neg: bool = False) -> List[str]:
    result = [path for path in paths if stat in basename(path)]
    if neg:
        for res in result:
            paths.remove(res)
        result = paths
    return result


def _filter_by_stat_shortcuts(paths: List[str], stats: List[str], neg: bool = False) -> List[str]:
    result = []
    for stat in stats:
        result += _filter_by_stat(paths, STATMAP[stat], neg)
    return list(set(result))


def _sort_files_by_epoch(npy_files: List[str]) -> List[str]:
    npy_files.sort(key=lambda x: int(splitext(basename(x))[0].split('epoch')[-1]))
    return npy_files


def _obtain_stat_from_npy(npy_files: List[str], stat: str) -> pd.DataFrame:
    filtered_paths = _filter_by_stat(npy_files, stat)
    sorted_paths = _sort_files_by_epoch(filtered_paths)
    res = [np.load(x) for x in sorted_paths]
    return res


def _group_by_directory(npy_files: List[str]) -> Dict[str, List[str]]:
    result = {}
    for file in npy_files:
        if basename(dirname(file)) in result:
            result[basename(dirname(file))].append(file)
        else:
            result[basename(dirname(file))] = [file]
    return result


def _obtain_stats_from_npy(npy_files: List[str]) -> pd.DataFrame:
    filtered = npy_files
    grouped_files = _group_by_directory(filtered)
    df_dict = {stat: _obtain_stat_from_npy(files, stat) for stat, files in grouped_files.items()}
    return pd.DataFrame.from_dict(df_dict)


def _recompute_value_from_cov(cov_path: str, stat: str, thresh: float) -> float:
    cov = torch.from_numpy(np.load(cov_path))
    if stat == 'lsat':
        return compute_saturation(cov, thresh)
    elif stat == 'idim':
        return compute_intrinsic_dimensionality(cov, thresh)


def _recompute_stat_from_cov(cov_name, cov_files: List[str], stat: str, thresh: float) -> Dict[str, List[float]]:
    sorted_cov_files = _sort_files_by_epoch(cov_files)
    name = cov_name.replace('covariance-matrix', STATMAP[stat])
    values = []
    for cov_file in sorted_cov_files:
        value = _recompute_value_from_cov(cov_file, stat, thresh)
        values.append(value)
    return {name: values}


def _recompute_stats(npy_files: List[str], stats: List[str], thresh: float) -> pd.DataFrame:
    # get everything except covariance matrix and the stats to compute
    filtered = _filter_by_stat_shortcuts(npy_files, stats=stats, neg=True)
    filtered = _filter_by_stat_shortcuts(filtered, stats=['cov'], neg=True)
    grouped_files = _group_by_directory(filtered)

    # get the covariance matrices
    cov_files = _filter_by_stat_shortcuts(npy_files, stats=['cov'])

    # obtain stats not recomputable through the covariance matrix
    df_dict = {stat: _obtain_stat_from_npy(files, stat) for stat, files in grouped_files.items()}

    # group the covariance matrix and recompute all stats for all stats and all layers
    grouped_covariance = _group_by_directory(cov_files)
    for cov_name, cov_paths in grouped_covariance.items():
        for stat in stats:
            stat_dict = _recompute_stat_from_cov(cov_name, cov_paths, stat, thresh)
            df_dict.update(stat_dict)
    return pd.DataFrame.from_dict(df_dict)


def reconstruct_csv_from_npy_data(npywriter_out_path: str,
                                  savefile: Optional[str] = None,
                                  thresh: Optional[float] = None,
                                  stats: List[str] = ['lsat', 'idim'],
                                  ) -> pd.DataFrame:
    """
    This function allows the user to reconstruct the csv as constructed by the csv-writer.
    It further allows computing metrics that were initially not computed during training, if the covariance matrix is
    stored during training using the npy-writer.
    It also allows to recompute stats using different thresholds than originally used during training.
    :param npywriter_out_path: The path were the npy-writer has written the output
    :param savefile: the target-file to store the csv into, if not set, the data is not saved
    :param thresh: the threshold for computing the intrinsic dimensionality and saturation. If not set, the function
    tries to find stored version of all metrics. If set to any value, the stats will be computed from the stored covariance matrix.
    If not covariance matrix is given and threshold is set an error is raised.
    :param stats: the statistic to compute or read. Valid values are 'sat' and 'idim'. All other metrics that were recorded
    during training and properly stored as npy-files will be read as well.
    :return:
    """
    npy_files: List[str] = _get_files(npywriter_out_path, 'npy')
    if thresh is None:
        if 'sat' in stats and not _check_stat_available(npy_files, 'saturation'):
            warnings.warn('No .npy file was found for saturation, stat will not be included')
            stats.remove('sat')
        if 'idim' in stats and not _check_stat_available(npy_files, 'intrinsic-dimensionality'):
            warnings.warn('No .npy file was found for intrinsic dimensionality, stat will not be included')
            stats.remove('idim')
        result = _obtain_stats_from_npy(npy_files)
    else:
        if not _check_stat_available(npy_files, 'covariance'):
            raise FileNotFoundError("Covariance Matrix was not found, recomputation impossible")
        result = _recompute_stats(npy_files, stats, thresh)
    result.to_csv(savefile, sep=';')
    return result
