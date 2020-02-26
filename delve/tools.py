from typing import Optional, List
import pandas as pd
import warnings
from delve.pca_layers import change_all_pca_layer_thresholds
from delve.pca_layers import change_all_pca_layer_thresholds_and_inject_random_directions
from os.path import join, basename, dirname, curdir
from os import listdir
from pathlib import Path


def _get_files(path: str, filext: str) -> List[str]:
    paths: List[str] = [str(x) for x in Path(path).rglob(f'*.{filext}')]
    return paths


def _check_stat_available(paths: List[str], stat: str) -> bool:
    return any([stat in basename(path) for path in paths])


def _filters_by_stat(paths: List[str]) -> List[str]:
    pass


def recompute_stats(npy_files: List[str], stats: List[str], delta: float) -> pd.DataFrame:
    pass


def obtain_stata_from_npy(npy_files: List[str], stats: List[str]) -> pd.DataFrame:
    pass


def reconstruct_csv_from_npy_data(npywriter_out_path: str,
                                  savefile: Optional[str] = None,
                                  delta: Optional[float] = None,
                                  stats: List[str] = ['sat', 'idim']
                                  ) -> pd.DataFrame:
    """
    Reconstruct the result table normally produced by the csv-output
    :param npywriter_out_path: path to the output produced by the npy-writer
    :param dst_path:  destination path to save the dataframe, if None, the dataframe won't be saved
    :param delta:     the delta value to reconstruct the results. If None is given, original values (if available) will be used)
    :return:
    """
    npy_files: List[str] = _get_files(npywriter_out_path, 'npy')
    if delta is None:
        if 'sat' in stats and _check_stat_available(npy_files, 'saturation'):
            warnings.warn('No .npy file was found for saturation, stat will not be included')
            stats.remove('sat')
        if 'idim' in stats and _check_stat_available(npy_files, 'intrinsic-dimensionality'):
            warnings.warn('No .npy file was found for intrinsic dimensionality, stat will not be included')
            stats.remove('idim')
        result = obtain_stata_from_npy(npy_files, stats)
    else:
        if not _check_stat_available(npy_files, 'covariance'):
            raise FileNotFoundError("Covariance Matrix was not found, recomputation impossible")
        result = recompute_stats(npy_files, stats)
    result.to_csv(savefile, sep=';')
    return result


def reconstruct_plot_from_npy_data(npywriter_out_path: str,
                                   savepath: Optional[str] = None,
                                   delta: Optional[float] = None,
                                   ) -> None:
    pass
