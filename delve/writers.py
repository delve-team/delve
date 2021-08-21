"""
This file contains alternative file writers
"""
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from shutil import make_archive
from typing import Callable, List, Tuple
import pathlib
import pandas as pd
import numpy as np
import os
import warnings
import pickle as pkl

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    pass

STATMAP = {
    'idim': 'intrinsic-dimensionality',
    'lsat': 'saturation',
    'cov': 'covariance-matrix',
    'det': 'covariance-determinant',
    'trc': 'covariance-trace',
    'dtrc': 'diagonal-trace',
    'embed': 'embedded-sample'
}


class AbstractWriter(ABC):

    def _check_savestate_ok(self, savepath: str) -> bool:
        """
        Checks if a savestate from a writer is okay; raises a warning if not
        :param savepath: the path to the savestate
        :return:
        """
        if not os.path.exists(savepath):
            warnings.warn(f'{savepath} does not exists, savestate for {self.__class__.__name__} cannot be loaded')
            return False
        else:
            return True

    @abstractmethod
    def resume_from_saved_state(self, initial_epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def add_scalar(self, name, value, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def add_scalars(self, prefix, value_dict, global_step, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def close(self):
        pass


class CompositWriter(AbstractWriter):

    def __init__(self, writers: List[AbstractWriter]):
        """
        This writer combines multiple writers.
        :param writers: List of writers. Each writer is called when the CompositeWriter is invoked.
        """
        super(CompositWriter, self).__init__()
        self.writers = writers

    def resume_from_saved_state(self, initial_epoch: int):
        for w in self.writers:
            try:
                w.resume_from_saved_state(initial_epoch)
            except NotImplementedError:
                warnings.warn(f'Writer {w.__class__.__name__} raised a NotImplementedError when attempting to resume training'
                              'This may result in corrupted or overwritten data.')

    def add_scalar(self, name, value, **kwargs):
        for w in self.writers:
            w.add_scalar(name, value, **kwargs)

    def add_scalars(self, prefix, value_dict, **kwargs):
        for w in self.writers:
            w.add_scalars(prefix, value_dict, **kwargs)

    def save(self):
        for w in self.writers:
            w.save()

    def close(self):
        for w in self.writers:
            w.close()


class CSVWriter(AbstractWriter):

    def __init__(self, savepath: str, **kwargs):
        """
        This writer produces a csv file with all saturation values.
        The csv-file is overwritten with
        an updated version every time save() is called.
        :param savepath: CSV file path
        """
        super(CSVWriter, self).__init__()
        self.value_dict = {}
        self.savepath = savepath

    def resume_from_saved_state(self, initial_epoch: int):
        self.epoch_counter = initial_epoch
        if self._check_savestate_ok(self.savepath+'.csv'):
            self.value_dict = pd.read_csv(self.savepath + '.csv', sep=';', index_col=0).to_dict('list')

    def add_scalar(self, name, value, **kwargs):
        if 'covariance-matrix' in name:
            return
        if name in self.value_dict:
            self.value_dict[name].append(value)
        else:
            self.value_dict[name] = [value]
        return

    def add_scalars(self, prefix, value_dict, **kwargs):
        for name in value_dict.keys():
            self.add_scalar(name, value_dict[name])

    def save(self):
        pd.DataFrame.from_dict(self.value_dict).to_csv(self.savepath + '.csv', sep=';')

    def close(self):
        pass


class NPYWriter(AbstractWriter):

    def __init__(self, savepath: str, zip: bool = False, **kwargs):
        """
        The NPYWriter creates a folder containing one subfolder for each stat.
        Each subfolder contains a npy-file with the saturation value for each epoch.
        This writer saves non-scalar values and can thus be used to save
        the covariance-matrix.
        :param savepath: The root folder to save the folder structure to
        :param zip: Whether to zip the output folder after every invocation
        """
        super(NPYWriter, self).__init__()
        self.savepath = savepath
        self.epoch_counter = {}
        self.zip = zip

    def resume_from_saved_state(self, initial_epoch: int):
        if self._check_savestate_ok(os.path.join(self.savepath, 'epoch_counter.pkl')):
            self.epoch_counter = pkl.load(open(os.path.join(self.savepath, 'epoch_counter.pkl'), 'rb'))
        return

    def _update_epoch_counter(self, name: str) -> int:
        if name not in self.epoch_counter:
            self.epoch_counter[name] = 0
        else:
            self.epoch_counter[name] += 1
        return self.epoch_counter[name]

    def _get_and_create_savepath(self, name: str, epoch) -> str:
        savepath = os.path.join(self.savepath, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        return os.path.join(savepath, name + f'_epoch{epoch}')

    def add_scalar(self, name, value, **kwargs):
        epoch = self._update_epoch_counter(name)
        svpth = self._get_and_create_savepath(name, epoch)
        np.save(svpth, value, **kwargs)

    def add_scalars(self, prefix, value_dict, **kwargs):
        for key in value_dict.keys():
            self.add_scalar(prefix + '_' + key, value_dict[key])

    def save(self):
        pkl.dump(self.epoch_counter, open(os.path.join(self.savepath, 'epoch_counter.pkl'), 'wb'))
        if self.zip:
            make_archive(
                base_name=os.path.basename(self.savepath),
                format='zip',
                root_dir=os.path.dirname(self.savepath),
                verbose=True
            )

    def close(self):
        pass


class PrintWriter(AbstractWriter):

    def __init__(self, **kwargs):
        """
        Prints output to the console
        """
        super(PrintWriter, self).__init__()

    def resume_from_saved_state(self, initial_epoch: int):
        pass

    def add_scalar(self, name, value, **kwargs):
        print(name, ':', value)

    def add_scalars(self, prefix, value_dict, **kwargs):
        for key in value_dict.keys():
            self.add_scalar(prefix + '_' + key, value_dict[key])

    def save(self):
        pass

    def close(self):
        pass


class TensorBoardWriter(AbstractWriter):

    def __init__(self, savepath: str, **kwargs):
        """
        Writes output to tensorflow logs
        :param savepath: the path for result logging
        """
        super(TensorBoardWriter, self).__init__()
        self.savepath = savepath
        self.writer = SummaryWriter(savepath)

    def resume_from_saved_state(self, initial_epoch: int):
        raise NotImplementedError('Resuming is not yet implemented for TensorBoardWriter')

    def add_scalar(self, name, value, **kwargs):
        if 'covariance-matrix' in name:
            return
        self.writer.add_scalar(name, value)

    def add_scalars(self, prefix, value_dict, **kwargs):
        self.writer.add_scalars(prefix, value_dict)

    def save(self):
        pass

    def close(self):
        self.writer.close()


class CSVandPlottingWriter(CSVWriter):

    def __init__(self, savepath: str, plot_manipulation_func: Callable[[plt.Axes], plt.Axes] = None, **kwargs):
        """
        This writer produces CSV files and plots.
        :param savepath: Path to store plots and CSV files
        :param plot_manipulation_func: A function mapping an axis object to an axis object by
                                       using pyplot code.
        :param kwargs:
        """
        super(CSVandPlottingWriter, self).__init__(savepath)
        self.plot_man_func = plot_manipulation_func if plot_manipulation_func is not None else lambda x: x
        self.primary_metric = None if not 'primary_metric' in kwargs else kwargs['primary_metric']
        self.fontsize = 16 if not 'fontsize' in kwargs else kwargs['fontsize']
        self.figsize = None if not 'figsize' in kwargs else kwargs['figsize']
        self.epoch_counter: int = 0
        self.stats = []
        self.sample_stats = list()
        self.sample_value_dict = dict()

    def resume_from_saved_state(self, initial_epoch: int):
        self.epoch_counter = initial_epoch
        if not self._check_savestate_ok(self.savepath + '.csv'):
            return
        self.value_dict = pd.read_csv(self.savepath + '.csv', sep=';', index_col=0).to_dict('list')

    def _look_for_stats(self):
        if len(self.stats) == 0:
            sat = False
            idim = False
            det = False
            trc = False
            dtrc = False
            embed = True
            for key in self.value_dict.keys():
                if 'saturation' in key:
                    sat = True
                if 'intrinsic-dimensionality' in key:
                    idim = True
                if 'covariance-determinant' in key:
                    det = True
                if 'covariance-trace' in key:
                    trc = True
                if 'diagonal-trace' in key:
                    dtrc = True
                if 'embed' in key:
                    embed = True
            if sat:
                self.stats.append('lsat_train')
                self.stats.append('lsat_eval')
            if idim:
                self.stats.append('idim_train')
                self.stats.append('idim_eval')
            if det:
                self.stats.append('det_train')
                self.stats.append('det_eval')
            if trc:
                self.stats.append('trc_train')
                self.stats.append('trc_eval')
            if dtrc:
                self.stats.append('dtrc_train')
                self.stats.append('dtrc_eval')
            if embed:
                self.sample_stats.append('embed')

    def add_scalar(self, name, value, **kwargs):
        if 'covariance-matrix' in name:
            return
        if name in self.value_dict:
            self.value_dict[name].append(value)
        else:
            self.value_dict[name] = [value]

    def add_sample_scalar(self, name, value, **kwargs):
        if name in self.sample_value_dict:
            self.sample_value_dict[name].append(value)
        else:
            self.sample_value_dict[name] = [value]

    def add_scalars(self, prefix, value_dict, sample_value_dict, **kwargs):
        for name in value_dict.keys():
            self.add_scalar(name, value_dict[name])
        if sample_value_dict:
            for name in sample_value_dict.keys():
                self.add_sample_scalar(name, sample_value_dict[name])

    def _find_longest_entry(self) -> int:
        return max(*(len(value) for value in self.value_dict.values()))

    def _pad_entry(self, entry, max):
        if len(entry) == max:
            return entry
        else:
            return [np.nan for _ in range(max-entry)] + entry

    def _pad_stat(self):
        max_entry = self._find_longest_entry()
        return {k: self._pad_entry(v, max_entry) for k, v in self.value_dict.items()}

    def save(self):
        self._look_for_stats()

        pd.DataFrame.from_dict(self.value_dict).to_csv(self.savepath + '.csv', sep=';')
        for stat in self.stats:
            plot_stat_level_from_results(self.savepath + '.csv', stat=stat, epoch=-1,
                                         primary_metric=self.primary_metric, fontsize=self.fontsize,
                                         figsize=self.figsize)
        for stat in self.sample_stats:
            plot_scatter_from_results(self.savepath + '.csv', -1, stat,
                                      pd.DataFrame.from_dict(self.sample_value_dict))
        self.epoch_counter += 1

    def close(self):
        pass


def extract_layer_stat(df, epoch=19, primary_metric=None, stat='saturation', state_mode="train") -> Tuple[pd.DataFrame, float]:
    """
    Extracts a specific statistic for a single epoch from a result dataframe as produced by the CSV-writer
    :param df: The dataframe produced by a CSVWriter
    :param epoch: Epoch to filter by
    :param primary_metric: Primary metric for performance evaluation (optional)
    :param stat: The statistic to match. Must be a substring matching all columns belonging to stat statistic like "saturation"
    :return: A dataframe with a single row, corresponding to the epoch containing only the columns that contain the substring
    described in the stat-parameter in their name. Second return value is the primary metric value
    """
    cols = list(df.columns)
    train_cols = [col for col in cols if
                  state_mode in col and not 'accuracy' in col and stat in col]
    if not np.any(epoch == df.index.values):
        raise ValueError(f'Epoch {epoch} could not be recoreded, dataframe has only the following indices: {df.index.values}')
    epoch_df = df[df.index.values == epoch]
    pm = None if primary_metric is None else epoch_df[primary_metric].values[0]
    epoch_df = epoch_df[train_cols]
    return epoch_df, pm


def plot_stat(df, stat, pm=-1, savepath='run.png', epoch=0, primary_metric=None, fontsize=16, figsize=None,
              line=True, scatter=True, ylim=(0, 1.0), alpha_line=.6, alpha_scatter=1.0, color_line=None,
              color_scatter=None,
              primary_metric_loc=(0.7, 0.8), show_col_label_x=True, show_col_label_y=True, show_grid=True, save=True,
              samples=False, stat_mode="train"):
    """

    :param df:
    :param stat:
    :param pm:
    :param savepath:
    :param epoch:
    :param primary_metric:
    :param fontsize:
    :param figsize:
    :param line:
    :param scatter:
    :param ylim:
    :param alpha_line:
    :param alpha_scatter:
    :param color_line:
    :param color_scatter:
    :param primary_metric_loc:
    :param show_col_label_x:
    :param show_col_label_y:
    :param show_grid:
    :param save:
    :return:
    """
    plt.clf()
    plt.cla()
    plt.close()
    if epoch == -1:
        epoch = df.index.values[-1]
    if figsize is not None:
        print(figsize)
        plt.figure(figsize=figsize)
    ax = plt.gca()
    col_names = [i for i in df.columns]
    if np.all(np.isnan(df.values[0])):
        return ax
    if line:
        if samples:
            pass
        else:
            ax.plot(list(range(len(col_names))), df.values[0], alpha=alpha_line, color=color_line)
    if scatter:
        if samples:
            for sample in df.values[0][0]:
                x = float(sample[0])
                y = float(sample[1])
                ax.scatter(x, y, alpha=alpha_scatter, color=color_scatter)
        else:
            ax.scatter(list(range(len(col_names))), df.values[0], alpha=alpha_scatter, color=color_scatter)
    if not samples:
        plt.xticks(list(range(len(col_names))), [col_name.split('_', )[1] for col_name in col_names],
                   rotation=90)
    if not ylim is None:
        ax.set_ylim(ylim)
    if primary_metric is not None:
        ax.text(primary_metric_loc[0], primary_metric_loc[1], f'{primary_metric}: {pm}')
    plt.yticks(fontsize=fontsize)
    if show_col_label_x:
        plt.xlabel('layers', fontsize=fontsize)
    plt.title(pathlib.Path(savepath).name.replace('_', ' ').replace('.csv', f' epoch: {epoch}'), fontsize=fontsize)
    if show_col_label_y:
        plt.ylabel(stat if not stat in STATMAP else STATMAP[stat], rotation='vertical', fontsize=fontsize)
    if show_grid:
        plt.grid()
    plt.tight_layout()
    if save:
        final_savepath = savepath.replace('.csv', f'_{stat}_{stat_mode}_epoch_{epoch}.png')
        print(final_savepath)
        plt.savefig(final_savepath)
    return ax


def plot_stat_level_from_results(savepath, epoch, stat, primary_metric=None, fontsize=16, figsize=None, line=True,
                                 scatter=True, ylim=(0, 1.0), alpha_line=.6, alpha_scatter=1.0, color_line=None,
                                 color_scatter=None,
                                 primary_metric_loc=(0.7, 0.8), show_col_label_x=True, show_col_label_y=True,
                                 show_grid=True, save=True, stat_mode="train"):
    df = pd.read_csv(savepath, sep=';')
    if "_" in stat:
        stat, stat_mode = stat.split("_")
    if epoch == -1:
        epoch = df.index.values[-1]

    epoch_df, pm = extract_layer_stat(df, stat=STATMAP[stat], epoch=epoch, primary_metric=primary_metric, state_mode=stat_mode)
    ax = plot_stat(df=epoch_df, pm=pm, savepath=savepath, epoch=epoch, primary_metric=primary_metric, fontsize=fontsize,
                   figsize=figsize, stat=stat, ylim=None if not stat is 'lsat' else (0, 1.0), line=line, scatter=scatter,
                   alpha_line=alpha_line, alpha_scatter=alpha_scatter, color_line=color_line,
                   color_scatter=color_scatter,
                   primary_metric_loc=primary_metric_loc, show_col_label_x=show_col_label_x,
                   show_col_label_y=show_col_label_y,
                   show_grid=show_grid, save=save, stat_mode=stat_mode)
    return ax


def plot_scatter_from_results(savepath, epoch, stat, df):
    if len(df) > 0:
        if "_" in stat:
            stat = stat.split("_")[0]
        ax = plot_stat(df=df, savepath=savepath, epoch=epoch, stat=stat, line=False, save=True, samples=True, ylim=None)
        return ax
    else:
        return None