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

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    pass

STATMAP = {
    'idim': 'intrinsic-dimensionality',
    'lsat': 'saturation',
    'cov': 'covariance-matrix'
}


class AbstractWriter(ABC):

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
        This writers allows you to have multiple writers.
        :param writers: a list of writers. function call of this writer is executed on every writer in this list.
        """
        super(CompositWriter, self).__init__()
        self.writers = writers

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
        This writers produces a csv file that can be used for analysis after training. The csv-file is overwritten with
        an updated version every time save() is called.
        :param savepath: The path to save the csv to
        """
        super(CSVWriter, self).__init__()
        self.value_dict = {}
        self.savepath = savepath

    def add_scalar(self, name, value, **kwargs):
        if 'covariance' in name:
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
        The npy-writers creates a folder containing one subfolder for each stat. Each subfolder contains a npy-file for each epoch containing the value
        for each epoch.
        Other than any other writer, this writer is able to save non-scalar values and can thus be used to save
        the covariance-matrix
        :param savepath: the root folder to save the folder structure and npy-files into
        :param zip: if set to True, the folder structure will be zipped everytime save() is called.
        """
        super(NPYWriter, self).__init__()
        self.savepath = savepath
        self.epoch_counter = {}
        self.zip = zip

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
        This is a very basic logger, it only prints everything to the console output
        """
        super(PrintWriter, self).__init__()

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
        Writes everything to tensorflow logs
        :param savepath: the path for result logging
        """
        super(TensorBoardWriter, self).__init__()
        self.savepath = savepath
        self.writer = SummaryWriter(savepath)

    def add_scalar(self, name, value, **kwargs):
        if 'covariance' in name:
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
        This writer produces the CSV similar to the CSV writer, but also provides plots, which are per default stored in
        the subfolder.
        :param savepath: path to store images and csvs
        :param plot_manipulation_func: this is a function mapping an axis object to an axis object, you can use this to
                                       to manipulate the plot even further, by writing arbitrary pyplot code in this
                                       function
        :param kwargs:
        """
        super(CSVandPlottingWriter, self).__init__(savepath)
        self.plot_man_func = plot_manipulation_func if plot_manipulation_func is not None else lambda x: x
        self.primary_metric = None if not 'primary_metric' in kwargs else kwargs['primary_metric']
        self.fontsize = 16 if not 'fontsize' in kwargs else kwargs['fontsize']
        self.figsize = None if not 'figsize' in kwargs else kwargs['figsize']
        self.epoch_counter: int = 0
        self.stats = []

    def _look_for_stats(self):
        if len(self.stats) == 0:
            sat = False
            idim = False
            for key in self.value_dict.keys():
                if 'saturation' in key:
                    sat = True
                if 'intrinsic-dimensionality' in key:
                    idim = True
            if sat:
                self.stats.append('lsat')
            if idim:
                self.stats.append('idim')

    def add_scalar(self, name, value, **kwargs):
        if 'covariance' in name:
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
        self._look_for_stats()
        pd.DataFrame.from_dict(self.value_dict).to_csv(self.savepath + '.csv', sep=';')
        for stat in self.stats:
            plot_stat_level_from_results(self.savepath + '.csv', stat=stat, epoch=self.epoch_counter,
                                         primary_metric=self.primary_metric, fontsize=self.fontsize,
                                         figsize=self.figsize)
        self.epoch_counter += 1

    def close(self):
        pass


def extract_layer_stat(df, epoch=19, primary_metric=None, stat='saturation') -> Tuple[pd.DataFrame, float]:
    """
    Extracts a specific statistic for a single epoch from a result dataframe as produced by the CSV-writer
    :param df: the dataframe as produced by the csv-writer
    :param epoch: the epoch to filter by
    :param primary_metric: the primary metric for logged for performance evaluation, may be left empty
    :param stat: the statistic to look for, must be a substring matching all columns belonging to stat statistic like "saturation"
    :return: a dataframe with a single row, corresponding to the epoch containing only the columns that contain the substring
    described in the stat-parameter in their name. Second return value is the primary metric value
    """
    cols = list(df.columns)
    train_cols = [col for col in cols if
                  'train' in col and not 'accuracy' in col and stat in col]
    epoch_df = df[df.index.values == epoch]
    pm = None if primary_metric is None else epoch_df[primary_metric].values[0]
    epoch_df = epoch_df[train_cols]
    return epoch_df, pm


def plot_stat(df, stat, pm=-1, savepath='run.png', epoch=0, primary_metric=None, fontsize=16, figsize=None,
              line=True, scatter=True, ylim=(0, 1.0), alpha_line=.6, alpha_scatter=1.0, color_line=None,
              color_scatter=None,
              primary_metric_loc=(0.7, 0.8), show_col_label_x=True, show_col_label_y=True, show_grid=True, save=True):
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
    if figsize is not None:
        print(figsize)
        plt.figure(figsize=figsize)
    ax = plt.gca()
    col_names = [i for i in df.columns]
    if line:
        ax.plot(list(range(len(col_names))), df.values[0], alpha=alpha_line, color=color_line)
    if scatter:
        ax.scatter(list(range(len(col_names))), df.values[0], alpha=alpha_scatter, color=color_scatter)
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
        final_savepath = savepath.replace('.csv', f'{stat}_epoch_{epoch}.png')
        print(final_savepath)
        plt.savefig(final_savepath)
    return ax


def plot_stat_level_from_results(savepath, epoch, stat, primary_metric=None, fontsize=16, figsize=None, line=True,
                                 scatter=True, ylim=(0, 1.0), alpha_line=.6, alpha_scatter=1.0, color_line=None,
                                 color_scatter=None,
                                 primary_metric_loc=(0.7, 0.8), show_col_label_x=True, show_col_label_y=True,
                                 show_grid=True, save=True):
    df = pd.read_csv(savepath, sep=';')
    epoch_df, pm = extract_layer_stat(df, stat=STATMAP[stat], epoch=epoch, primary_metric=primary_metric)
    ax = plot_stat(df=epoch_df, pm=pm, savepath=savepath, epoch=epoch, primary_metric=primary_metric, fontsize=fontsize,
                   figsize=figsize, stat=stat, ylim=None if stat is 'idim' else (0, 1.0), line=line, scatter=scatter,
                   alpha_line=alpha_line, alpha_scatter=alpha_scatter, color_line=color_line,
                   color_scatter=color_scatter,
                   primary_metric_loc=primary_metric_loc, show_col_label_x=show_col_label_x,
                   show_col_label_y=show_col_label_y,
                   show_grid=show_grid, save=save)
    return ax
