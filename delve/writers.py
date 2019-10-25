"""
This file contains alternative file writers
"""
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import pathlib
import pandas as pd
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    pass


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


class CSVWriter(AbstractWriter):

    def __init__(self, savepath: str, **kwargs):
        super(CSVWriter, self).__init__()
        self.value_dict = {}
        self.savepath = savepath

    def add_scalar(self, name, value, **kwargs):
        if name in self.value_dict:
            self.value_dict[name].append(value)
        else:
            self.value_dict[name] = [value]
        return

    def add_scalars(self, prefix, value_dict, **kwargs):
        for name in value_dict.keys():
            self.add_scalar(name, value_dict[name])

    def save(self):
        pd.DataFrame.from_dict(self.value_dict).to_csv(self.savepath+'.csv', sep=';')

    def close(self):
        pass


class PrintWriter(AbstractWriter):

    def __init__(self, savepath: str, **kwargs):
        super(PrintWriter, self).__init__()

    def add_scalar(self, name, value, **kwargs):
        print(name, ':', value)

    def add_scalars(self, prefix, value_dict, **kwargs):
        for key in value_dict.keys():
            self.add_scalar(prefix+'_'+key, value_dict[key])

    def save(self):
        pass

    def close(self):
        pass


class TensorBoardWriter(AbstractWriter):

    def __init__(self, savepath: str, **kwargs):
        super(TensorBoardWriter, self).__init__()
        self.savepath = savepath
        self.writer = SummaryWriter(savepath)

    def add_scalar(self, name, value, **kwargs):
        self.writer.add_scalar(name, value)

    def add_scalars(self, prefix, value_dict, **kwargs):
        self.writer.add_scalars(prefix, value_dict)

    def save(self):
        pass

    def close(self):
        self.writer.close()


class CSVandPlottingWriter(CSVWriter):

    def __init__(self, savepath: str, **kwargs):
        super(CSVandPlottingWriter, self).__init__(savepath)
        self.primary_metric = None if not 'primary_metric' in kwargs else kwargs['primary_metric']
        self.fontsize = 16 if not 'fontsize' in kwargs else kwargs['fontsize']
        self.figsize = None if not 'figsize' in kwargs else kwargs['figsize']
        self.epoch_counter: int = 0

    def add_scalar(self, name, value, **kwargs):
        if name in self.value_dict:
            self.value_dict[name].append(value)
        else:
            self.value_dict[name] = [value]
        return

    def add_scalars(self, prefix, value_dict, **kwargs):
        for name in value_dict.keys():
            self.add_scalar(name, value_dict[name])

    def save(self):
        pd.DataFrame.from_dict(self.value_dict).to_csv(self.savepath+'.csv', sep=';')
        plot_saturation_level_from_results(self.savepath+'.csv', epoch=self.epoch_counter, primary_metric=self.primary_metric, fontsize=self.fontsize, figsize=self.figsize)
        self.epoch_counter += 1

    def close(self):
        pass

def extract_layer_saturation(df, epoch=19, primary_metric = None):
    cols = list(df.columns)
    train_cols = [col for col in cols if
                  'train' in col and not 'accuracy' in col and not 'loss' in col]
    epoch_df = df[df.index.values == epoch]
    pm = None if primary_metric is None else epoch_df[primary_metric].values[0]
    epoch_df = epoch_df[train_cols]
    return epoch_df, pm


def plot_saturation_level(df, pm=-1, savepath='run.png', epoch=0, primary_metric=None, fontsize=16, figsize=None):
    plt.clf()
    if figsize is not None:
        print(figsize)
        plt.figure(figsize=figsize)
    ax = plt.gca()
    col_names = [i for i in df.columns]
    ax.bar(list(range(len(col_names))), df.values[0])
    plt.xticks(list(range(len(col_names))), [col_name.replace('train-saturation_', '') for col_name in col_names], rotation=90)
    ax.set_ylim((0, 100))
    if primary_metric is not None:
        ax.text(1, 80, f'{primary_metric}: {pm}')
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Layers', fontsize=fontsize)
    plt.title(pathlib.Path(savepath).name.replace('_', ' ').replace('.csv', f' epoch: {epoch}'), fontsize=fontsize)
    plt.ylabel('Saturation in %', rotation='vertical', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(savepath.replace('.csv', f'_epoch_{epoch}.png'))
    return


def plot_saturation_level_from_results(savepath, epoch, primary_metric=None, fontsize=16, figsize=None):
    df = pd.read_csv(savepath, sep=';')
    epoch_df, pm = extract_layer_saturation(df, epoch=epoch, primary_metric=primary_metric)
    plot_saturation_level(epoch_df, pm, savepath, epoch, primary_metric=primary_metric, fontsize=fontsize, figsize=figsize)
