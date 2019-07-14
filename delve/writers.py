"""
This file contains alternative file writers
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter


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

    def __init__(self, savepath: str):
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
        self.save()


class PrintWriter(AbstractWriter):

    def __init__(self, savepath: str):
        pass

    def add_scalar(self, name, value, **kwargs):
        print(name,':', value)

    def add_scalars(self, prefix, value_dict, **kwargs):
        for key in value_dict.keys():
            self.add_scalar(prefix+'_'+key, value_dict[key])

    def save(self):
        print('Jobs done')

    def close(self):
        print('I am going home now')


class TensorBoardWriter(AbstractWriter):

    def __init__(self, savepath):
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