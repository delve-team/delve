import os
import sys

import torch
import torch.nn

from delve import CheckLayerSat
from delve.writers import CSVandPlottingWriter


def test_dense_saturation_runs():
    save_path = 'temp/'
    model = torch.nn.Sequential(torch.nn.Linear(10, 88))

    writer = CSVandPlottingWriter(save_path, fontsize=16, primary_metric='test_accuracy')
    saturation = CheckLayerSat(save_path,
                               [writer], model,
                               stats=['lsat', 'idim'])

    test_input = torch.randn(5, 10)
    output = model(test_input)
    return True


def test_lstm_saturation_runs():
    save_path = 'temp/test'

    # Run 1
    timeseries_method = 'timestepwise'

    model = torch.nn.Sequential()
    lstm = torch.nn.LSTM(10, 88, 2)
    lstm.name = 'lstm2'
    model.add_module('lstm', lstm)

    writer = CSVandPlottingWriter(save_path, fontsize=16, primary_metric='test_accuracy')
    saturation = CheckLayerSat(save_path,
                               [writer], model,
                               stats=['lsat', 'idim'],
                               timeseries_method=timeseries_method)

    input = torch.randn(5, 3, 10)
    output, (hn, cn) = model(input)
    saturation.close()


def test_lstm_saturation_embed_runs():
    save_path = 'temp/test'
    # Run 2
    timeseries_method = 'last_timestep'

    model = torch.nn.Sequential()
    lstm = torch.nn.LSTM(10, 88, 2)
    model.add_module('lstm', lstm)

    writer = CSVandPlottingWriter(save_path, fontsize=16)
    saturation = CheckLayerSat(save_path,
                               [writer], model,
                               stats=['lsat', 'idim', 'embed'],
                               timeseries_method=timeseries_method)

    input = torch.randn(5, 3, 10)
    output, (hn, cn) = model(input)
    assert saturation.logs['train-covariance-matrix']['lstm'].saved_samples.shape == torch.Size([5, 88])

    input = torch.randn(8, 3, 10)
    output, (hn, cn) = model(input)
    assert saturation.logs['train-covariance-matrix']['lstm'].saved_samples.shape == torch.Size([13, 88])
    saturation.add_saturations()
    saturation.close()
    return True
