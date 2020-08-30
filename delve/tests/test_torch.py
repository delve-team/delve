import os
import sys
from delve import CheckLayerSat
from delve.writers import CSVandPlottingWriter
import torch
import torch.nn


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
    save_path = 'temp/'

    # Run 1
    timeseries_method = 'timestepwise'
    model = torch.nn.Sequential(torch.nn.LSTM(10, 88, 2))

    writer = CSVandPlottingWriter(save_path, fontsize=16, primary_metric='test_accuracy')
    saturation = CheckLayerSat(save_path,
                               [writer], model,
                               stats=['lsat', 'idim'],
                               timeseries_method=timeseries_method)

    input = torch.randn(5, 3, 10)
    output, (hn, cn) = model(input)
    saturation.close()

    # Run 2
    timeseries_method = 'last_timestep'
    model = torch.nn.Sequential(torch.nn.LSTM(10, 88, 2))

    writer = CSVandPlottingWriter(save_path, fontsize=16, primary_metric='test_accuracy')
    saturation = CheckLayerSat(save_path,
                               [writer], model,
                               stats=['lsat', 'idim'],
                               timeseries_method=timeseries_method)

    input = torch.randn(5, 3, 10)
    output, (hn, cn) = model(input)
    saturation.close()
    return True
