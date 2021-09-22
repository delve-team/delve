import tempfile

import pytest
import torch
import torch.nn

from delve import CheckLayerSat
from delve.writers import CSVandPlottingWriter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEMP_DIR = tempfile.TemporaryDirectory()
TEMP_DIRNAME = TEMP_DIR.name


def test_dense_saturation_runs():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(torch.nn.Linear(10, 88)).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = CheckLayerSat(save_path, [writer],
                      model,
                      stats=['lsat', 'idim'],
                      device=device)

    test_input = torch.randn(5, 10).to(device)
    _ = model(test_input)
    return True


def test_lstm_saturation_runs():
    save_path = TEMP_DIRNAME

    # Run 1
    timeseries_method = 'timestepwise'

    model = torch.nn.Sequential().to(device)
    lstm = torch.nn.LSTM(10, 88, 2)
    lstm.name = 'lstm2'
    model.add_module('lstm', lstm)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    saturation = CheckLayerSat(save_path, [writer],
                               model,
                               stats=['lsat', 'idim'],
                               timeseries_method=timeseries_method,
                               device=device)

    input = torch.randn(5, 3, 10).to(device)
    output, (hn, cn) = model(input)
    saturation.close()


def test_lstm_saturation_embed_runs():
    save_path = TEMP_DIRNAME
    # Run 2
    timeseries_method = 'last_timestep'

    model = torch.nn.Sequential().to(device)
    lstm = torch.nn.LSTM(10, 88, 2)
    model.add_module('lstm', lstm)

    writer = CSVandPlottingWriter(save_path, fontsize=16)
    saturation = CheckLayerSat(save_path, [writer],
                               model,
                               stats=['lsat', 'idim', 'embed'],
                               timeseries_method=timeseries_method,
                               device=device)

    input = torch.randn(5, 3, 10).to(device)
    output, (hn, cn) = model(input)
    assert saturation.logs['train-covariance-matrix'][
        'lstm'].saved_samples.shape == torch.Size([5, 88])

    input = torch.randn(8, 3, 10)
    output, (hn, cn) = model(input)
    assert saturation.logs['train-covariance-matrix'][
        'lstm'].saved_samples.shape == torch.Size([8, 88])
    saturation.add_saturations()
    saturation.close()
    return True


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup temp directory once we are finished"""
    def remove_tempdir():
        TEMP_DIR.cleanup()

    request.addfinalizer(remove_tempdir)
