import tempfile

import pytest
import torch
import torch.nn

from delve import SaturationTracker
from delve.pca_layers import LinearPCALayer, Conv2DPCALayer, \
    change_all_pca_layer_thresholds_and_inject_random_directions, change_all_pca_layer_thresholds
from delve.writers import CSVandPlottingWriter, PrintWriter, NPYWriter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEMP_DIR = tempfile.TemporaryDirectory()
TEMP_DIRNAME = TEMP_DIR.name


def test_dense_saturation_runs_with_many_writers():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(torch.nn.Linear(10, 88)).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    writer2 = NPYWriter(save_path)
    writer3 = PrintWriter()
    sat = SaturationTracker(save_path, [writer, writer2, writer3],
                            model,
                            stats=['lsat', 'idim'],
                            device=device)

    test_input = torch.randn(5, 10).to(device)
    _ = model(test_input)
    sat.add_scalar("test_accuracy", 1.0)
    sat.add_saturations()

    return True


def test_dense_saturation_runs():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(torch.nn.Linear(10, 88)).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = SaturationTracker(save_path, [writer],
                          model,
                          stats=['lsat', 'idim'],
                          device=device)

    test_input = torch.randn(5, 10).to(device)
    _ = model(test_input)
    return True


def test_dense_saturation_runs_with_pca():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 88),
        LinearPCALayer(88)
    ).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = SaturationTracker(save_path, [writer],
                          model,
                          stats=['lsat', 'idim'],
                          device=device)

    test_input = torch.randn(5, 10).to(device)
    _ = model(test_input)
    model.eval()
    _ = model(test_input)
    return True


def test_conv_saturation_runs_with_pca():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 88, (3, 3)),
        Conv2DPCALayer(88)
    ).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = SaturationTracker(save_path, [writer],
                          model,
                          stats=['lsat', 'idim'],
                          device=device)

    test_input = torch.randn(32, 4, 10, 10).to(device)
    _ = model(test_input)
    model.eval()
    _ = model(test_input)
    return True


def test_conv_saturation_runs_with_pca_injecting_random_directions():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 88, (3, 3)),
        Conv2DPCALayer(88)
    ).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = SaturationTracker(save_path, [writer],
                          model,
                          stats=['lsat', 'idim'],
                          device=device)

    test_input = torch.randn(32, 4, 10, 10).to(device)
    _ = model(test_input)
    model.eval()
    x = model(test_input)
    change_all_pca_layer_thresholds_and_inject_random_directions(0.99, model)
    y = model(test_input)
    return x != y


def test_conv_saturation_runs_with_pca_change_threshold():
    save_path = TEMP_DIRNAME
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 88, (3, 3)),
        Conv2DPCALayer(88)
    ).to(device)

    writer = CSVandPlottingWriter(save_path,
                                  fontsize=16,
                                  primary_metric='test_accuracy')
    _ = SaturationTracker(save_path, [writer],
                          model,
                          stats=['lsat', 'idim'],
                          device=device)

    test_input = torch.randn(32, 4, 10, 10).to(device)
    _ = model(test_input)
    model.eval()
    x = model(test_input)
    change_all_pca_layer_thresholds(0.5, model)
    z = model(test_input)
    return x != z


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
    saturation = SaturationTracker(save_path, [writer],
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
    saturation = SaturationTracker(save_path, [writer],
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
