import csv
import pandas as pd

def extract_metrics_from_ordeered_dict(ordered_dict, mode='train', result={}):
    for key in ordered_dict.keys():
        name = key.split('/')[-1]
        result[mode+'_'+name] = [ordered_dict[key]]
    return result
def extract_metrics_from_scalaer_dict(log_dict):
    result_dict = {}
    for key in log_dict.keys():
        mode = key.split('-')[0]
        extract_metrics_from_ordeered_dict(log_dict[key], mode, result_dict)
    return result_dict

def log_to_csv(value_dict, savename):
    df = pd.DataFrame.from_dict(value_dict)
    df.to_csv(savename+'.csv', sep=';')

def record_metrics(value_dict, log_dict, train_accuracy, train_loss, test_accuracy, test_loss, epoch, time):

    result_dict = extract_metrics_from_scalaer_dict(log_dict)
    result_dict['train_accuracy'] = [train_accuracy]
    result_dict['test_accuracy'] = [test_accuracy]
    result_dict['train_loss'] = [train_loss]
    result_dict['test_loss'] = [test_loss]
    result_dict['epoch'] = [epoch]
    result_dict['time_per_step'] = [time]
    if value_dict is None:
        return result_dict
    else:
        for key in value_dict.keys():
            value_dict[key] += result_dict[key]
    return value_dict
