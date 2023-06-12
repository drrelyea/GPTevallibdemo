from metrics import ndcg_score, list_existence, METRICS, BatchedMetricCalculator
import pandas as pd
from typing import Any, Dict, Sequence
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import numpy as np
import pytest
import os
import tempfile
import shutil
import json
import random
import string
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Set up a temporary directory for the test files.
TEST_DIR = tempfile.mkdtemp()

@pytest.fixture
def config_file():
    '''Create a config file for the test.
    '''
    config_file = os.path.join(TEST_DIR, 'config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'input_file': os.path.join(TEST_DIR, 'input.csv'),
            'output_file': os.path.join(TEST_DIR, 'output.csv'),
            'model_id': 'model_id',
            'eval_id': 'eval_id',
            'kpi': 'precision',
            'kpi_args': {},
        }, f)
    return config_file

@pytest.fixture
def input_file():
    '''Create an input file for the test.
    '''
    input_file = os.path.join(TEST_DIR, 'input.csv')
    with open(input_file, 'w') as f:
        f.write('y_true,y_score\n')
        for i in range(10):
            f.write(f'{i},{i}\n')
    return input_file

@pytest.fixture
def output_file():
    '''Create an output file for the test.
    '''
    output_file = os.path.join(TEST_DIR, 'output.csv')
    with open(output_file, 'w') as f:
        f.write('model_id,eval_id,kpi,kpi_args,value\n')
    return output_file

@pytest.fixture
def batched_metric_calculator(config_file, input_file, output_file):
    '''Create a BatchedMetricCalculator for the test.
    '''
    return BatchedMetricCalculator(config_file)

@pytest.fixture
def batched_metric_calculator_with_kpi_args(config_file, input_file, output_file):
    '''Create a BatchedMetricCalculator for the test.
    '''
    return BatchedMetricCalculator(config_file, kpi_args={'k': 5})

def test_process_batch(batched_metric_calculator):
    '''Test the process_batch method assuming the kpi is precision.
    '''
    df = pd.DataFrame({
        'y_true': [0, 1, 0, 1, 0, 1, 0, 1],
        'y_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8],
    })
    kpi_output = batched_metric_calculator.process_batch(df)
    assert batched_metric_calculator._y_true == [0, 1, 0, 1, 0, 1, 0, 1]
    assert batched_metric_calculator._y_score == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8]
    assert kpi_output == 0.5

def test_process_batch_with_kpi_args(batched_metric_calculator_with_kpi_args):
    '''Test the process_batch method assuming the kpi is precision.
    '''
    df = pd.DataFrame({
        'y_true': [0, 1, 0, 1, 0, 1, 0, 1],
        'y_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8],
    })
    kpi_output = batched_metric_calculator_with_kpi_args.process_batch(df)
    assert batched_metric_calculator_with_kpi_args._y_true == [0, 1, 0, 1, 0, 1, 0, 1]
    assert batched_metric_calculator_with_kpi_args._y_score == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8]
    assert kpi_output == 0.5

def test_process_batch_with_kpi_args_error(batched_metric_calculator):
    '''Test the process_batch method assuming the kpi is precision.
    '''
    df = pd.DataFrame({
        'y_true': [0, 1, 0, 1, 0, 1, 0, 1],
        'y_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8],
    })
    with pytest.raises(ValueError):
        batched_metric_calculator.process_batch(df)