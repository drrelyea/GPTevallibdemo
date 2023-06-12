import pytest
import pandas as pd
import yaml
import os
from metrics import BatchedMetricCalculator, list_existence, ndcg_at_k, dcg_at_k
import ast
import numpy as np

@pytest.fixture
def setup_data():
    input_file = 'input_test.csv'
    output_file = 'output_test.csv'
    agg_output_file = 'aggregated_kpis_test.csv'
    config_file = 'config_test.yml'
    
    # Create a sample input data
    data = {
        'truth': [1, 0, 1, 1, 0],
        'prediction': [1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(input_file, index=False)

    # Create a sample configuration file
    config = {
        'input_file': input_file,
        'output_file': output_file,
        'agg_output_file': agg_output_file,
        'model_id': 'test_model',
        'eval_id': 'test_eval',
        'kpi': 'precision',
        'kpi_args': {},
    }
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    yield input_file, output_file, agg_output_file, config_file

    # Teardown code
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(agg_output_file):
        os.remove(agg_output_file)
    if os.path.exists(config_file):
        os.remove(config_file)

def test_process_batch(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data
    calculator = BatchedMetricCalculator(config_file)
    df = pd.read_csv(input_file)
    result = calculator.process_batch(df)
    assert result == 1.0

def test_calculate_metrics(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data
    calculator = BatchedMetricCalculator(config_file)
    calculator.calculate_metrics()
    output_data = pd.read_csv(output_file)
    assert output_data['kpi_value'][0] == 1.0

def test_output_file_content_and_format(setup_data):

    input_file, output_file, agg_output_file, config_file = setup_data
    calculator = BatchedMetricCalculator(config_file)
    calculator.calculate_metrics()
    output_data = pd.read_csv(output_file)

    assert list(output_data.columns) == ['model_id', 'eval_id', 'kpi', 'kpi_args', 'kpi_value']
    assert output_data['model_id'][0] == 'test_model'
    assert output_data['eval_id'][0] == 'test_eval'
    assert output_data['kpi'][0] == 'precision'
    expected_kpi_args = {}  # assuming no args for precision in config
    assert ast.literal_eval(output_data['kpi_args'][0]) == expected_kpi_args

def test_different_configurations(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data
    with open(config_file, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    config['kpi'] = 'recall'
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    calculator = BatchedMetricCalculator(config_file)
    df = pd.read_csv(input_file)
    result = calculator.process_batch(df)
    assert pytest.approx(result, 0.01) == 0.67

def test_different_kpi_functions(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data
    with open(config_file, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    config['kpi'] = 'f1'
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    calculator = BatchedMetricCalculator(config_file)
    df = pd.read_csv(input_file)
    result = calculator.process_batch(df)
    assert pytest.approx(result, 0.01) == 0.8

def test_list_existence():
    # Model output for 4 items
    y_pred = [[14,15,16,17,18],[1,4,5,7,3],[5,2,6,7,4],[129,4325,453,542,432]]
    
    # Truth data
    y_true = [15,2,1,453]
    
    assert list_existence(y_true, y_pred) == [1,0,0,1]

    # Test when all truth data exists in the predictions
    y_true = [14,1,5,129]
    assert list_existence(y_true, y_pred) == [1,1,1,1]

    # Test when none of the truth data exists in the predictions
    y_true = [100,200,300,400]
    assert list_existence(y_true, y_pred) == [0,0,0,0]

def test_aggregate_kpis(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data
    calculator = BatchedMetricCalculator(config_file)
    calculator.calculate_metrics()
    calculator.aggregate_kpis()
    agg_data = pd.read_csv(agg_output_file)
    assert list(agg_data.columns) == ['model_id', 'eval_id', 'kpi', 'kpi_args', 'kpi_value']
    assert agg_data['model_id'][0] == 'test_model'
    assert agg_data['eval_id'][0] == 'test_eval'
    assert agg_data['kpi'][0] == 'precision'
    expected_kpi_args = {}  # assuming no args for precision in config
    assert ast.literal_eval(agg_data['kpi_args'][0]) == expected_kpi_args
    assert agg_data['kpi_value'][0] == 1.0  # assuming 'precision' was 1.0 for all batches

def test_compare_models(setup_data):
    input_file, output_file, agg_output_file, config_file = setup_data

    # Setting up two models with different precision values
    model1_id = 'test_model1'
    model2_id = 'test_model2'
    eval_id = 'test_eval'
    kpi = 'precision'
    kpi_args = {}
    kpi_value_model1 = 0.9
    kpi_value_model2 = 0.8
    data = {
        'model_id': [model1_id, model2_id],
        'eval_id': [eval_id, eval_id],
        'kpi': [kpi, kpi],
        'kpi_args': [str(kpi_args), str(kpi_args)],
        'kpi_value': [kpi_value_model1, kpi_value_model2],
    }
    df = pd.DataFrame(data)
    df.to_csv(agg_output_file, index=False)

    # Running the comparison
    calculator = BatchedMetricCalculator(config_file)
    winning_model = calculator.compare_models(model1_id, model2_id, eval_id, kpi, kpi_args)

    assert winning_model == model1_id  # As model1 has higher precision

def test_dcg_at_k():
    # Test 1: example case
    r = [3, 2, 3, 0, 1, 2]
    k = 4
    expected_result = 3 + (2/np.log2(3)) + (3/np.log2(4)) + (0/np.log2(5))
    assert np.isclose(dcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected DCG={expected_result}"

    # Test 2: edge case with zero relevance
    r = [0, 0, 0, 0, 0]
    k = 5
    expected_result = 0.0
    assert np.isclose(dcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected DCG={expected_result}"

    # Test 3: edge case with single value
    r = [5]
    k = 1
    expected_result = 5.0
    assert np.isclose(dcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected DCG={expected_result}"


def test_ndcg_at_k():
    # Test 1: example case with perfect ranking
    r = [3, 3, 2, 2, 1, 0]
    k = 6
    expected_result = 1.0  # In this case, the DCG equals the IDCG
    assert np.isclose(ndcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected NDCG={expected_result}"

    # Test 2: example case with imperfect ranking
    r = [3, 2, 3, 0, 1, 2]  # We've switched the first two elements compared to the previous test
    k = 6
    expected_result = dcg_at_k(r, k) / dcg_at_k(sorted(r, reverse=True), k)
    assert np.isclose(ndcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected NDCG={expected_result}"

    # Test 3: edge case with zero relevance
    r = [0, 0, 0, 0, 0]
    k = 5
    expected_result = 1.0  # All zero relevance scores are treated as perfect ranking, so NDCG should be 1
    assert np.isclose(ndcg_at_k(r, k), expected_result), f"For r={r} and k={k}, expected NDCG={expected_result}"
