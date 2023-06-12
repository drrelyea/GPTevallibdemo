import yaml
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

def ndcg_score(y_true, y_score, k=None, ignore_ties=False):
    '''Calculate the normalized discounted cumulative gain.
    '''
    from sklearn.metrics import ndcg_score as sk_ndcg_score
    return sk_ndcg_score(y_true, y_score, k=k, ignore_ties=ignore_ties)

def list_existence(y_true, y_score: Sequence[int]):
    '''For each row, calculate whether the integer in y_score is in y_true.
    '''
    return [int(y_score[i] in y_true[i]) for i in range(len(y_true))]

METRICS = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'accuracy': accuracy_score,
    'roc_auc': roc_auc_score,
    'average_precision': average_precision_score,
    'log_loss': log_loss,
    'brier_score_loss': brier_score_loss,
    'balanced_accuracy': balanced_accuracy_score,
    'cohen_kappa': cohen_kappa_score,
    'matthews_corrcoef': matthews_corrcoef,
    'confusion_matrix': confusion_matrix,
    'roc_curve': roc_curve,
    'precision_recall_curve': precision_recall_curve,
    'ndcg': ndcg_score,
    'list_existence': list_existence,
}

class BatchedMetricCalculator:
    '''Takes the output from a model and calculates metrics on it.
    Metrics can be calculated per row or per batch.
    Metrics are written to a file that also contains the model id, the id of the
        evaluation set, the name of the kpi, and the arguments passed to the kpi.
    '''
    def __init__(self, config_file: str):
        '''Uses a config file to identify the input and output files, the model id,
        the eval_id, the kpi, and the arguments passed to the kpi.
        '''
        with open(config_file, 'r') as f:
            self.config: Dict[str, Any] = yaml.load(f)

        assert 'input_file' in self.config
        assert 'output_file' in self.config
        assert 'model_id' in self.config
        assert 'eval_id' in self.config
        assert 'kpi' in self.config
        assert 'kpi_args' in self.config
        assert 'aggregated_output_file' in self.config
    
    def calculate_metrics(self) -> None:
        '''Calculates the metrics and writes them to the output file.
        '''
        for batch in pd.read_csv(self.config['input_file'], chunksize=self.config.get('batch_size', 10000)):
            kpi_values = self.process_batch(batch)
            kpi_values['model_id'] = self.config['model_id']
            kpi_values['eval_id'] = self.config['eval_id']
            kpi_values['kpi'] = self.config['kpi']
            kpi_values['kpi_args'] = self.config['kpi_args']
            kpi_values.to_csv(self.config['output_file'], mode='a', header=False)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        '''Calculates the metrics for a batch.
        '''
        truth_cols = [col for col in batch.columns if col.startswith('truth_')]
        inference_cols = [col for col in batch.columns if col.startswith('inference_')]
        kpi_func = METRICS[self.config['kpi']]
        return kpi_func(batch[truth_cols].values, batch[inference_cols].values, **self.config['kpi_args'])
        
    def aggregate_metrics(self) -> None:
        '''Aggregates the metrics from the output file.
        '''
        df = pd.read_csv(self.config['output_file'])
        df = df.groupby(['model_id', 'eval_id', 'kpi', 'kpi_args']).mean()
        df.to_csv(self.config['aggregated_output_file'], mode='w')

    def compare_models(self, model_id1: str, model_id2: str, eval_id, kpi, kpi_args: Dict[str, Any]) -> None:
        aggregated_df = pd.read_csv(self.config['aggregated_output_file'])
        aggregated_df = aggregated_df.set_index(['model_id', 'eval_id', 'kpi', 'kpi_args'])
        model1_df = aggregated_df.loc[(model_id1, eval_id, kpi, str(kpi_args))]
        model2_df = aggregated_df.loc[(model_id2, eval_id, kpi, str(kpi_args))]
        comparison_method = self.config.get('comparison_method', 'greater')
        if comparison_method == 'greater':
            model1_better = model1_df > model2_df
        elif comparison_method == 'less':
            model1_better = model1_df < model2_df
        else:
            raise ValueError('Comparison method must be greater or less.')
        if model1_better:
            print(f'Model {model_id1} is better than model {model_id2}.')
        else:
            print(f'Model {model_id2} is better than model {model_id1}.')

