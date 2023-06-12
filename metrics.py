import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from typing import List, Sequence, Dict, Any, Union


def list_existence(y_true: Sequence[int], y_pred: Sequence[Sequence[int]]) -> List[int]:
    """
    Compute list_existence for a list of predictions and truth values
    Args:
        y_true: List of truth values
        y_pred: List of predictions, each being a list of integers
    Returns:
        list_existence values, 1 if the truth exists in the prediction, 0 otherwise
    """
    return [1 if truth in preds else 0 for truth, preds in zip(y_true, y_pred)]


def dcg_at_k(r: Sequence[float], k: int) -> float:
    """
    Compute DCG at rank k for a list of relevance scores
    Args:
        r: List of relevance scores
        k: Rank
    Returns:
        DCG value
    """
    r_np = np.asfarray(r)[:k]
    if r_np.size:
        return np.sum(r_np / np.log2(np.arange(2, r_np.size + 2)))
    return 0.

def ndcg_at_k(r: Sequence[float], k: int) -> float:
    """
    Compute NDCG at rank k for a list of relevance scores
    Args:
        r: List of relevance scores
        k: Rank
    Returns:
        NDCG value
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if dcg_max == 0.0:
        return 1.0
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


# Mapping KPI names to their corresponding functions
METRICS = {
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "mse": mean_squared_error,
    "ndcg": ndcg_at_k,
    "list_existence": list_existence,
    # "custom": calculate_custom_metric
}

KPI_COMPARISONS = {
    'precision': 'greater',
    'recall': 'greater',
    'f1': 'greater',
    'mse': 'less',
    'ndcg': 'greater',
    'list_existence': 'greater',
    # Add more KPIs here as needed
}


class BatchedMetricCalculator:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as ymlfile:
            self.cfg: Dict[str, Any] = yaml.safe_load(ymlfile)

        # Make sure the required keys exist
        assert "input_file" in self.cfg, "Config file missing 'input_file'"
        assert "output_file" in self.cfg, "Config file missing 'output_file'"
        assert "model_id" in self.cfg, "Config file missing 'model_id'"
        assert "eval_id" in self.cfg, "Config file missing 'eval_id'"
        assert "kpi" in self.cfg, "Config file missing 'kpi'"

        assert self.cfg["kpi"] in METRICS, f"Unsupported KPI {self.cfg['kpi']}"

        # Load the optional arguments for the KPI function
        self.kpi_args = self.cfg.get('kpi_args', {})

    def calculate_metrics(self) -> None:
        # Read the input file in batches        
        for batch in pd.read_csv(self.cfg["input_file"], chunksize=self.cfg.get("batch_size", 10000)):
            kpi_values = self.process_batch(batch)

            # Check if kpi_values is an array or a single number
            if isinstance(kpi_values, np.ndarray):
                # If kpi_values is an array, proceed as before
                output_data = pd.DataFrame({
                    'model_id': [self.cfg['model_id']] * len(kpi_values),
                    'eval_id': [self.cfg['eval_id']] * len(kpi_values),
                    'kpi': [self.cfg['kpi']] * len(kpi_values),
                    'kpi_args': [str(self.kpi_args)] * len(kpi_values),
                    'kpi_value': kpi_values
                })
            else:
                # If kpi_values is a single number, create a DataFrame with a single row
                output_data = pd.DataFrame({
                    'model_id': [self.cfg['model_id']],
                    'eval_id': [self.cfg['eval_id']],
                    'kpi': [self.cfg['kpi']],
                    'kpi_args': [str(self.kpi_args)],
                    'kpi_value': [kpi_values]
                })

            output_data.to_csv(self.cfg["output_file"], mode='a', index=False)

    def process_batch(self, batch: pd.DataFrame) -> Union[np.ndarray, float]:
        truth_columns = [col for col in batch.columns if 'truth' in col]
        prediction_columns = [col for col in batch.columns if 'prediction' in col]
        y_true = batch[truth_columns].values
        y_pred = batch[prediction_columns].values

        # Calculate the KPI
        kpi_func = METRICS[self.cfg["kpi"]]
        kpi_values = kpi_func(y_true, y_pred, **self.kpi_args)

        return kpi_values

    def aggregate_kpis(self):
        """ Aggregate KPIs and write them to a new file.

        Args:
            agg_output_file (str): File name to write aggregated KPIs.
        """
        # Make sure the required key exists
        assert "agg_output_file" in self.cfg, "Config file missing 'agg_output_file'"

        # Load KPI values
        df = pd.read_csv(self.cfg["output_file"])
        
        # Aggregate KPIs
        df_agg = df.groupby(['model_id', 'eval_id', 'kpi', 'kpi_args']).agg({'kpi_value': 'mean'}).reset_index()

        # Save the aggregated data
        df_agg.to_csv(self.cfg["agg_output_file"], index=False)

    def compare_models(self, model_id1, model_id2, eval_id, kpi, kpi_args):
        aggregated_kpis = pd.read_csv(self.cfg['agg_output_file'])

        model1_kpi = aggregated_kpis[(aggregated_kpis['model_id'] == model_id1) & (aggregated_kpis['eval_id'] == eval_id) 
                                     & (aggregated_kpis['kpi'] == kpi) & (aggregated_kpis['kpi_args'] == str(kpi_args))]['kpi_value'].values[0]
        model2_kpi = aggregated_kpis[(aggregated_kpis['model_id'] == model_id2) & (aggregated_kpis['eval_id'] == eval_id) 
                                     & (aggregated_kpis['kpi'] == kpi) & (aggregated_kpis['kpi_args'] == str(kpi_args))]['kpi_value'].values[0]
        
        comparison_method = KPI_COMPARISONS[kpi]
        if comparison_method == 'greater':
            if model1_kpi > model2_kpi:
                return model_id1
            else:
                return model_id2
        elif comparison_method == 'less':
            if model1_kpi < model2_kpi:
                return model_id1
            else:
                return model_id2