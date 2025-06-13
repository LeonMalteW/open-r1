from typing import List, Dict, Any, Tuple
import numpy as np
from aenum import extend_enum
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from janus_swi_evaluator import evaluate, EvaluationConfig, default_evaluation_metrics


K_VALUES = [1, 32, 64]  # [1, 4, 8, 16, 32, 64]


METRIC_DIRECTIONS = {
    "f1": True,
    "true_positives": True,
    "false_positives": False,
    "false_negatives": False,
    "precision": True,
    "recall": True,
    "accuracy": True,
    "matthews_correlation": True,
    "balanced_accuracy": True,
    "rule_scope_similarity": True,
    "undergeneralization_penalty": False,
    "overgeneralization_penalty": False,
    "pred_fact_consistency": True,
    "gt_fact_consistency": True,
    "avg_healing_steps": False,
    "total_healing_steps": False,
}


def create_prolog_pass_metric(k_values: List[int] = K_VALUES):
    """Factory function to create pass@k metrics and detailed metrics"""

    def prolog_data_preparator(
        predictions: List[str], formatted_doc: Doc, golds: List[str]
    ) -> Dict[str, float]:
        results = {}
        all_metrics = []
        is_correct_list = []

        # Evaluate all predictions
        for pred in predictions:
            try:
                # Try to evaluate the prediction
                metrics = evaluate(
                    config=EvaluationConfig(
                        PrologTimeout=10,
                        EnableHealing=True,
                        Fallback=None,
                        EnableExtracting=True,
                        MinScore=20,
                    ),
                    facts=[formatted_doc.original_query],
                    ground_truth_rules=golds,
                    predicted_rules=[pred],
                )
                # If evaluation succeeded, mark as correct and store metrics
                is_correct_list.append(1)
                all_metrics.append(metrics)
            except Exception:
                # If evaluation failed, mark as incorrect and use zeros for metrics
                is_correct_list.append(0)
                all_metrics.append({m: 0.0 for m in default_evaluation_metrics})

        # Calculate pass@k for each k value
        for k in k_values:
            # Consider only the first k predictions
            k_predictions = is_correct_list[: min(k, len(is_correct_list))]
            k_metrics = all_metrics[: min(k, len(all_metrics))]

            # Check if any prediction is correct
            passed = 1 if any(k_predictions) else 0
            results[f"prolog_pass@1:{k}_samples"] = passed

            # Calculate best metric values for this k
            for metric_name in default_evaluation_metrics:
                # Get all values for this metric in the k-set
                values = [m.get(metric_name, 0.0) for m in k_metrics]

                # For metrics where higher is better, take max value
                if METRIC_DIRECTIONS[metric_name]:
                    best_value = max(values) if values else 0.0
                # For metrics where lower is better, take min value
                else:
                    best_value = min(values) if values else 0.0

                results[f"{metric_name}_pass@1:{k}_samples"] = best_value

        return results

    # Generate metric names for all k values and all metrics
    metric_names = []
    for k in k_values:
        metric_names.append(f"prolog_pass@1:{k}_samples")
        for metric_name in default_evaluation_metrics:
            metric_names.append(f"{metric_name}_pass@1:{k}_samples")

    return SampleLevelMetricGrouping(
        metric_name=metric_names,
        sample_level_fn=prolog_data_preparator,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.REASONING,
        higher_is_better={
            **{f"prolog_pass@1:{k}_samples": True for k in k_values},
            **{
                f"{metric_name}_pass@1:{k}_samples": METRIC_DIRECTIONS[metric_name]
                for metric_name in default_evaluation_metrics
                for k in k_values
            },
        },
        corpus_level_fn={name: np.mean for name in metric_names},
    )


# Create and register the metric
prolog_pass_metric = create_prolog_pass_metric()
extend_enum(Metrics, "PROLOG_PASS_METRIC", prolog_pass_metric)


def prompt_fn(line, task_name: str):
    concise_instruction = "\n\nProvide only the Prolog rule as your answer, without any explanation or additional text."
    return Doc(
        task_name=task_name,
        query=line["prompt"] + "\n" + line["data"] + concise_instruction,
        choices=[line["answer"]],
        gold_index=0,
        original_query=line["data"],
    )


CUSTOM_TASK = LightevalTaskConfig(
    name="V-LOL-Benchmark",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="ahmad21omar/V-LOL-Benchmark",
    hf_subset="default",
    hf_filter=lambda line: int(line["difficulty"][11:]) == 1
    and len(line["prompt"]) < 47000,
    hf_avail_splits=["train", "test", "test_small", "eval"],
    evaluation_splits=["test_small"],
    generation_size=max(K_VALUES),  # Generate enough samples for the largest k
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.PROLOG_PASS_METRIC],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
