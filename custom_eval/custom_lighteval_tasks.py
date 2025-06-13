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


K_VALUES = [1, 32, 64]


METRIC_DIRECTIONS = {
    "f1": True,
    "matthews_correlation": True,
    "balanced_accuracy": True,
    "rule_scope_similarity": True,
    "undergeneralization_penalty": False,
    "overgeneralization_penalty": False,
    "pred_fact_consistency": True,
    "gt_fact_consistency": True,
    "avg_healing_steps": False,
}


def create_prolog_pass_metric(k_values: List[int] = K_VALUES):
    """Factory function to create pass@k metrics and detailed metrics"""

    def prolog_data_preparator(
        predictions: List[str], formatted_doc: Doc, golds: List[str]
    ) -> Dict[str, float]:
        results = {}
        successful_k_metrics_per_pred = []
        is_correct_list = []

        for pred in predictions:
            try:
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
                is_correct_list.append(1)

                successful_k_metrics_per_pred.append(
                    {
                        m_name: metrics.get(m_name, 0.0)
                        for m_name in METRIC_DIRECTIONS.keys()
                    }
                )
            except Exception:
                is_correct_list.append(0)
                successful_k_metrics_per_pred.append(
                    {m_name: 0.0 for m_name in METRIC_DIRECTIONS.keys()}
                )

        for k in k_values:

            k_predictions_slice = is_correct_list[: min(k, len(is_correct_list))]
            k_metrics_slice = successful_k_metrics_per_pred[
                : min(k, len(successful_k_metrics_per_pred))
            ]

            passed_at_k = 1 if any(k_predictions_slice) else 0
            results[f"prolog_pass@1:{k}_samples"] = passed_at_k

            for metric_name in METRIC_DIRECTIONS.keys():
                current_k_metric_values = []
                for i in range(len(k_predictions_slice)):
                    if k_predictions_slice[i] == 1:
                        current_k_metric_values.append(
                            k_metrics_slice[i].get(metric_name, 0.0)
                        )

                if current_k_metric_values:
                    if METRIC_DIRECTIONS[metric_name]:
                        best_value_for_k = max(current_k_metric_values)
                    else:
                        best_value_for_k = min(current_k_metric_values)
                else:
                    best_value_for_k = 0.0

                results[f"{metric_name}_pass@1:{k}_samples"] = best_value_for_k

        return results

    metric_names = []
    for k in k_values:
        metric_names.append(f"prolog_pass@1:{k}_samples")
        for metric_name in METRIC_DIRECTIONS.keys():
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
                for metric_name in METRIC_DIRECTIONS.keys()
                for k in k_values
            },
        },
        corpus_level_fn={name: np.mean for name in metric_names},
    )


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
    generation_size=max(K_VALUES),
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.PROLOG_PASS_METRIC],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
