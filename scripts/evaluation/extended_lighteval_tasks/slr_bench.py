from typing import List

from datasets import get_dataset_config_names
from evaluate import load
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc

symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")

from aenum import extend_enum
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def prompt_fn(line: dict, task_name: str):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[line["validation program"]],
        gold_index=0,
    )


def symbolic_judge_sample_metric(
    predictions: List[str], golds: List[str], formatted_doc: Doc, **kwargs
) -> dict:
    return {
        "prediction": predictions[0],
        "gold": golds[0],
    }


def symbolic_judge_corpus_aggregation(items: list) -> dict:
    predictions = [item["prediction"] for item in items]
    golds = [item["gold"] for item in items]

    references = [
        {
            "validation_program": gold,
            "evaluation_config": {
                "positive_predicate": "eastbound",
                "negative_predicate": "westbound",
            },
        }
        for gold in golds
    ]

    results = symbolic_judge.compute(predictions=predictions, references=references)
    return {
        "accuracy": results["accuracy"],
        "partial_score": results["partial_score"],
        "syntax_score": results["syntax_score"],
    }



def create_symbolic_judge_corpus_aggregation(n_samples: int):

    def symbolic_judge_corpus_aggregation_n(items: list[dict]) -> dict:

        def group_by_nth_element(data_list, n):
            grouped_lists = []
            for i in range(n):
                grouped_lists.append(data_list[i::n])
            return grouped_lists

        sample_list = group_by_nth_element(items, n_samples)

        sample_results = []
        for sample in sample_list:
            sample_results.append(symbolic_judge_corpus_aggregation(sample))

        sample_results.sort(key=lambda x: x["accuracy"], reverse=True)

        results = sample_results[0]

        return {
            f"accuracy_1_{n_samples}n": results["accuracy"],
            f"partial_score_1_{n_samples}n": results["partial_score"],
            f"syntax_score_1_{n_samples}n": results["syntax_score"],
        }

    return symbolic_judge_corpus_aggregation_n


sampling_ratios = [1, 4, 8, 16, 32]

for n in sampling_ratios:
    metric = SampleLevelMetric(
        metric_name=f"symbolic_judge_1_{n}n",
        higher_is_better=True,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        sample_level_fn=symbolic_judge_sample_metric,
        corpus_level_fn=create_symbolic_judge_corpus_aggregation(n),
    )

    extend_enum(Metrics, f"symbolic_judge_1_{n}n", metric)


available_subsets = get_dataset_config_names("AIML-TUDA/SLR-Bench")

TASKS_TABLE = []


for subset in available_subsets:
    _task_name = f"SLR-Bench:{subset}"
    task_config = LightevalTaskConfig(
        name=_task_name,
        suite=["extended"],
        prompt_function=prompt_fn,
        hf_repo="AIML-TUDA/SLR-Bench",
        hf_subset=subset,
        hf_avail_splits=["validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        metric=[
            Metrics.symbolic_judge_1_1n,
            Metrics.symbolic_judge_1_16n,
            Metrics.symbolic_judge_1_32n,
        ],
        trust_dataset=True,
        generation_size=32768,
        version=1,
    )
    TASKS_TABLE.append(task_config)
