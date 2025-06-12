from typing import List

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

from janus_swi_evaluator import evaluate, EvaluationConfig
from janus_swi_evaluator.prolog_types import (
    default_evaluation_metrics,
)


def prolog_data_preparator(
    predictions: List[str], formatted_doc: Doc, golds: List[str]
):
    try:
        return evaluate(
            config=EvaluationConfig(
                PrologTimeout=10,
                EnableHealing=True,
                Fallback=None,
                EnableExtracting=True,
                MinScore=20,
            ),
            facts=[formatted_doc.original_query],
            ground_truth_rules=golds,
            predicted_rules=predictions,
        )
    except Exception as e:
        print(e)
        return {}


metric_names = [k for k, v in default_evaluation_metrics.items()]
prolog_metric = SampleLevelMetricGrouping(
    metric_name=metric_names,
    sample_level_fn=prolog_data_preparator,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.REASONING,
    higher_is_better={name: True for name in metric_names},
    corpus_level_fn={name: np.mean for name in metric_names},
)

extend_enum(Metrics, "PROLOG_METRIC", prolog_metric)


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
    generation_size=1,
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.PROLOG_METRIC],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
