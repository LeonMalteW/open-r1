from aenum import extend_enum
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    SampleLevelMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

import re
from janus_swi_evaluator import evaluate, EvaluationConfig, extract_pattern_match


def prolog_data_preparator(predictions: list[str], formatted_doc: Doc, **kwargs):
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError("predictions must be a single-element list")

    if type(predictions) != list and len(predictions) != 1:
        raise ValueError("Invalid input of prolog_data_preparator")

    results = -1
    try:

        answer = extract_pattern_match(predictions[0], formatted_doc.choices[0])
        if answer is not None:
            results = evaluate(
                config=EvaluationConfig(
                    PrologTimeout=10,
                    EnableHealing=True,
                    Fallback=None,
                ),
                facts=[formatted_doc.original_query],
                ground_truth_rules=formatted_doc.choices,
                predicted_rules=[answer],
            )
    except ValueError as e:
        print(e)

    return {
        "golds": formatted_doc.choices,
        "predictions": predictions,
        "prolog_eval": results,
    }


def prolog_accuracy_aggregator(items: list[dict]):
    eval_results = [item["prolog_eval"] for item in items if item["prolog_eval"] != -1]
    if not eval_results:
        return 0.0
    return sum(eval_results) / len(eval_results)


prolog_metric = SampleLevelMetric(
    metric_name="prolog_accuracy",
    sample_level_fn=prolog_data_preparator,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    corpus_level_fn=prolog_accuracy_aggregator,
)

extend_enum(Metrics, "PROLOG_METRIC", prolog_metric)


def prompt_fn(line, task_name: str):
    return Doc(
        task_name=task_name,
        query=line["prompt"] + "\n" + line["data"],
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
    evaluation_splits=["test"],
    generation_size=1,
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.PROLOG_METRIC],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
