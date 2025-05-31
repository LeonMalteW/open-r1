import multiprocessing
import time
import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    CorpusLevelMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from prolog.PrologEvaluator import is_valid_prolog, eva
from prolog.PrologParser import extract_ground_truth, extract_trains, extract_predicted


def prolog_data_preparator(predictions: list[str], formatted_doc: Doc, **kwargs):

    if type(predictions) != list and len(predictions) != 1:
        raise ValueError("Invalid input of prolog_data_preparator")

    ground_truth = formatted_doc.choices[0]
    original_prompt = formatted_doc.instruction
    original_data = formatted_doc.query.replace(original_prompt, "", 1).strip()

    prolog_specific_data = {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "prompt": original_prompt,
        "data": original_data,
        "quick_validity_check": is_valid_prolog(predictions[0]),
    }

    return {
        "golds": [ground_truth],
        "predictions": predictions,
        "prolog_data": prolog_specific_data,
    }


def custom_corpus_level_accuracy_calculator(all_sample_data: list[dict]):
    tasks_for_pool = []

    _is_valid_prolog = []
    for sample_data_wrapper in all_sample_data:
        sample_data = sample_data_wrapper["prolog_data"]

        trains_facts = extract_trains(sample_data["data"])
        ground_truth = extract_ground_truth(sample_data["ground_truth"])
        predicted_rules = extract_predicted(sample_data["predictions"])

        _is_valid_prolog.append(sample_data["quick_validity_check"])

        for predicted_rule in predicted_rules:
            tasks_for_pool.append((trains_facts, ground_truth, predicted_rule))

    num_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores) as pool:
        raw_eva_results = pool.starmap(eva, tasks_for_pool)
    precisions = []
    recalss = []
    f1s = []
    accuracies = []
    entropy_scores = []

    for res in raw_eva_results:
        if "error" in res:
            print(res["error"])
        if "precision" in res:
            precisions.append(res["precision"])
        if "recall" in res:
            recalss.append(res["recall"])
        if "f1" in res:
            f1s.append(res["f1"])
        if "acc" in res:
            accuracies.append(res["acc"])
        if "entropy_score" in res:
            entropy_scores.append(res["entropy_score"])

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalss) if recalss else 0
    mean_f1 = np.mean(f1s) if f1s else 0
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_entropy_score = np.mean(entropy_scores) if entropy_scores else 0

    return {
        "complex_accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "entropy_score": mean_entropy_score,
        "incorect": _is_valid_prolog.count(False),
    }


prolog_metric = CorpusLevelMetric(
    metric_name="complex_accuracy",
    sample_level_fn=prolog_data_preparator,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=custom_corpus_level_accuracy_calculator,
    higher_is_better=True,
)

extend_enum(Metrics, "prolog_metric", prolog_metric)


def prompt_fn(line, task_name: str):
    return Doc(
        task_name=task_name,
        query=line["prompt"] + " " + line["data"],
        choices=[line["answer"]],
        gold_index=0,
        instruction=line["prompt"],
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
    metric=[Metrics.prolog_metric],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
