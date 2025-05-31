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


def prolog_data_preparator(predictions: list[str], formatted_doc: Doc, **kwargs):

    if type(predictions) != list and len(predictions) != 1:
        raise ValueError("Invalid input of prolog_data_preparator")

    ground_truth = formatted_doc.choices[0]
    original_prompt = formatted_doc.instruction
    original_data = formatted_doc.query.replace(original_prompt, "", 1).strip()

    return {
        "prediction": predictions[0],
        "ground_truth": ground_truth,
        "prompt": original_prompt,
        "data": original_data,
        "quick_validity_check": is_valid_prolog(predictions[0]),
    }


def custom_corpus_level_accuracy_calculator(all_sample_data: list[dict]):
    all_results = []
    tasks_for_pool = []

    for sample_data in all_sample_data:
        if not sample_data["quick_validity_check"]:
            all_results.append((0, 0, 0, 0, 0, "Skipped due to quick validity check"))
            continue

        ground_truth = sample_data["ground_truth"]
        trains_facts = sample_data["data"]
        predicted_rule = sample_data["prediction"]
        tasks_for_pool.append((trains_facts, ground_truth, predicted_rule))

    num_cores = multiprocessing.cpu_count()

    if not tasks_for_pool:
        raise Exception(
            "No valid samples to process with eva. \n"
            f"{all_sample_data[0].get('prediction','NaN')}"
        )
    else:
        with multiprocessing.Pool(processes=num_cores) as pool:
            raw_eva_results = pool.starmap(eva, tasks_for_pool)

        valid_accuracies = [res["acc"] for res in raw_eva_results if "acc" in res]
        final_accuracy_score = np.mean(valid_accuracies) if valid_accuracies else 0.0

    return final_accuracy_score


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
    evaluation_splits=["eval"],
    generation_size=1,
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.prolog_metric, Metrics.rouge_t5, Metrics.bert_score],
    trust_dataset=True,
)

TASKS_TABLE = [CUSTOM_TASK]
