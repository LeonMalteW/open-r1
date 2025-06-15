from aenum import extend_enum
import numpy as np
from Bio import Align
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    SampleLevelMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics


match_score = 2
mismatch_score = -1
open_gap_score = -2.0
extend_gap_score = -0.5

aligner = Align.PairwiseAligner()
aligner.mode = "local"
aligner.match_score = match_score
aligner.mismatch_score = mismatch_score
aligner.open_gap_score = open_gap_score
aligner.extend_gap_score = extend_gap_score
aligner.target_end_gap_score = 0.0
aligner.query_end_gap_score = 0.0


def alignment_data_preparator(
    predictions: list[str], formatted_doc: Doc, aligner_obj=aligner, **kwargs
):
    gold_rule = formatted_doc.choices[0]
    predicted_rule = predictions[0]

    alignments = aligner_obj.align(gold_rule, predicted_rule)
    if not alignments:
        return None

    return alignments[0].score


ALIGNMENT_ACCURACY = SampleLevelMetric(
    metric_name="alignment_accuracy",
    sample_level_fn=alignment_data_preparator,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    corpus_level_fn=np.nanmean,
)


def prompt_fn(line, task_name: str):
    concise_instruction = "\n\nProvide only the Prolog rule as your answer, without any explanation or additional text.\n"
    return Doc(
        task_name=task_name,
        query=line["prompt"] + concise_instruction,
        choices=[line["ground-truth rule"]],
        gold_index=0,
        original_query=line["validation program"],
    )


extend_enum(Metrics, "ALIGNMENT_ACCURACY", ALIGNMENT_ACCURACY)

K_VALUES = [1, 32, 64]

TASKS_TABLE = []
CURRICULUM_TIERS = ["easy", "basic", "medium", "hard"]

for tier in CURRICULUM_TIERS:
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"V-LOL-Benchmark-Local-Alignment-Tier-{tier}",
            suite=["custom"],
            prompt_function=prompt_fn,
            hf_repo="ahmad21omar/MetaBench",
            hf_subset="default",
            hf_filter=lambda line, current_tier=tier: line["curriculum tier"].lower()
            == current_tier.lower(),
            hf_avail_splits=["train", "test", "eval"],
            evaluation_splits=["test"],
            generation_size=max(K_VALUES),
            few_shots_split=None,
            few_shots_select=None,
            metric=[Metrics.PROLOG_PASS_METRIC],
            trust_dataset=True,
        )
    )
