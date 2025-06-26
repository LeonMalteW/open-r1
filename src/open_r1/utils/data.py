import logging
from typing import Dict

import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)

        # Add necessary fields for symbolic judge if missing
        if "validation_program" not in dataset.column_names:
            logger.info("Adding symbolic judge fields to dataset")
            dataset = dataset.map(lambda x: {
                "validation_program": create_validation_program(x),
                "evaluation_config": get_evaluation_config(x)
            })

        return dataset

    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []
        symbolic_judge_datasets = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )

            # Add symbolic judge fields if needed
            if "validation_program" not in ds.column_names:
                logger.info(f"Adding symbolic judge fields to {dataset_config.id}")
                ds = ds.map(lambda x: {
                    "validation_program": create_validation_program(x),
                    "evaluation_config": get_evaluation_config(x)
                })
                symbolic_judge_datasets.append(dataset_config.id)

            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)

            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(f"Subsampled to {len(ds)} examples (weight={dataset_config.weight})")

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created mixture with {len(combined_dataset)} examples")

            # Log symbolic judge datasets
            if symbolic_judge_datasets:
                logger.info(f"Symbolic judge datasets: {', '.join(symbolic_judge_datasets)}")

            if args.dataset_mixture.test_split_size is not None:
                split = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size,
                    seed=seed
                )
                logger.info(f"Split into train ({len(split['train'])}) and test ({len(split['test'])})")
                return split
            else:
                return DatasetDict({"train": combined_dataset})

        raise ValueError("No datasets loaded")

    raise ValueError("Provide dataset_name or dataset_mixture")


def create_validation_program(example: Dict) -> str:
    """Construct validation program for symbolic judge"""
    # Customize this based on your dataset structure
    program = []

    # Add background knowledge
    if "background" in example:
        program.append(example["background"])

    # Add positive examples
    if "positive_examples" in example:
        for ex in example["positive_examples"]:
            program.append(f"{example['positive_predicate']}({ex}).")

    # Add negative examples
    if "negative_examples" in example:
        for ex in example["negative_examples"]:
            program.append(f"{example['negative_predicate']}({ex}).")

    return "\n".join(program)


def get_evaluation_config(example: Dict) -> Dict:
    """Get evaluation config for symbolic judge"""
    # Customize this based on your dataset
    config = {
        "positive_predicate": example.get("positive_predicate", "eastbound"),
        "negative_predicate": example.get("negative_predicate", "westbound")
    }

    # Add problem-specific parameters
    if "parameters" in example:
        config.update(example["parameters"])

    return config