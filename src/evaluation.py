import logging

import hydra
from omegaconf import DictConfig
from tabulate import tabulate

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.evaluators import LocalizationEvaluation
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(
        config.datamodule,
        epoch_counter=epoch_counter, drop_last=False
    )
    datamodule.prepare_data()
    dataset = datamodule.test_set if config.split == "test" else datamodule.val_set
    
    if config.task == "localization":
        evaluator = LocalizationEvaluation(config.prediction_path, dataset)
        rmses = evaluator.get_rmse_all_los_nlos()
        accuracies = evaluator.get_accuracy_all_los_nlos(config.allowable_errors)
        log.info(("\n" + tabulate(zip(["All", "LOS", "NLOS"], rmses), headers=["RMSE"])))
        log.info(
            "\n" + tabulate(
                [
                    ["All"] + accuracies[0],
                    ["LOS"] + accuracies[1],
                    ["NLOS"] + accuracies[2]
                ],
                headers=[f"{t}m acc" for t in config.allowable_errors]
            )
        )

    else:
        raise ValueError(f"Unknown task <{config.task}")
