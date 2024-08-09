import shutil
import yaml
from pathlib import Path
from ml_collections import ConfigDict
import jax
import os
from absl import logging

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# jax.config.update('jax_debug_nans', True)

from jax_trainer.datasets import build_dataset_module
from jax_trainer.examples.img_classifier import ImgClassifierTrainer

def colored_str(s: str, stype: str):
    if stype.lower() == 'info':
        return f'\033[0;34m{s}\033[0m'
    elif stype.lower() == 'warning':
        return f'\033[0;33m{s}\033[0m'
    elif stype.lower() == 'error':
        return f'\033[0;31m{s}\033[0m'
    else:
        return s


def main():
    config = yaml.safe_load(Path("jax_trainer/examples/cifar10_cls.yaml").read_text())
    config = ConfigDict(config)

    config.dataset.batch_size *= jax.device_count()
    dataset = build_dataset_module(config.dataset)
    sample_input = next(iter(dataset.train_loader))

    # set logging level to INFO
    # logging.basicConfig(level=logging.INFO)

    # logging.set_verbosity(logging.INFO)
    logging.info('[INFO] tmp.py: dataset.train_loader')
    logging.warning(colored_str('[WARNING] tmp.py: dataset.train_loader', stype='warning'))
    logging.error(colored_str('[ERROR] tmp.py: dataset.train_loader', stype='error'))

    trainer = ImgClassifierTrainer(
        trainer_config=config.trainer,
        model_config=config.model,
        optimizer_config=config.optimizer,
        data_module=dataset,
        sample_input=sample_input,
    )

    eval_metrics = trainer.train_model()
    print(eval_metrics)

    test_metrics = trainer.test_model(dataset.test_loader)
    print(test_metrics)


if __name__ == "__main__":
    main()