import optax
from absl.testing import absltest
from ml_collections import ConfigDict

from jax_trainer.datasets import DatasetModule, build_dataset_module


class TestBuildDatasets(absltest.TestCase):
    # Test if constructing various optimizers work

    def test_build_cifar10(self):
        dataset_config = {
            "constructor": "jax_trainer.datasets.build_cifar10_datasets",
            "batch_size": 128,
            "num_workers": 4,
            "data_dir": "data/",
        }
        dataset_config = ConfigDict(dataset_config)
        dataset_module = build_dataset_module(dataset_config)
        self.assertTrue(isinstance(dataset_module, DatasetModule))

        for loaders in [
            dataset_module.train_loader,
            dataset_module.valid_loader,
            dataset_module.test_loader,
        ]:
            batch = next(iter(loaders))
            self.assertEqual(batch.size, 128)
            self.assertEqual(batch.input.shape, (128, 32, 32, 3))
            self.assertEqual(batch.target.shape, (128,))
            self.assertEqual(batch.input.dtype, "float32")
            self.assertEqual(batch.target.dtype, "int64")
