import unittest
from unittest.mock import MagicMock, patch

import torch

from cellmaps_vnn.data_wrapper import TrainingDataWrapper
from cellmaps_vnn.vnn import VNN
from cellmaps_vnn.vnn_trainer import VNNTrainer


class TestVNNTrainer(unittest.TestCase):
    """Tests for module `vnn_trainer` in `cellmaps_vnn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.data_wrapper = MagicMock(spec=TrainingDataWrapper)

        self.data_wrapper.train_feature = torch.randn(100, 10)
        self.data_wrapper.train_label = torch.randint(0, 2, (100,))
        self.data_wrapper.val_feature = torch.randn(40, 10)
        self.data_wrapper.val_label = torch.randint(0, 2, (40,))
        self.data_wrapper.cuda = '1'
        self.data_wrapper.batchsize = 16
        self.data_wrapper.epochs = 5
        self.data_wrapper.lr = 0.00
        self.data_wrapper.delta = 0.01
        self.data_wrapper.modeldir = '/path/to/modeldir'
        self.data_wrapper.cell_features = torch.randn(10, 5)
        self.data_wrapper.alpha = 0.5

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_initialization(self):
        vnn_trainer = VNNTrainer(self.data_wrapper)
        self.assertIsNone(vnn_trainer.model)
        self.assertEqual(vnn_trainer.data_wrapper, self.data_wrapper)
