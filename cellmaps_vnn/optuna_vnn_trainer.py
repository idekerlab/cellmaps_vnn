import math

import optuna
from optuna.trial import TrialState
from cellmaps_vnn.vnn_trainer import VNNTrainer
import logging

logger = logging.getLogger(__name__)


class OptunaVNNTrainer(VNNTrainer):
    """
    Trainer for neural networks with Optuna optimization.
    """

    def __init__(self, data_wrapper, n_trials=3):
        """
        Initializes the Optuna NN Trainer.

        :param data_wrapper: Wrapper for the training data.
        :type data_wrapper: TrainingDataWrapper
        """
        super().__init__(data_wrapper)
        self._n_trials = n_trials

        user_lr = self.data_wrapper.lr
        lr_candidates = [user_lr, user_lr * 0.1, user_lr * 2, 1.2e-4, 1.5e-4, 1.8e-4, 2e-4, 3e-4, 4e-4,
                         5e-4, 1e-3]
        self._lr_candidates = list(set(lr_candidates))

    def exec_study(self):
        """
        Executes the Optuna study to optimize the model's hyperparameters.

        :returns: Best trial parameters from the Optuna study.
        :rtype: dict
        """
        logger.info("Starting Optuna study...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._train_with_trial, n_trials=self._n_trials)
        return self._print_result(study)

    def _train_with_trial(self, trial):
        """
        Wraps the train_model method to work with Optuna trials.

        :param trial: Current Optuna trial.
        :type trial: optuna.trial.Trial
        :returns: Maximum validation correlation achieved.
        :rtype: float
        """
        self._setup_trials(trial)
        return super().train_model()

    def _setup_trials(self, trial):
        """
        Sets up hyperparameter suggestions for a trial.

        :param trial: Current Optuna trial.
        :type trial: optuna.trial.Trial
        """
        logger.info("Setting up trial parameters...")

        # Learning rate tuning
        self.data_wrapper.lr = trial.suggest_categorical("lr", self._lr_candidates)

        # Genotype hidden layer sizes
        self.data_wrapper.genotype_hiddens = trial.suggest_categorical("genotype_hiddens", [4])

        # Batch size tuning
        batch_size = self.data_wrapper.batchsize
        if batch_size > len(self.train_feature) / 4:
            batch_size = 2 ** int(math.log(len(self.train_feature) / 4, 2))
            self.data_wrapper.batchsize = trial.suggest_categorical("batchsize", [batch_size])

        for key, value in trial.params.items():
            logger.info(f"Parameter {key}: {value}")

    @staticmethod
    def _print_result(study):
        """
        Prints and returns the results of the Optuna study.

        :param study: Optuna study object.
        :type study: optuna.study.Study
        :returns: Best trial parameters.
        :rtype: dict
        """
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics:")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Number of pruned trials: {len(pruned_trials)}")
        logger.info(f"Number of complete trials: {len(complete_trials)}")

        best_trial = study.best_trial
        logger.info(f"Best trial value: {best_trial.value}")
        logger.info(f"Best trial parameters: {best_trial.params}")

        return best_trial.params
