import math

import optuna
from optuna.trial import TrialState

from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.vnn_trainer import VNNTrainer
import logging

logger = logging.getLogger(__name__)


class OptunaVNNTrainer(VNNTrainer):
    """
    Trainer for neural networks with Optuna optimization.
    """

    def __init__(self, data_wrapper, n_trials=20, batchsize_vals=None, lr_vals=None, wd_vals=None, alpha_vals=None,
                 genotype_hiddens_vals=None, patience_vals=None, delta_vals=None, min_dropout_layer_vals=None,
                 dropout_fraction_vals=None):
        """
        Initializes the Optuna NN Trainer.

        :param data_wrapper: Wrapper for the training data.
        :type data_wrapper: TrainingDataWrapper
        """
        super().__init__(data_wrapper)
        self._n_trials = n_trials

        self._param_ranges = {
            "batch_size": batchsize_vals,
            "lr": lr_vals,
            "weight_decay": wd_vals,
            "alpha": alpha_vals,
            "genotype_hiddens": genotype_hiddens_vals,
            "patience": patience_vals,
            "delta": delta_vals,
            "min_dropout_layer": min_dropout_layer_vals,
            "dropout_fraction": dropout_fraction_vals
        }

        if all(param is None for param in self._param_ranges.values()):
            raise CellmapsvnnError("At least one parameter value must be provided as a list or range to perform "
                                   "hyperparameter optimization.")

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

        for param_name, param_range in self._param_ranges.items():
            if param_range is not None:
                if isinstance(param_range, list):
                    trial_value = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    trial_value = trial.suggest_uniform(param_name, *param_range)
                else:
                    raise ValueError(f"Invalid parameter range format for {param_name}: {param_range}")

                setattr(self.data_wrapper, param_name, trial_value)
                logger.info(f"Parameter {param_name}: {trial_value}")

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
