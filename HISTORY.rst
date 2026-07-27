=======
History
=======

0.3.0 (2026-07-27)
------------------

* Add size attribute to hierarchy nodes indicating the number of genes with data in each node.
* Add functionality to accept input feature files as ro-crate in training and prediction flows.
* Rename model file from model_final.pt to model.pt for consistency.
* Added mode-aware CLI translation with --mode support, RO-Crate validation, and automatic config/model resolution.
* Added config.yaml generation for every training run with used hyperparameters.

0.2.2 (2025-07-25)
------------------

* Bug fixes: fixed hyperparameter optimization and generation of config file with optimal parameters

0.2.1 (2025-07-01)
------------------

* Bug fixes: fixed _annotate_interactomes_of_systems method's return value and fix hierarchy annotations

0.2.0 (2025-06-26)
------------------

* Add annotation in hierarchy nodes of which gene have data for VNN (train)
* Add fake generator of gene importance scores in interactomes of hierarchy system (the real generator will be
  implemented in the future)

0.1.0 (2024-12-26)
------------------

* First release on PyPI.
