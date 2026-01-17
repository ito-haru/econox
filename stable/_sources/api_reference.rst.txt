API Reference
=============

.. toctree::
   :maxdepth: 2

Workflow (High-Level API)
-------------------------
The primary interface for managing the estimation workflow.

Estimator
^^^^^^^^^^^^^^^^
.. automodule:: econox.workflow.estimator
   :members:
   :undoc-members:
   :show-inheritance:

Simulator
^^^^^^^^^^^^^^^^
.. automodule:: econox.workflow.simulator
   :members:
   :undoc-members:
   :show-inheritance:

Core Interfaces (Protocols)
---------------------------
Abstract base classes and protocols defining the contract for custom components.

.. automodule:: econox.protocols
   :members:
   :undoc-members:
   :show-inheritance:

Structures (Data & State)
-------------------------
Data containers for models, parameters, and results.

Structural Model
^^^^^^^^^^^^^^^^
.. automodule:: econox.structures.model
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Management
^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.structures.params
   :members:
   :undoc-members:
   :show-inheritance:

Result Containers
^^^^^^^^^^^^^^^^^
.. automodule:: econox.structures.results
   :members:
   :undoc-members:
   :show-inheritance:

Logic (Physics & Rules)
-----------------------
Components defining the economic logic and mechanics of the model.

Utility Functions
^^^^^^^^^^^^^^^^^
.. automodule:: econox.logic.utility
   :members:
   :undoc-members:
   :show-inheritance:

Probability Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.logic.distribution
   :members:
   :undoc-members:
   :show-inheritance:

Feedback Mechanisms
^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.logic.feedback
   :members:
   :undoc-members:
   :show-inheritance:

Transition Dynamics
^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.logic.dynamics
   :members:
   :undoc-members:
   :show-inheritance:

Terminal Approximators
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.logic.terminal
   :members:
   :undoc-members:
   :show-inheritance:

Solvers (Algorithms)
--------------------
Numerical algorithms for solving economic models (Forward Problems).

Dynamic Programming
^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.solvers.dynamic_programming
   :members:
   :undoc-members:
   :show-inheritance:

Equilibrium Solvers
^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.solvers.equilibrium
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Backends
^^^^^^^^^^^^^^^^^^^^^

.. note::
   **Import Note**: This module is not exposed in the top-level ``econox`` namespace.
   You must access it via the submodule:

   .. code-block:: python

      from econox.optim import Minimizer
      # or
      import econox.optim as opt

.. automodule:: econox.optim
   :members:
   :undoc-members:
   :show-inheritance:

Methods (Estimation Techniques)
-------------------------------
Strategies for estimating model parameters from data (Inverse Problems).

Base Classes
^^^^^^^^^^^^
.. automodule:: econox.methods.base
   :members:
   :undoc-members:
   :show-inheritance:

Analytical Methods (OLS/2SLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.methods.analytical
   :members:
   :undoc-members:
   :show-inheritance:

Numerical Methods (MLE/GMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.methods.numerical
   :members:
   :undoc-members:
   :show-inheritance:

Variance & Inference
^^^^^^^^^^^^^^^^^^^^
.. automodule:: econox.methods.variance
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. note::
   **Import Note**: Configuration constants are located in the ``econox.config`` submodule.

   .. code-block:: python

      from econox import config
      print(config.NUMERICAL_EPSILON)

Global settings and constants used throughout the library.

.. automodule:: econox.config
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. note::
   **Import Note**: Helper functions are located in the ``econox.utils`` submodule.

   .. code-block:: python

      from econox.utils import get_from_pytree

General utility functions for array manipulation and PyTree handling.

.. automodule:: econox.utils
   :members:
   :undoc-members:
   :show-inheritance: