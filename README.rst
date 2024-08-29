Trajectory Prediction Model Comparison
=======================================

This project is focused on comparing various models in predicting trajectories, including:
- **Naive Models**: Linear Regression
- **Traditional Models**: Long Short-Term Memory (LSTM), Legendre Memory Units (LMU)
- **Novel Strategies**: Transformer, BitNet

The project evaluates these models based on their performance in forecasting human motion trajectories, with a particular focus on complex and dynamic environments such as sports.

Contents
--------

- `Project Overview <#project-overview>`_
- `Models <#models-included>`_
- `Setup Instructions <#setup-instructions>`_
- `Usage <#usage>`_
- `Results <#results>`_
- `Contributing <#contributing>`_
- `License <#license>`_

Project Overview
----------------

This project provides a framework for testing and comparing different neural network architectures and traditional models in predicting human motion trajectories. The aim is to identify which models perform best in terms of accuracy, robustness, and computational efficiency.

Models Included
---------------

- **Linear Regression**: A simple baseline model.
- **LSTM**: A popular recurrent neural network model.
- **LMU**: A newer architecture designed to remember longer sequences.
- **Transformer**: An attention-based model originally designed for NLP tasks.
- **BitNet**: A binary neural network optimized for efficiency.

Setup Instructions
------------------

Prerequisites
^^^^^^^^^^^^^

The project requires a Linux operating system (tested on Ubuntu), Python 3.8 or higher, and Conda is recommended for managing dependencies.

Installation
^^^^^^^^^^^^

Clone the repository and navigate into the project directory::

    git clone https://github.com/your-username/trajectory-prediction.git
    cd trajectory-prediction

Set up a Conda environment::

    conda create -n trajectory-env python=3.8
    conda activate trajectory-env

Install dependencies using ``pip`` from the ``pyproject.toml``::

    pip install .

.. note::

    This project has been tested and is guaranteed to work on Linux environments only. Performance and compatibility on other operating systems are not ensured.

Usage
-----

Prepare your dataset according to the format described in the project documentation. Run the training script using the following command::

    python train.py --model <model_name> --dataset <path_to_dataset>

Replace ``<model_name>`` with the desired model (``linear``, ``lstm``, ``lmu``, ``transformer``, ``bitnet``).

To evaluate the models, use the following command::

    python evaluate.py --model <model_name> --dataset <path_to_dataset>

Results
-------

The results of the model comparisons are stored in the ``results/`` directory. You can find metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and others. The project also includes graphical evaluations comparing the predicted trajectories against ground truth.

Contributing
------------

If you would like to contribute to this project, please fork the repository and submit a pull request. Issues and suggestions are welcome.

License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for more details.

.. image:: images/model_comparison.png
   :alt: Model Comparison Graph
