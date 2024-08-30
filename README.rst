Trajectory Prediction Model Comparison
=======================================

This project compares various models in predicting trajectories, including:
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

This project requires the following configuration:

- **Operating System**: Linux (POSIX compliant), tested on Ubuntu
- **Python**: 3.8 or higher
- **CUDA**: GPU support recommended for training (tested with CUDA 11.8)
- **Dependencies**: Listed in ``pyproject.toml``, installable via ``pip``
- **Conda**: Recommended for managing the environment

Installation
^^^^^^^^^^^^

First, clone the repository and navigate into the project directory::

    git clone https://github.com/va42xuqi/master_den.git
    cd master_den

Set up a Conda environment::

    conda create -n trajectory-env python=3.8
    conda activate trajectory-env

Install dependencies using ``pip`` from the ``pyproject.toml``::

    pip install .

Ensure that your environment is configured for GPU usage if available, with PyTorch installed as follows::

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

.. note::

    This project is tested and guaranteed to work on Linux environments only. Performance and compatibility on other operating systems are not ensured.

Usage
-----

Prepare your dataset according to the format described in the project documentation. The project can be run in various modes, including training, animation, and benchmarking.

To run the training script, use the following command::

    python train.py --arch <model_name> --mode <mode> --scene <scene> --pred_len 100 --hist_len 50 --logger wandb

Replace ``<model_name>`` with the desired model (e.g., ``linear``, ``lstm``, ``lmu``, ``transformer``, ``bitnet``) and set the appropriate ``mode`` and ``scene`` parameters.

For animations or benchmarks, adjust the mode parameter accordingly::

    python train.py --arch lstm --mode animate --scene NBA

Results
-------

The results of the model comparisons are stored in the ``results/`` directory. You can find metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and others. The project also includes graphical evaluations comparing the predicted trajectories against ground truth.

Contributing
------------

If you would like to contribute to this project, please fork the repository and submit a pull request. Issues and suggestions are welcome.

License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for more details.

