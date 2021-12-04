# PyTorchTimeSeries

This repository demonstrates how to implement tap-delays (TDL) in PyTorch framework for time series modeling. Currently, NARX and NARMAX have been implemented.

NARX   - Nonlinear Autoregressive eXogenous
NARMAX - Nonlinear Autoregressive Moving Average eXogenous 

**NOTE: This repository is a work in progress. Working on more general implementation of TDL. Also the examples will be cleaned up in the future. At this current stage not all test and examples work properly.**

## Project Layout:
* **Example Folder:**
Contains examples for implementing PyTorch NARX and NARMAX modules for time-series modeling on real-life data. Also, included an example for implementing an MRAC in PyTorch which is a neural network controller.

* **Helper Folder:**
Contains helper methods for training NARX and NARMAX networks. Link to the helper methods are shown below

    | Helper Files: | 
    | ------------- |
    | Helper Methods [**NARX**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Helper/InputOutput/narx_helper_methods.py) |
    | Helper Methods [**NARMAX**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Helper/InputOutput/narmax_helper_methods.py) |

* **Model:**
Contains PyTorch implementation of NARX and NARMAX modules for time-series modeling. Link to the modules are shown below.

    | Modules: |
    | ------------ |
    | Modules [**NARX**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Model/InputOutput/narx_model.py) |
    | Modules [**NARMAX**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Model/InputOutput/narmax_model.py) |

* **Optimizer:**
Contains PyTorch classes for optimization approaches shown below.

    | Optimizer Classes: |
    | ------------ |
    | Optimizer [**LM**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Optimizer/lm.py): Levenberg-Marquardt optimization algorithm. NOTE: This isn't an efficient implementation of LM, since pyTorch dosen't allow for gradient calculations with respect to a vector of errors. So the jacobian calculation isn't very efficient. |
    | Optimizer [**SCG**](https://github.com/joekelley120/PyTorchTimeSeries/blob/master/Optimizer/scg.py): Scaled conjugate gradient optimization algorithm. |
