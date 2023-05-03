# Setup Guide

## Package Installation

1. Install a stable version of `PyTorch` (tested with version 1.13.1+cu116): https://pytorch.org/get-started/locally/
2. Install the latest version of `MatPlotLib`: https://pypi.org/project/matplotlib/
3. Install `thop` to track FLOPs and model parameter counts: https://pypi.org/project/thop/

## Download and setup dataset

1. The HolStep dataset may be download [here](http://cl-informatik.uibk.ac.at/cek/holstep/).
2. Create a new folder called `data` in the project root directory.
2. After download and extracing the HolStep dataset, place the  `holstep` folder under the `data` folder.
3. you should have a folder called `data` with the following structure in the project root directory:

```
data
├── holstep
│   ├── train
│   │   ├── 01345
│   │   ├── .....
│   ├── test
│   │   ├── 01345
│   │   ├── .....
```

## Train and evaluate the models
1. To train and evaluate the models, run the following command in the project root directory:
```
python epoch.py
```

2. We mainly train the SiameseCNNLSTM model and the SiameseTransformer models. To pick the specific model for faster completion, you can comment out the other models in the `epoch.py` file, under the `if __name__ == '__main__':` section.

3. You can choose to switch between cpu and gpu by changing the `device` variable in the top of the `epoch.py` file.
