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


## Refernces

Katz, Garrett. "TransformerHolstep.ipynb" Deep Automated Theorem Proving. CIS 700, Spring 2023.

Katz, Garrett. "TransformerMetamathV2.ipynb" Deep Automated Theorem Proving. CIS 700, Spring 2023.

Hunter, J. D. "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, vol. 9, no. 3, 2007, pp. 90-95.

Harris, C.R., Millman, K.J., et al. Array programming with NumPy. Nature 585, 2020, pp. 357–362. https://doi.org/10.1038/s41586-020-2649-2

Paszke, A., Gross, S., et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32, Curran Associates, Inc., pp. 8024–8035. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

Zhu, Ligeng. "pytorch-OpCounter". GitHub, https://github.com/Lyken17/pytorch-OpCounter/ 