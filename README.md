# DeepAdapter
## A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome
Codes and tutorial for [A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome](https://www.biorxiv.org/content/10.1101/2024.02.04.578839v1).

## Updates
- the package is available in `pypi` now. Please install it with `pip install deepadapter`

# Get started
## Before training the codes, download our tutorials.
* `DA-Example-Tutorial.ipynb`: the tutorial of re-training DeepAdapter using the example dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-Example-Tutorial.ipynb));
* `DA-YourOwnData-Tutorial.ipynb`: the tutorial of training DeepAdapter using your own dataset ([click here to download](https://github.com/mjDelta/DeepAdapter/blob/main/DA-YourOwnData-Tutorial.ipynb)).

## Re-train DeepAdapter with the provided example datsets or your own dataset
**Step 1**: create a new conda environment
```sh
$ # Create a new conda environment
$ conda create -n DA python=3.9
$ # Activate environment
$ conda activate DA
```
**Step 2**: install the package with `pip`
```sh
$ # Install the our package
$ pip install deepadapter
```
**Step 3**: launch jupyter notebook and double-click to open tutorials
```sh
$ # Launch jupyter notebook
$ jupyter notebook
```
**After opening the tutorials, please press Shift-Enter to execute a "cell" in `.ipynb`.**
