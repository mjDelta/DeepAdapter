# DeepAdapter
## A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome
Codes and tutorial for [A self-adaptive and versatile tool for eliminating multiple undesirable variations from transcriptome](https://www.biorxiv.org/content/10.1101/2024.02.04.578839v1).

We make an one-line code to utilize DeepAdapter for convenient usage.
```sh
$ from deepAdapter import run as RUN
$ trainer = RUN.train(,
    train_list = train_list,
    val_list = val_list,
    test_list = test_list,
    label2unw = label2bat,
    label2wnt = bio_label2bat,
    net_args = net_args,
    out_dir = out_dir)
```

# Get started
## Re-train DeepAdapter with the provided example datsets or your own dataset
**Step 1**: download the codes
```sh
$ # Clone this repository to your local computer
$ git clone https://github.com/mjDelta/DeepAdapter.git
$ cd DeepAdapter
```
**Step 2**: install the supported packages
```sh
$ # Create a new conda environment
$ conda create -n DA python=3.9
$ # Activate environment
$ conda activate DA
$ # Install dependencies
$ pip install -r requirements.txt
$ # Launch jupyter notebook
$ jupyter notebook
```
**Step 3**: double-click to open tutorials
* `DA-Example-Tutorial.ipynb`: the tutorial of re-training DeepAdapter using the example dataset;
* `DA-YourOwnData-Tutorial.ipynb`: the tutorial of training DeepAdapter using your own dataset.

**After opening the tutorials, please press Shift-Enter to execute a "cell" in `.ipynb`.**
