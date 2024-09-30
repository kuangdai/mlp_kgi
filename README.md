# Knot Gathering Initialization (KGI)

**Note: This version is prepared for double-blind peer review of the following paper**

2985: Enhancing Performance of Multilayer Perceptrons by Knot-Gathering Initialization

## Usage

To use KGI, you only need the `kgi.py` file, which is built solely on PyTorch.

* `apply_kgi_to_layer`: This function applies KGI to a single `nn.Linear` layer.
* `apply_kgi_to_model`: This function applies KGI to a module, automatically skipping non-`nn.Linear` layers.

## Reproducing Experiments

The provided Jupyter Notebooks allow for the reproduction of all experiments. The necessary dependencies are listed in `requirements.txt`.

Training logs are included in this repository, and all the figures in the paper (located in the `figs/` directory) can be reproduced by running all cells in the respective notebooks. 
If you want to re-train the models, simply modify the output directories. Training is performed by checking for the existence of training logs, so changing the output directories ensures the models are trained from scratch.

For the GPT-2 experiment, follow additional instructions provided in `07_gpt2/readme.md`.

### Files Not Included Due to Size Constraints

Due to size limitations for supplementary materials, the following files are not included in this repository:

* `datasets/muons.pkl`: This dataset is required for `04_muons.ipynb` to reproduce `figs/muons_data.pdf` and for training.
* `results/disent_models/*.pt`: Pretrained model weights required for `06_disentanglement.ipynb` to reproduce `figs/disent_latent.pdf`.

To comply with ICLR’s policy, we are unable to provide direct URLs for these files. Please request them via OpenReview, and we will share the files through an anonymous, temporary file transfer service.
