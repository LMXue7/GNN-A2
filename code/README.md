
# Instructions for reproducing the experiments 

## Environmental Requirement

0. You need to set up the environment for running the experiments (Anaconda 3-2020.10 or above and Python 3.7 or above). First, you can install Anaconda by following the instruction from  the [website](https://docs.anaconda.com/anaconda/install/).
   
   Next, you can create a virtual environment using the following commands:
   <pre><code>$ conda create -n GNN_AA python=3.7 anaconda
   $ conda activate GNN_AA</code></pre>

1. Install Pytorch with version 1.12.0 or later.

   For example (with CPU only version):
   ```
   $ pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
   ```
2. Install `````` package with version 2.3.1.

   *(Note that it may need to appropriatly install the package ```torch-geometric``` based on the CUDA version (or CPU version if GPU is not avaliable). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/2.3.1/notes/installation.html for more information of installing prerequisites of ```torch-geometric```)*

   For example (CUDA=11.3):
   ```
   $ CUDA=cu113
   $ TORCH=1.12.0
   $ pip install  --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   $ pip install  --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   $ pip install  --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   $ pip install  --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   $ pip install torch-geometric==2.3.1
   ```
3. Download the code and data.

   clone this repository:
   ```
   $ git clone https://github.com/LMXue7/GNN-AA.git
   ```

   and go to the folder of the repository:
   ```
   $ cd GNN-AA
   ```

   Now, our source codes are in the folder ```code/``` and the datasets are in the folder ```data/```.

4. Install other packages listed in requirements.txt.
   ```
   $ pip install -r requirements.txt
   ```

## Run the code


Go to the ```code/``` folder and run the ```main.py``` file:
   ```
   $ cd code
   $ python main.py --dataset=ml-1m --num_user_features=4 --dim=64 --hidden_layer=256 
   ```
   Main arguments:
   ```
   --dataset [ml-1m, book-crossing, taobao]: the dataset to run
   --dim: the embedding dimension of each attribute
   --hidden_layer: the MLP hidden layer for the inner interaction modeling function
   --l2_weight: the regularization weight
   --lr: the learning rate
   --num_user_features: the number of user attributes. Currently we assume all users have the same number of attributes. Here are 3 for book-crossing, 8 for taobao and 4 for ml-1m.
   ```
   For more argument options, please refer to ```main.py```
