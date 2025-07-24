# 🧭 RASA: Removal of Absolute Spatial Attributes

**RASA** is a lightweight, iterative post-processing module that removes absolute spatial bias from Vision Transformer (ViT) patch embeddings. It enhances semantic understanding by disentangling positional information—especially useful in self-supervised learning setups where location labels are absent.

---

## 🌟 Highlights

- ✅ Works **post-hoc** on pretrained ViTs (e.g., DINOv2)
- 🔁 **Iterative debiasing** using Gram-Schmidt orthogonal projections
- 🔌 Modular `RASAHead` integrates into any ViT pipeline
- 📊 Boosts **semantic clustering** and **spatial entropy**
- ⚙️ Compatible with [FRANCA](https://github.com/valeoai/franca) training framework
- 🧠 Built with PyTorch & PyTorch Lightning

---

## 🧠 Method Overview

RASA learns to predict normalized 2D patch coordinates from ViT embeddings. It then removes the components in feature space that are most predictive of position.

1. Train a simple linear regressor to predict 2D patch coordinates.
2. Orthonormalize its weight vectors using Gram-Schmidt.
3. Project patch embeddings onto this 2D subspace and subtract it.
4. Repeat iteratively until position-predictive signal vanishes.

```math
Z^{(t)}_s = Z^{(t-1)}_s - \left\langle Z_s, u_r \right\rangle u_r - \left\langle Z_s, u_c \right\rangle u_c
```

---

## 📦 Installation

RASA is part of the [FRANCA repository](https://github.com/valeoai/franca).

```bash
git clone https://github.com/valeoai/franca.git
cd franca

# Create environment
conda create -n franca python=3.11 ipython
conda activate franca

# Install dependencies
pip install -r requirements.txt
```

> ✅ Optional: For experiment tracking with [Neptune](https://neptune.ai):
```bash
pip install -U neptune==1.13.0
```

---

## ⚙️ Setup

### 🔧 Configuration

To set up training, create or adapt a configuration file.  
An example is provided at:

```
./rasa/experiments/configs/rasa_dito.yml
```

Modify parameters such as:
- when `only_load_weights` is `True`:
    - `train.checkpoint`: path to pretrained DINOv2 
- when `only_load_weights` is `False`:
    - `train.checkpoint`: path to checkpoint of RASA lightning module to continue training
    - In this case `start_pos_layers` can also be adapted to continue training of the rest of linear layers within the RASAhead.
- Learning rate, weight decay, epochs, etc.

### 📁 Dataset Preparation

To prepare your dataset for training and evaluation, refer to [dataset_rasa.md](rasa/dataset_rasa.md).

---

## 🚀 Training

Once your config and dataset are ready, start training:

```bash
# Set paths
export PYTHONPATH="${PYTHONPATH}:./dinov2"
export PYTHONPATH="${PYTHONPATH}:./rasa"

cd rasa
python experiments/train_rasa.py --config_path ./rasa/experiments/configs/rasa_dito.yml
```

---

## 📥 Loading Pretrained Models

### 🔹 Option 1: Load manually

```python
from rasa.src.rasa_head import RASAHead
from dinov2.models import build_model_from_cfg
from omegaconf import OmegaConf
import torch

# Load ViT backbone
config = OmegaConf.merge(
    OmegaConf.load("./dinov2/configs/ssl_default_config.yaml"),
    OmegaConf.load("path/to/dito/model/config.yaml")
    # e.g., ./dinov2/configs/train/LAION/vitl14.yaml
)
backbone, _ = build_model_from_cfg(config, only_teacher=True)

# Load weights
path_to_dito_chkpt = "<YOUR CHECKPOINT TO DITO>"
state_dict = torch.load(path_to_dito_chkpt, map_location="cpu")
if "teacher" in state_dict:
    state_dict = state_dict["teacher"]
state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
backbone.load_state_dict(state_dict, strict=False)

# Load RASA head
rasa_head = RASAHead(input_dim=backbone.embed_dim, n_pos_layers=9, pos_out_dim=2)
path_to_rasa_chkpt = "<YOUR CHECKPOINT TO RASA>"
rasa_head.load_state_dict(torch.load(path_to_rasa_chkpt, map_location="cpu"), strict=False)

# Load Images, here we Create Random Ones
batch = torch.randn(32, 3, 224, 224)
# Apply DITO First to get the patch embeddings
x = backbone.forward_features(batch)["x_norm_patchtokens"]
# Apply RASA Second to remove the positional information
x_debiased = rasa_head(x, use_pos_pred=True, return_pos_info=False)
```

### 🔹 Option 2: Use PyTorch Lightning wrapper

```python
from src.rasa import RASA

model = RASA.load_from_checkpoint("/path/to/rasa_model.ckpt", map_location="cpu")
# Load Images, here we Create Random Ones
batch = torch.randn(32, 3, 224, 224)
# Apply DITO First to get the patch embeddings
x = model.backbone.forward_features(batch)["x_norm_patchtokens"]
# Apply RASA Second to remove the positional information
x_debiased = model.head(x, use_pos_pred=True, return_pos_info=False)
```

---

## 📊 Evaluation

RASA includes multiple evaluation strategies:

- ❓ **How well can embeddings predict position?**
- 🧭 **How position-dependent are features before/after debiasing?**
- 🎯 **Unsupervised segmentation (mIoU) on patch clustering**
- 📉 **Entropy of patch activations across spatial locations**


Evaluation is handled inside the `RASA` LightningModule.

---

## 📂 Project Structure

```
rasa/
├── src/                    
│   ├── rasa.py             # LightningModule
│   ├── transforms.py       # Data Pipeline Transformations 
│   └── rasa_head.py        # RASAHead torch Module
├── experiments/
│   ├── train_rasa.py       # Training entrypoint
│   ├── utils.py            # Training Utils
│   └── configs/
├── data/                   # Setting Datasets Modules
│   ├── coco/               # COCO
│   ├── imagenet/           # Imagenet 1k and 100
│   ├── VOCdevkit/          # Pascal VOC
│   └── utils.py            # Data Handling Utils
└── dataset_rasa.md         # Dataset setup instructions
```

---

## 📄 License

This project is released under the Apache License 2.0.

---


