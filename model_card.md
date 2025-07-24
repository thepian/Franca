# Model Card for Franca

These are Vision Transformer models trained following the method described in the papers:
"Franca: Nested Clustering with Matryoshka for Scalable Visual Representation Learning"

We provide 7 models:
- DINOv2-B and DINOv2-L reproduced on IN-21K. (without distillation)
- Franca-B, Franca-L and Franca-G pretrained on IN-21K.
- Franca-L and Franca-G pretrained on LAION-600M.

## Model Details
The model takes an image as input and returns a class token and patch tokens

The embedding dimension is:
- 768 for ViT-B.
- 1024 for ViT-L.
- 1536 for ViT-g.

The models follow a Transformer architecture, with a patch size of 14.

For a 224x224 image, this results in 1 class token + 256 patch tokens

The models can accept larger images provided the image shapes are multiples of the patch size (14).
If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

### Model Description

- **Developed by:** Valeo.ai
- **Model type:** Vision Transformer
- **License:** Research-Only RAIL License. See [model license](LICENSE_MODEL)

- **Repository:** https://github.com/valeoai/Franca
- **Paper:** https://arxiv.org/abs/2507.14137

## Uses

The models are vision backbones providing multi-purpose features for downstream tasks.

### Direct Use

The models can be used without fine-tuning, with downstream classifiers as simple as linear layers, to obtain competitive results:

- on image classification, using k-NN classifiers on the class token.
- on image classification, with logistic regression classifiers applied on the class token.
- on image classification, with a linear layer applied on the class token and the average of the patch tokens.
- on depth estimation, linear segmentation, overclustering, In-Context Learning

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch

# Franca -- In21k
franca_vitb14 = torch.hub.load('valeoai/Franca', 'franca_vitb14')
franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14')
franca_vitg14 = torch.hub.load('valeoai/Franca', 'franca_vitg14')

# Franca -- Laion600M
franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='LAION600m')
franca_vitg14 = torch.hub.load('valeoai/Franca', 'franca_vitg14', weights='LAION600m')

# Dinov2 baseline -- In21k
franca_vitb14 = torch.hub.load('valeoai/Franca', 'franca_vitb14', weights='Dinov2_In21k')
franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='Dinov2_In21k')
```


## Loading intermediate checkpoints

We aim to provide intermediate checkpoints to the community to study training dynamics, analyze convergence behavior, conduct representation analysis, and study emergent properties across time.

To load a Franca model directly using the checkpoint (intermediate or final) from [link](coming soon), use the example below:

```python
import torch
from PIL import Image
from torchvision import transforms
from franca.hub.backbones import _make_franca_model

# --- Step 1: Choose model config ---
arch_name = "vit_large"
img_size = 224
ckpt_path = "<path to checkpoint>"

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# --- Step 2: Build and load model ---
model = _make_franca_model(
    arch_name=arch_name,
    img_size=img_size,
    pretrained=True,
    local_state_dict=ckpt_path,
)

# --- Step 3: Forward pass ---
model.cuda()
model.eval()

image = Image.open("assets/dog.jpg")
x = transform(image).unsqueeze(0).cuda()

with torch.no_grad():
    feats = model.forward_features(x)
    cls_token = feats["x_norm_clstoken"]
    patch_tokens = feats["x_norm_patchtokens"]

print("CLS token shape:", cls_token.shape)
print("Patch token shape:", patch_tokens.shape)
```
