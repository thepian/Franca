import os
import torch
import coremltools as ct
from franca.models import build_franca_backbone  # adjust if needed

CHECKPOINT_PATH = "checkpoints/franca_vit_small.pth"
OUTPUT_DIR = "coreml_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Franca wrapper
class FrancaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

# Load model
model = build_franca_backbone(arch="vit_small")  # adjust params to match checkpoint
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

wrapped = FrancaWrapper(model)
example_input = torch.randn(1, 3, 518, 518)

with torch.no_grad():
    traced = torch.jit.trace(wrapped, example_input)
    traced = torch.jit.freeze(traced)

image_input = ct.ImageType(
    name="image",
    shape=example_input.shape,
    scale=1/255.0,
    bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
)

mlmodel = ct.convert(
    traced,
    inputs=[image_input],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)

coreml_path = os.path.join(OUTPUT_DIR, "franca_fp32.mlmodel")
mlmodel.save(coreml_path)

# Optional: fp16
from coremltools.models.neural_network.quantization_utils import quantize_weights
mlmodel_fp16 = quantize_weights(mlmodel, nbits=16)
mlmodel_fp16.save(os.path.join(OUTPUT_DIR, "franca_fp16.mlmodel"))

print(f"Models saved to {OUTPUT_DIR}")
