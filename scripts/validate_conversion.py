import torch
import coremltools as ct
import numpy as np
from franca.models import build_franca_backbone

CHECKPOINT_PATH = "checkpoints/franca_vit_small.pth"
COREML_PATH = "coreml_models/franca_fp32.mlmodel"

# Load Franca
model = build_franca_backbone(arch="vit_small")
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

# Random test input
input_torch = torch.randn(1, 3, 518, 518)
with torch.no_grad():
    output_torch = model(input_torch).numpy()

# Load Core ML
mlmodel = ct.models.MLModel(COREML_PATH)
input_np = (input_torch.numpy() * 255).astype(np.float32)  # reverse scale
pred_coreml = mlmodel.predict({"image": input_np})["output"]

# Cosine similarity
cos_sim = np.dot(output_torch.flatten(), pred_coreml.flatten()) / (
    np.linalg.norm(output_torch) * np.linalg.norm(pred_coreml)
)
print(f"Cosine similarity: {cos_sim:.6f}")
