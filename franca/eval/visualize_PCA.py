import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm

franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14')

# Get the patch size from the model
patch_size = model.patch_size
feat_dim = 1024  # for vitl14

# Calculate dimensions that are multiples of patch_size
image_size = 224 * 2  # 448, which is cleanly divisible by patch_size (14)
patch_h, patch_w = image_size // patch_size, image_size // patch_size

# Define transforms
# For PCA visualization
transform_pca = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

# For saving regular cropped frames
transform_frames = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def process_image(img_path, output_pca_dir, output_frames_dir, sequence_name):
    """Process a single image: extract features, compute PCA, and save visualizations"""
    try:
        # Load and prepare image
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)

        # Save center-cropped frame
        img_t_frame = transform_frames(img)
        frame_img = transforms.ToPILImage()(img_t_frame)
        frame_output_path = os.path.join(output_frames_dir, sequence_name, img_name)
        os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
        frame_img.save(frame_output_path)

        # Process for PCA visualization
        img_t = transform_pca(img).unsqueeze(0).cuda()

        with torch.no_grad():
            features_dict = model.forward_features(img_t)
            features = features_dict["x_norm_patchtokens"]
            total_features = features.squeeze(0).cpu().numpy()  # (h*w, feat_dim)

        # PCA step 1
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(total_features)

        # Min-max scaling
        pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (
            pca_features[:, 0].max() - pca_features[:, 0].min()
        )

        # Threshold background
        pca_features_bg = pca_features[:, 0] > 0.35
        pca_features_fg = ~pca_features_bg

        # PCA step 2 (foreground)
        if np.sum(pca_features_fg) > 0:  # Check if there are foreground features
            pca.fit(total_features[pca_features_fg])
            pca_features_left = pca.transform(total_features[pca_features_fg])

            # Normalize new features
            for i in range(3):
                min_val = pca_features_left[:, i].min()
                max_val = pca_features_left[:, i].max()
                if max_val > min_val:  # Prevent division by zero
                    pca_features_left[:, i] = (pca_features_left[:, i] - min_val) / (max_val - min_val)

            # Merge back
            pca_features_rgb = pca_features.copy()
            pca_features_rgb[pca_features_bg] = 0
            pca_features_rgb[pca_features_fg] = pca_features_left
        else:
            # If no foreground is detected, use the original PCA features
            pca_features_rgb = pca_features

        pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)

        # 1. Interpolate to original image size
        pca_features_rgb_tensor = torch.from_numpy(pca_features_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, patch_h, patch_w)
        pca_features_rgb_upsampled = F.interpolate(
            pca_features_rgb_tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        pca_features_rgb_upsampled = (
            pca_features_rgb_upsampled.squeeze(0).permute(1, 2, 0).numpy()
        )  # (image_size, image_size, 3)

        # 2. Background mask upsample
        pca_features_bg_mask = pca_features_bg.reshape(patch_h, patch_w).astype(np.float32)
        pca_features_bg_mask_tensor = (
            torch.from_numpy(pca_features_bg_mask).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, patch_h, patch_w)
        pca_features_bg_mask_upsampled = F.interpolate(
            pca_features_bg_mask_tensor, size=(image_size, image_size), mode="nearest"
        )
        pca_features_bg_mask_upsampled = pca_features_bg_mask_upsampled.squeeze(0).squeeze(0).numpy() > 0.5  # binary mask

        # 3. Apply mask
        pca_features_rgb_upsampled[pca_features_bg_mask_upsampled] = 0  # black background

        # 4. Save
        final_image = (pca_features_rgb_upsampled * 255).astype(np.uint8)
        final_image_pil = Image.fromarray(final_image)

        pca_output_path = os.path.join(output_pca_dir, sequence_name, img_name)
        os.makedirs(os.path.dirname(pca_output_path), exist_ok=True)
        final_image_pil.save(pca_output_path)

        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def process_davis_dataset(davis_root_dir, output_base_dir):
    """Process all image sequences in the DAVIS dataset"""
    # Create output directories
    output_pca_dir = os.path.join(output_base_dir, "pca_visualizations")
    output_frames_dir = os.path.join(output_base_dir, "cropped_frames")
    os.makedirs(output_pca_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    # Find all sequence folders in the DAVIS dataset
    sequence_folders = [f for f in os.listdir(davis_root_dir) if os.path.isdir(os.path.join(davis_root_dir, f))]

    total_processed = 0
    total_failed = 0

    for sequence in tqdm(sequence_folders, desc="Processing sequences"):
        sequence_dir = os.path.join(davis_root_dir, sequence)
        image_files = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))

        if not image_files:
            print(f"No JPG images found in {sequence_dir}")
            continue

        print(f"Processing sequence: {sequence} ({len(image_files)} frames)")

        sequence_processed = 0
        sequence_failed = 0

        for img_path in tqdm(image_files, desc=f"Processing {sequence}", leave=False):
            success = process_image(img_path, output_pca_dir, output_frames_dir, sequence)
            if success:
                sequence_processed += 1
            else:
                sequence_failed += 1

        print(f"Completed {sequence}: {sequence_processed} processed, {sequence_failed} failed")
        total_processed += sequence_processed
        total_failed += sequence_failed

    print(f"Processing complete. Total: {total_processed} processed, {total_failed} failed")


if __name__ == "__main__":
    # Set paths
    # Update this with your DAVIS dataset path
    davis_root_dir = "/lustre/fsn1/projects/rech/yic/uco38ei/davis_2021/davis_data/JPEGImages/480p/"
    # Update this with your desired output path
    output_base_dir = "/lustre/fsn1/projects/rech/fhx/commun/pca_viz_daviz/franca_l"

    # Process the dataset
    process_davis_dataset(davis_root_dir, output_base_dir)
