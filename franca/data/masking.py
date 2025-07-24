# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import random

import numpy as np


class BlockMasking:
    """Block masking strategy for patch-based masking."""

    def __init__(
        self,
        input_size: tuple[int, int],
        roll: bool = True,
        min_aspect: float = 0.5,
        max_aspect: float = None,
    ):
        self.height, self.width = input_size
        self.roll = roll
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)

        # Ensure num_masking_patches doesn't exceed total patches
        num_masking_patches = min(num_masking_patches, self.height * self.width)

        # Sample aspect ratio, not too large or too small for image
        min_lar = max(self.log_aspect_ratio[0], np.log(num_masking_patches / (self.width**2)))
        max_lar = min(
            self.log_aspect_ratio[1],
            np.log(self.height**2 / (num_masking_patches + 1e-5)),
        )

        # Ensure min_lar <= max_lar
        if min_lar > max_lar:
            min_lar = max_lar

        aspect_ratio = math.exp(random.uniform(min_lar, max_lar))

        # Use ceil so mask is >= num_masking_patches
        h = int(np.ceil(math.sqrt(num_masking_patches * aspect_ratio)))
        w = int(np.ceil(math.sqrt(num_masking_patches / aspect_ratio)))

        # Ensure dimensions don't exceed image size
        h = min(h, self.height)
        w = min(w, self.width)

        # Handle edge case where h or w might be larger than height/width
        if h >= self.height:
            h = self.height
            top = 0
        else:
            top = random.randint(0, self.height - h)

        if w >= self.width:
            w = self.width
            left = 0
        else:
            left = random.randint(0, self.width - w)

        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[top : top + h, left : left + w] = True

        # Truncate ids to get exactly num_masking_patches
        ids = np.where(mask.flatten())[0]
        if len(ids) > num_masking_patches:
            ids = ids[:num_masking_patches]

        mask = np.zeros((self.height, self.width), dtype=bool).flatten()
        mask[ids] = True
        mask = mask.reshape((self.height, self.width))

        if self.roll:
            shift_x = random.randint(0, mask.shape[0] - 1)
            shift_y = random.randint(0, mask.shape[1] - 1)
            mask = np.roll(mask, (shift_x, shift_y), (0, 1))

        return mask


class InverseBlockMasking(BlockMasking):
    """Inverse block masking strategy - masks everything except a block."""

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        # Handle edge cases to prevent errors
        total_patches = self.height * self.width

        # If num_masking_patches is 0, return all-False mask
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)

        # If num_masking_patches equals total patches, return all-True mask
        if num_masking_patches >= total_patches:
            return np.ones((self.height, self.width), dtype=bool)

        # Calculate complement patches, ensure it's at least 1
        complement_patches = total_patches - num_masking_patches
        complement_patches = max(1, complement_patches)

        # Get the complement mask and invert it
        mask = super().__call__(complement_patches)
        return ~mask


class MaskingGenerator:
    """Generator for creating masks using various masking strategies."""

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
        use_block_masking=True,
        use_inverse_block=True,
        roll=True,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        # Add new parameters for using the block masking strategy
        self.use_block_masking = use_block_masking
        self.use_inverse_block = use_inverse_block

        # Initialize the block masking class if needed
        if self.use_block_masking:
            if self.use_inverse_block:
                self.block_masker = InverseBlockMasking(
                    input_size=(self.height, self.width),
                    roll=roll,
                    min_aspect=min_aspect,
                    max_aspect=max_aspect,
                )
            else:
                self.block_masker = BlockMasking(
                    input_size=(self.height, self.width),
                    roll=roll,
                    min_aspect=min_aspect,
                    max_aspect=max_aspect,
                )

    def __repr__(self):
        repr_str = (
            f"Generator({self.height}, {self.width} -> "
            f"[{self.min_num_patches} ~ {self.max_num_patches}], "
            f"max = {self.num_masking_patches}, "
            f"{self.log_aspect_ratio[0]:.3f} ~ {self.log_aspect_ratio[1]:.3f})"
        )
        if self.use_block_masking:
            block_type = "inverse " if self.use_inverse_block else ""
            repr_str += f" using {block_type}block masking"
        return repr_str

    def get_shape(self):
        """Get the shape of the mask."""
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        """Apply masking to a specific region."""
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        """Generate a mask with the specified number of patches."""
        # Use the new block masking strategy if enabled
        if self.use_block_masking:
            return self.block_masker(num_masking_patches)

        # Otherwise fall back to the original implementation
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
