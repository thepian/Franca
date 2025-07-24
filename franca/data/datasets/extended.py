# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset

from franca.data.datasets.decoders import ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        # Add a counter for error tracking
        self._error_count = 0
        self._max_errors = 100  # Limit errors to avoid flooding logs

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            # Use the robust decoder
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            if self._error_count < self._max_errors:
                print(f"Warning: cannot read image for sample {index}: {str(e)}")
                self._error_count += 1
                if self._error_count == self._max_errors:
                    print("Maximum error count reached. Suppressing further error messages.")

            # Return a gray placeholder image instead of raising an exception
            image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        try:
            target = self.get_target(index)
            target = TargetDecoder(target).decode()
        except Exception as e:
            print(f"Warning: cannot read target for sample {index}: {str(e)}")
            # Return a default target (0)
            target = 0

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
