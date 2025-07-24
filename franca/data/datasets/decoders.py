# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        try:
            # First try the standard way
            return Image.open(f).convert(mode="RGB")
        except Exception as e:
            try:
                # If that fails, try explicitly as JPEG
                f.seek(0)  # Reset file pointer
                return Image.open(f, formats=["JPEG"]).convert(mode="RGB")
            except Exception as nested_e:
                # Log the error for debugging
                print(f"Error decoding image: {e}, secondary error: {nested_e}")
                # Return a small placeholder image instead of raising an exception
                return Image.new("RGB", (16, 16), color=(128, 128, 128))


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target
