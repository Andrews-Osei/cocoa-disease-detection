"""
Preprocessing pipeline for cocoa disease classification.
"""

import io
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
from PIL import Image, ImageOps
import torch
from torchvision import transforms


class CocoaImagePreprocessor:
    """
    Standard image preprocessing for EfficientNet-B0 cocoa disease detection.
    
    Converts raw image inputs into normalized PyTorch tensors matching the training configuration:
    - Input dimensions: 224x224 RGB
    - Normalization mean: [0.485, 0.456, 0.406]
    - Normalization std: [0.229, 0.224, 0.225]
    """

    DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)
    DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, metadata_path: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self.config = config
        elif metadata_path is not None and Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.config = meta.get("preprocessing_config", {})
        else:
            self.config = {}

        raw_size = self.config.get("input_size", self.DEFAULT_INPUT_SIZE)
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            self.input_size: Tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))
        else:
            self.input_size = self.DEFAULT_INPUT_SIZE

        self.mean = self.config.get("normalization_mean", list(self.DEFAULT_MEAN))
        self.std = self.config.get("normalization_std", list(self.DEFAULT_STD))

        # Standard deterministic inference pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image(self, image_source: Union[str, Path, bytes, bytearray, Image.Image, Any]) -> Image.Image:
        """
        Safely open and validate an image from various input sources.
        Converts any valid image format (RGBA, Grayscale, CMYK) to 3-channel RGB.
        """
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            try:
                img = Image.open(path)
                img.load()
            except Exception as exc:
                raise ValueError(f"Could not open image file: {exc}") from exc
        elif isinstance(image_source, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(image_source))
                img.load()
            except Exception as exc:
                raise ValueError(f"Invalid image bytes: {exc}") from exc
        elif hasattr(image_source, "read"):
            try:
                if hasattr(image_source, "seek"):
                    image_source.seek(0)
                img = Image.open(image_source)
                img.load()
            except Exception as exc:
                raise ValueError(f"Invalid or corrupted image file: {exc}") from exc
        else:
            raise TypeError(f"Unsupported image input type: {type(image_source)}")

        # Handle EXIF orientation if present
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Convert to 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Validate minimum pixel dimensions
        if img.width < 10 or img.height < 10:
            raise ValueError(f"Image dimensions ({img.width}x{img.height}) are too small for analysis.")

        return img

    def preprocess(self, image_source: Union[str, Path, bytes, Image.Image, Any]) -> torch.Tensor:
        """
        Validates, converts, and normalizes the image into a batched PyTorch tensor.
        Output shape: (1, 3, 224, 224).
        """
        image = self.load_image(image_source)
        tensor = self.transform(image)
        # Add batch dimension: (3, H, W) -> (1, 3, H, W)
        return tensor.unsqueeze(0)
