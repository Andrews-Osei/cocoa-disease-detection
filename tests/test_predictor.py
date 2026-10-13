"""
Tests for Cocoa Disease Detection Preprocessor and Predictor.
"""

import io
import pytest
from PIL import Image
import torch

from src.preprocessing import CocoaImagePreprocessor
from src.predictor import CocoaDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    """Instantiate CocoaDiseasePredictor once for test suite."""
    return CocoaDiseasePredictor(device="cpu")


def test_predictor_initialization(predictor):
    """Verify that predictor loads model weights and metadata correctly on CPU."""
    assert predictor.model is not None
    assert predictor.num_classes == 9
    assert len(predictor.classes) == 9
    assert "healthy" in predictor.classes
    assert "black_pod_rot" in predictor.classes
    assert str(predictor.device) == "cpu"


def test_preprocessing_output_shape():
    """Verify preprocessing produces correctly sized (1, 3, 224, 224) tensors."""
    preprocessor = CocoaImagePreprocessor()
    dummy_img = Image.new("RGB", (300, 400), color=(120, 80, 50))
    tensor = preprocessor.preprocess(dummy_img)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocessing_mode_conversion():
    """Ensure RGBA, Grayscale, and CMYK images are properly converted to RGB."""
    preprocessor = CocoaImagePreprocessor()
    
    rgba_img = Image.new("RGBA", (250, 250), color=(100, 150, 200, 128))
    tensor_rgba = preprocessor.preprocess(rgba_img)
    assert tensor_rgba.shape == (1, 3, 224, 224)

    gray_img = Image.new("L", (250, 250), color=128)
    tensor_gray = preprocessor.preprocess(gray_img)
    assert tensor_gray.shape == (1, 3, 224, 224)


def test_corrupted_image_handling():
    """Ensure invalid or corrupted byte streams raise ValueError."""
    preprocessor = CocoaImagePreprocessor()
    fake_bytes = io.BytesIO(b"This is not a real image header")
    with pytest.raises(ValueError):
        preprocessor.load_image(fake_bytes)


def test_too_small_image_handling():
    """Ensure extremely small thumbnail images (below 10x10) are rejected."""
    preprocessor = CocoaImagePreprocessor()
    tiny_img = Image.new("RGB", (5, 5), color=(255, 0, 0))
    with pytest.raises(ValueError):
        preprocessor.load_image(tiny_img)


def test_prediction_output_structure(predictor):
    """Verify that predictor returns complete dictionary with probabilities and top predictions."""
    dummy_img = Image.new("RGB", (224, 224), color=(60, 140, 70))
    result = predictor.predict(dummy_img)

    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "all_classes_sorted" in result
    assert "top_3" in result
    assert "is_healthy" in result
    assert "is_uncertain" in result

    assert result["predicted_class"] in predictor.classes
    assert 0.0 <= result["confidence"] <= 1.0

    # Ensure all 9 classes have non-negative probabilities summing to ~1.0
    probs = result["probabilities"]
    assert len(probs) == 9
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0


def test_prediction_on_sample_image(predictor):
    """Run prediction on an actual sample image in assets/samples."""
    sample_path = "assets/samples/healthy_pod_sample.jpg"
    result = predictor.predict(sample_path)
    assert result["predicted_class"] in predictor.classes
    assert result["confidence"] > 0.0
    assert len(result["top_3"]) == 3
