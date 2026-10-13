"""
Predictor module for Cocoa Disease Classification.
Handles model architecture instantiation, weight loading, CPU inference, and result post-processing.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
import torchvision.models as models

from .preprocessing import CocoaImagePreprocessor


class CocoaDiseasePredictor:
    """
    Cocoa Disease Detection Predictor using PyTorch EfficientNet-B0.
    """

    DEFAULT_CLASSES = [
        "anthracnose",
        "black_pod_rot",
        "cssvd",
        "frosty_pod_rot",
        "healthy",
        "mirid",
        "monilia",
        "pod_borer",
        "witches_broom",
    ]

    CLASS_DISPLAY_NAMES = {
        "anthracnose": "Anthracnose",
        "black_pod_rot": "Black Pod Rot",
        "cssvd": "Cocoa Swollen Shoot Virus (CSSVD)",
        "frosty_pod_rot": "Frosty Pod Rot",
        "healthy": "Healthy Cocoa",
        "mirid": "Mirid / Capsid Pest Damage",
        "monilia": "Moniliasis (Monilia Pod Rot)",
        "pod_borer": "Cocoa Pod Borer Damage",
        "witches_broom": "Witches' Broom Disease",
    }

    DISEASE_DESCRIPTIONS = {
        "anthracnose": (
            "Fungal infection caused by Colletotrichum species. Characterized by dark, sunken necrotic lesions "
            "on pods, leaves, and young shoots, often leading to leaf blight and pod dieback."
        ),
        "black_pod_rot": (
            "One of the most destructive cocoa diseases globally, caused by Phytophthora species (P. palmivora / P. megakarya). "
            "Presents as rapid dark brown to black rot spreading over pods, eventually turning the whole pod mummified."
        ),
        "cssvd": (
            "Cocoa Swollen Shoot Virus Disease (CSSVD), transmitted by mealybugs. Symptoms include red vein banding, "
            "chlorotic leaf mottling, and characteristic swelling of stems and chupons."
        ),
        "frosty_pod_rot": (
            "Severe fungal disease caused by Moniliophthora roreri. Early symptoms include irregular pod swelling "
            "and premature ripening, followed by a dense white-to-cream frosty layer of spores covering the pod surface."
        ),
        "healthy": (
            "No visible symptoms of fungal infection, viral lesions, or pest damage observed on the cocoa pod or foliage."
        ),
        "mirid": (
            "Damage caused by capsid bugs (mirids like Sahlbergella singularis or Distantiella theobroma). "
            "Feeding punctures cause necrotic dark lesions on pods and shoots, frequently leading to dieback and fungal entry."
        ),
        "monilia": (
            "Moniliasis / Monilia pod rot. Closely related to frosty pod rot, characterized by external lesions, "
            "pod distortion, and destruction of internal cocoa beans."
        ),
        "pod_borer": (
            "Infestation by the Cocoa Pod Borer (Conopomorpha cramerella) larva. Larvae bore into pod pulp and placenta, "
            "causing premature uneven yellowing, clumped unextractable beans, and exit holes."
        ),
        "witches_broom": (
            "Fungal disease caused by Moniliophthora perniciosa. Induces abnormal vegetative growth resulting in "
            "dense clusters of twigs resembling brooms, cushion brooms, and distorted, hard-surfaced pods."
        ),
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        uncertainty_threshold: float = 0.50,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.uncertainty_threshold = uncertainty_threshold

        self.model_path = self._resolve_file_path(
            model_path,
            candidates=[
                Path("models/best_model.pth"),
                Path("best_model.pth"),
                Path(__file__).parent.parent / "models" / "best_model.pth",
                Path(__file__).parent.parent / "best_model.pth",
            ],
            description="Trained model weights (.pth)",
        )

        self.metadata_path = self._resolve_file_path(
            metadata_path,
            candidates=[
                Path("artifacts/model_metadata.json"),
                Path("model_metadata.json"),
                Path("artifacts/class_mapping.json"),
                Path(__file__).parent.parent / "artifacts" / "model_metadata.json",
                Path(__file__).parent.parent / "model_metadata.json",
            ],
            description="Model metadata (.json)",
            required=False,
        )

        self.metadata = self._load_metadata()
        self.classes = self.metadata.get("classes", self.DEFAULT_CLASSES)
        self.num_classes = len(self.classes)

        # Initialize preprocessor
        self.preprocessor = CocoaImagePreprocessor(config=self.metadata.get("preprocessing_config"))

        # Build and load model
        self.model = self._build_model()
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _resolve_file_path(
        self,
        specified_path: Optional[Union[str, Path]],
        candidates: List[Path],
        description: str,
        required: bool = True,
    ) -> Optional[Path]:
        if specified_path:
            p = Path(specified_path)
            if p.is_file():
                return p.resolve()
            elif required:
                raise FileNotFoundError(f"Specified {description} not found at: {specified_path}")

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        if required:
            searched = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"Could not find {description}. Searched in: {searched}")
        return None

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path and self.metadata_path.is_file():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load metadata from {self.metadata_path}: {e}")
        return {
            "model_name": "EfficientNet-B0",
            "num_classes": len(self.DEFAULT_CLASSES),
            "classes": self.DEFAULT_CLASSES,
        }

    def _build_model(self) -> nn.Module:
        """
        Instantiate EfficientNet-B0 backbone with the custom 2-layer classifier head:
        Linear(1280 -> 256) -> ReLU() -> Dropout(0.2) -> Linear(256 -> num_classes)
        """
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )
        return model

    def _load_weights(self) -> None:
        """Load state dictionary on CPU or specified device."""
        if not self.model_path or not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Handle checkpoint dictionaries that store weights under 'state_dict' or 'model_state_dict'
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
            if missing:
                print(f"Warning: Missing keys in state_dict: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys in state_dict: {unexpected}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights from {self.model_path}: {e}") from e

    def predict(self, image_source: Any) -> Dict[str, Any]:
        """
        Run inference on an image.
        Returns predicted class, confidence, sorted probabilities, and agronomic info.
        """
        tensor = self.preprocessor.preprocess(image_source)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        top_prob, top_idx = torch.max(probs, dim=0)
        predicted_idx = int(top_idx.item())
        confidence = float(top_prob.item())
        predicted_class = self.classes[predicted_idx]
        display_name = self.CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class.replace("_", " ").title())

        # Build full probability breakdown
        prob_dict: Dict[str, float] = {}
        all_classes_list: List[Dict[str, Any]] = []

        for idx, cls_name in enumerate(self.classes):
            p = float(probs[idx].item())
            prob_dict[cls_name] = p
            all_classes_list.append({
                "class": cls_name,
                "display_name": self.CLASS_DISPLAY_NAMES.get(cls_name, cls_name.replace("_", " ").title()),
                "probability": p,
                "percentage": p * 100.0,
            })

        # Sort descending by probability
        all_classes_list.sort(key=lambda x: x["probability"], reverse=True)

        is_uncertain = confidence < self.uncertainty_threshold
        is_healthy = predicted_class == "healthy"

        return {
            "predicted_class": predicted_class,
            "display_name": display_name,
            "confidence": confidence,
            "confidence_percentage": confidence * 100.0,
            "is_healthy": is_healthy,
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": self.uncertainty_threshold,
            "description": self.DISEASE_DESCRIPTIONS.get(predicted_class, ""),
            "probabilities": prob_dict,
            "all_classes_sorted": all_classes_list,
            "top_3": all_classes_list[:3],
            "raw_logits": logits[0].cpu().tolist(),
        }

    def get_model_specs(self) -> Dict[str, Any]:
        """Return model metadata, architecture details, and evaluation metrics."""
        return {
            "model_name": self.metadata.get("model_name", "EfficientNet-B0"),
            "task_type": self.metadata.get("task_type", "Image Classification"),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "input_size": self.preprocessor.input_size,
            "normalization_mean": self.preprocessor.mean,
            "normalization_std": self.preprocessor.std,
            "evaluation_metrics": self.metadata.get("evaluation_metrics", {}),
            "model_path": str(self.model_path) if self.model_path else "Not found",
            "device": str(self.device),
        }
