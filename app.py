#!/usr/bin/env python3
"""Local Gradio prototype for the selected ConvNeXt-Tiny vehicle classifier.

Usage:
  python app.py --data-dir /Users/yourname/Desktop/Dataset

The app accepts one cropped vehicle image and returns prediction, confidence,
and a Grad-CAM overlay. Run it locally, verify it with a fresh image, and use a
real screenshot as prototype evidence in the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import convnext_tiny

IMG_SIZE = 224
CLASS_NAMES = ["Bus", "Car", "Motorcycle", "Truck"]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        predicted_index = int(logits.argmax(dim=1).item())
        logits[:, predicted_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np, logits.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local vehicle-classification prototype")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing results/convnexttiny_best.pth")
    parser.add_argument("--share", action="store_true", help="Request a temporary public Gradio URL")
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = data_dir / "results" / "convnexttiny_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint}")

    device = choose_device()
    model = load_model(checkpoint, device)
    gradcam = GradCAM(model, model.features[-1])
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def predict(image: Image.Image) -> Tuple[Dict[str, float], str, Image.Image]:
        if image is None:
            raise gr.Error("Please upload a vehicle image.")
        original = ImageOps.exif_transpose(image).convert("RGB")
        display = original.resize((IMG_SIZE, IMG_SIZE))
        input_tensor = preprocess(display).unsqueeze(0).to(device)

        cam, logits = gradcam.generate(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        prediction_index = int(np.argmax(probabilities))
        prediction = CLASS_NAMES[prediction_index]
        confidence = float(probabilities[prediction_index])

        original_np = np.asarray(display).astype(np.float32) / 255.0
        heatmap = cm.get_cmap("jet")(cam)[..., :3]
        overlay = np.clip(0.55 * original_np + 0.45 * heatmap, 0, 1)
        overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))

        label_scores = {name: float(score) for name, score in zip(CLASS_NAMES, probabilities)}
        summary = f"Predicted class: {prediction}\nConfidence: {confidence:.2%}\nDevice: {device.type}"
        return label_scores, summary, overlay_image

    with gr.Blocks(title="Vehicle Type Classifier") as demo:
        gr.Markdown("# Vehicle Type Classifier\nUpload one cropped vehicle image. This prototype performs image classification; it does not detect multiple vehicles in a scene.")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input vehicle image")
            heatmap_image = gr.Image(type="pil", label="Grad-CAM overlay")
        with gr.Row():
            labels = gr.Label(num_top_classes=4, label="Class probabilities")
            result_text = gr.Textbox(label="Prediction summary", lines=3)
        predict_button = gr.Button("Classify vehicle")
        predict_button.click(predict, inputs=input_image, outputs=[labels, result_text, heatmap_image])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
