from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import shap

from torchvision import datasets
from torchvision.models import convnext_tiny


# -------------------------------------------------
# Paths
# -------------------------------------------------

DATA_DIR = Path("/Users/ardaozdemir/Desktop/Dataset")
TEST_DIR = DATA_DIR / "test"
RESULTS_DIR = DATA_DIR / "results"
MODEL_PATH = RESULTS_DIR / "convnexttiny_best.pth"

OUTPUT_DIR = RESULTS_DIR / "shap_convnexttiny"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Settings
# -------------------------------------------------

IMG_SIZE = 224
IMAGES_PER_CLASS = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -------------------------------------------------
# Device
# -------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -------------------------------------------------
# Load dataset information
# -------------------------------------------------

test_dataset = datasets.ImageFolder(TEST_DIR)
class_names = test_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Test images:", len(test_dataset))


# -------------------------------------------------
# Load ConvNeXtTiny model
# -------------------------------------------------

model = convnext_tiny(weights=None)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, num_classes)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("Loaded model:", MODEL_PATH)


# -------------------------------------------------
# Image loading
# -------------------------------------------------

def load_image_as_numpy(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # SHAP expects image as H x W x C
    arr = np.array(img).astype(np.float32)

    return arr


# -------------------------------------------------
# Prediction function for SHAP
# -------------------------------------------------

def predict(images_np):
    """
    SHAP sends images as numpy arrays:
    batch_size x height x width x channels
    """

    images_np = images_np.astype(np.float32)

    # Convert 0-255 to 0-1
    if images_np.max() > 1.0:
        images_np = images_np / 255.0

    images_np = np.clip(images_np, 0.0, 1.0)

    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images_tensor = (images_tensor - mean) / std
    images_tensor = images_tensor.to(device)

    with torch.no_grad():
        outputs = model(images_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()


# -------------------------------------------------
# Select 2 images per class
# -------------------------------------------------

class_to_paths = {}

for image_path, label in test_dataset.samples:
    class_name = class_names[label]
    class_to_paths.setdefault(class_name, []).append((image_path, label))

selected_samples = []

for class_name, samples in class_to_paths.items():
    chosen = random.sample(samples, min(IMAGES_PER_CLASS, len(samples)))
    selected_samples.extend(chosen)

print(f"Selected {len(selected_samples)} images for SHAP.")


# -------------------------------------------------
# Create SHAP explainer
# -------------------------------------------------

masker = shap.maskers.Image("blur(16,16)", (IMG_SIZE, IMG_SIZE, 3))

explainer = shap.Explainer(
    predict,
    masker,
    output_names=class_names
)


# -------------------------------------------------
# Generate SHAP PDF explanations
# -------------------------------------------------

summary_lines = []

for i, (image_path, true_label) in enumerate(selected_samples, start=1):
    print(f"\nProcessing image {i}/{len(selected_samples)}")
    print("Image:", image_path)

    image_np = load_image_as_numpy(image_path)

    preds = predict(np.expand_dims(image_np, axis=0))
    pred_label = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_label])

    true_class = class_names[true_label]
    pred_class = class_names[pred_label]

    print(f"True: {true_class}")
    print(f"Predicted: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    # SHAP can be slow.
    # 500 is acceptable for report figures.
    # Increase to 1000 if you want cleaner explanations, but it will take longer.
    shap_values = explainer(
        np.expand_dims(image_np, axis=0),
        max_evals=500,
        batch_size=8,
        outputs=[pred_label]
    )

    plt.figure()
    shap.plots.image(shap_values, show=False)

    status = "correct" if true_label == pred_label else "wrong"
    image_name = Path(image_path).stem

    save_path = OUTPUT_DIR / (
        f"{i:02d}_{status}_true-{true_class}_"
        f"pred-{pred_class}_conf-{confidence:.2f}_{image_name}.pdf"
    )

    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        dpi=600
    )

    plt.close()

    print("Saved:", save_path)

    summary_lines.append(
        f"{i:02d}: true={true_class}, predicted={pred_class}, "
        f"confidence={confidence:.4f}, file={save_path.name}"
    )


# -------------------------------------------------
# Save summary
# -------------------------------------------------

summary_path = OUTPUT_DIR / "shap_summary.txt"

with open(summary_path, "w") as f:
    f.write("SHAP Explanation Summary - ConvNeXtTiny\n")
    f.write("======================================\n\n")
    f.write("\n".join(summary_lines))

print("\nSHAP completed.")
print("Output folder:", OUTPUT_DIR)
print("Summary saved:", summary_path)