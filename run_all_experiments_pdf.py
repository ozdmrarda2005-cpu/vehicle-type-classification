from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter, ImageDraw
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# SETTINGS
# ==========================================================

DATA_DIR = Path("/Users/ardaozdemir/Desktop/Dataset")
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE_CUSTOM = 16
BATCH_SIZE_TRANSFER = 8
EPOCHS_CUSTOM = 30
EPOCHS_TRANSFER = 25
PATIENCE = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ==========================================================
# TRANSFORMS
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# DATASETS
# ==========================================================

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_test_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=val_test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ==========================================================
# MODELS
# ==========================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_custom_cnn():
    return CustomCNN(num_classes)


def build_efficientnetv2():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_convnexttiny():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

# ==========================================================
# UTILS
# ==========================================================

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, macro_f1, weighted_f1, balanced_acc, all_labels, all_preds


def measure_inference_time(model, loader):
    model.eval()
    times = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            start = time.time()
            _ = model(images)
            end = time.time()

            times.append((end - start) / images.size(0))

    return sum(times) / len(times)


def save_confusion_matrix_pdf(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_loss_curve_pdf(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_report_txt(
    model_name,
    save_path,
    test_acc,
    test_macro_f1,
    test_weighted_f1,
    test_bal_acc,
    inference_time,
    total_params,
    trainable_params,
    report
):
    with open(save_path, "w") as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted-F1: {test_weighted_f1:.4f}\n")
        f.write(f"Test Balanced Accuracy: {test_bal_acc:.4f}\n")
        f.write(f"Average inference time per image: {inference_time:.6f} seconds\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
        f.write(report)


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_model(model_name, model_builder, batch_size, epochs, learning_rate):
    print("\n" + "=" * 70)
    print(f"Training {model_name}")
    print("=" * 70)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_builder().to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model_path = RESULTS_DIR / f"{model_name.lower()}_best.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_macro_f1, val_weighted_f1, val_bal_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val Macro-F1: {val_macro_f1:.4f}")
        print(f"Val Weighted-F1: {val_weighted_f1:.4f}")
        print(f"Val Balanced Accuracy: {val_bal_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))

    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_bal_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion
    )

    inference_time = measure_inference_time(model, test_loader)

    report = classification_report(y_true, y_pred, target_names=class_names)

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
    print(f"Test Weighted-F1: {test_weighted_f1:.4f}")
    print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Average inference time per image: {inference_time:.6f} seconds")
    print(report)

    save_report_txt(
        model_name=model_name,
        save_path=RESULTS_DIR / f"{model_name.lower()}_classification_report.txt",
        test_acc=test_acc,
        test_macro_f1=test_macro_f1,
        test_weighted_f1=test_weighted_f1,
        test_bal_acc=test_bal_acc,
        inference_time=inference_time,
        total_params=total_params,
        trainable_params=trainable_params,
        report=report
    )

    save_confusion_matrix_pdf(
        y_true,
        y_pred,
        f"{model_name} Confusion Matrix",
        RESULTS_DIR / f"{model_name.lower()}_confusion_matrix.pdf"
    )

    save_loss_curve_pdf(
        train_losses,
        val_losses,
        f"{model_name} Training and Validation Loss",
        RESULTS_DIR / f"{model_name.lower()}_loss_curve.pdf"
    )

    return {
        "model_name": model_name,
        "model": model,
        "model_path": model_path,
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        "balanced_accuracy": test_bal_acc,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "y_true": y_true,
        "y_pred": y_pred
    }


# ==========================================================
# GRAD-CAM FOR CONVNEXTTINY
# ==========================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, output

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def denormalize_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()


def create_overlay(image_np, cam):
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.55 * image_np + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def run_gradcam_convnexttiny(convnext_model):
    print("\n" + "=" * 70)
    print("Generating Grad-CAM PDFs for ConvNeXtTiny")
    print("=" * 70)

    output_dir = RESULTS_DIR / "gradcam_convnexttiny"
    output_dir.mkdir(exist_ok=True)

    test_for_gradcam = datasets.ImageFolder(DATA_DIR / "test", transform=val_test_transform)

    class_to_indices = {}
    for idx, (_, label) in enumerate(test_for_gradcam.samples):
        class_name = class_names[label]
        class_to_indices.setdefault(class_name, []).append(idx)

    selected_indices = []
    for class_name, indices in class_to_indices.items():
        selected_indices.extend(random.sample(indices, min(2, len(indices))))

    target_layer = convnext_model.features[-1]
    gradcam = GradCAM(convnext_model, target_layer)

    for count, idx in enumerate(selected_indices, start=1):
        image_tensor, true_label = test_for_gradcam[idx]
        image_path, _ = test_for_gradcam.samples[idx]

        input_tensor = image_tensor.unsqueeze(0).to(device)

        cam, output = gradcam.generate(input_tensor)

        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_label = torch.max(probabilities, dim=1)

        predicted_label = predicted_label.item()
        confidence = confidence.item()

        true_class = class_names[true_label]
        predicted_class = class_names[predicted_label]
        status = "correct" if true_label == predicted_label else "wrong"

        original_np = denormalize_tensor(image_tensor)
        overlay = create_overlay(original_np, cam)

        image_name = Path(image_path).stem

        save_path = output_dir / (
            f"{count:02d}_{status}_true-{true_class}_"
            f"pred-{predicted_class}_conf-{confidence:.2f}_{image_name}.pdf"
        )

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        plt.suptitle(f"True: {true_class} | Predicted: {predicted_class} | Confidence: {confidence:.2f}")
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        plt.close()

        print("Saved:", save_path)

    gradcam.remove_hooks()


# ==========================================================
# BACKGROUND SENSITIVITY FOR CONVNEXTTINY
# ==========================================================

def load_original_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img


def blur_background_keep_center(path):
    img = load_original_image(path)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=10))

    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)

    keep_width = int(IMG_SIZE * 0.72)
    keep_height = int(IMG_SIZE * 0.66)

    left = (IMG_SIZE - keep_width) // 2
    top = (IMG_SIZE - keep_height) // 2
    right = left + keep_width
    bottom = top + keep_height

    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=25,
        fill=255
    )

    result = Image.composite(img, blurred, mask)
    return result


def run_background_sensitivity(convnext_model):
    print("\n" + "=" * 70)
    print("Running background sensitivity experiment")
    print("=" * 70)

    output_dir = RESULTS_DIR / "background_sensitivity_convnexttiny"
    output_dir.mkdir(exist_ok=True)

    raw_test_dataset = datasets.ImageFolder(DATA_DIR / "test")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def evaluate_variant(image_function, variant_name):
        y_true = []
        y_pred = []
        confidences = []

        convnext_model.eval()

        with torch.no_grad():
            for image_path, label in raw_test_dataset.samples:
                img = image_function(image_path)
                tensor = preprocess(img).unsqueeze(0).to(device)

                outputs = convnext_model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, dim=1)

                y_true.append(label)
                y_pred.append(pred.item())
                confidences.append(confidence.item())

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        report = classification_report(y_true, y_pred, target_names=class_names)

        print(f"\n{variant_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(report)

        return {
            "name": variant_name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_acc": balanced_acc,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidences": confidences,
            "report": report
        }

    original_results = evaluate_variant(load_original_image, "Original Test Images")
    blurred_results = evaluate_variant(blur_background_keep_center, "Background-Blurred Test Images")

    report_path = output_dir / "background_sensitivity_report.txt"

    with open(report_path, "w") as f:
        f.write("Background Sensitivity Experiment - ConvNeXtTiny\n")
        f.write("================================================\n\n")

        f.write("Original Test Images\n")
        f.write("--------------------\n")
        f.write(f"Accuracy: {original_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {original_results['macro_f1']:.4f}\n")
        f.write(f"Weighted-F1: {original_results['weighted_f1']:.4f}\n")
        f.write(f"Balanced Accuracy: {original_results['balanced_acc']:.4f}\n\n")
        f.write(original_results["report"])

        f.write("\n\nBackground-Blurred Test Images\n")
        f.write("------------------------------\n")
        f.write(f"Accuracy: {blurred_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1: {blurred_results['macro_f1']:.4f}\n")
        f.write(f"Weighted-F1: {blurred_results['weighted_f1']:.4f}\n")
        f.write(f"Balanced Accuracy: {blurred_results['balanced_acc']:.4f}\n\n")
        f.write(blurred_results["report"])

        f.write("\n\nPerformance Drop\n")
        f.write("----------------\n")
        f.write(f"Accuracy drop: {original_results['accuracy'] - blurred_results['accuracy']:.4f}\n")
        f.write(f"Macro-F1 drop: {original_results['macro_f1'] - blurred_results['macro_f1']:.4f}\n")

    save_confusion_matrix_pdf(
        blurred_results["y_true"],
        blurred_results["y_pred"],
        "ConvNeXtTiny Background-Blurred Confusion Matrix",
        output_dir / "background_blurred_confusion_matrix.pdf"
    )

    # Balanced examples: 2 per class
    class_to_samples = {}
    for image_path, label in raw_test_dataset.samples:
        class_name = class_names[label]
        class_to_samples.setdefault(class_name, []).append((image_path, label))

    selected_samples = []
    for class_name, samples in class_to_samples.items():
        selected_samples.extend(samples[:2])

    for i, (image_path, true_label) in enumerate(selected_samples, start=1):
        original_img = load_original_image(image_path)
        blurred_img = blur_background_keep_center(image_path)

        true_class = class_names[true_label]

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Original\nTrue: {true_class}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(blurred_img)
        plt.title("Background Blurred")
        plt.axis("off")

        plt.tight_layout()

        save_path = output_dir / f"example_{i:02d}_true-{true_class}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        plt.close()

        print("Saved:", save_path)


# ==========================================================
# SUMMARY TABLE
# ==========================================================

def save_model_comparison_table(results):
    save_path = RESULTS_DIR / "model_comparison_summary.txt"

    with open(save_path, "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("========================\n\n")
        f.write("Model\tAccuracy\tMacro-F1\tWeighted-F1\tBalanced Accuracy\tInference Time\tParameters\n")

        for r in results:
            f.write(
                f"{r['model_name']}\t"
                f"{r['accuracy']:.4f}\t"
                f"{r['macro_f1']:.4f}\t"
                f"{r['weighted_f1']:.4f}\t"
                f"{r['balanced_accuracy']:.4f}\t"
                f"{r['inference_time']:.6f}\t"
                f"{r['total_params']:,}\n"
            )

    print("Saved:", save_path)


# ==========================================================
# RUN EVERYTHING
# ==========================================================

if __name__ == "__main__":
    results = []

    custom_result = train_model(
        model_name="custom_cnn",
        model_builder=build_custom_cnn,
        batch_size=BATCH_SIZE_CUSTOM,
        epochs=EPOCHS_CUSTOM,
        learning_rate=0.001
    )
    results.append(custom_result)

    efficientnet_result = train_model(
        model_name="efficientnetv2",
        model_builder=build_efficientnetv2,
        batch_size=BATCH_SIZE_TRANSFER,
        epochs=EPOCHS_TRANSFER,
        learning_rate=0.0001
    )
    results.append(efficientnet_result)

    convnext_result = train_model(
        model_name="convnexttiny",
        model_builder=build_convnexttiny,
        batch_size=BATCH_SIZE_TRANSFER,
        epochs=EPOCHS_TRANSFER,
        learning_rate=0.0001
    )
    results.append(convnext_result)

    save_model_comparison_table(results)

    run_gradcam_convnexttiny(convnext_result["model"])
    run_background_sensitivity(convnext_result["model"])

    print("\nAll experiments completed.")
    print("Results saved in:", RESULTS_DIR)