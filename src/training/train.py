"""
train.py

Script d'entraînement principal : charge les datasets (train/val) via
data_loader, construit et entraîne les modèles (CNN baseline, ResNet50,
VGG16) définis dans models.py, puis évalue leurs performances (accuracy,
recall) sur le dataset de validation, et affiche le meilleur modèle
selon le recall.
"""

from src.training.data_loader import get_train_dataset, get_val_dataset
from src.training.models import build_baseline_cnn, build_resnet50, build_vgg16

train_dataset = get_train_dataset()
val_dataset = get_val_dataset()

# CNN Baseline
print("=== Training Baseline CNN ===")
baseline_model = build_baseline_cnn(num_classes=3)
baseline_history = baseline_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
baseline_loss, baseline_accuracy, baseline_recall = baseline_model.evaluate(val_dataset)
print(f"Baseline CNN - Accuracy: {baseline_accuracy:.4f} - Recall: {baseline_recall:.4f}")

# ResNet50
print("=== Training ResNet50 ===")
resnet_model = build_resnet50(num_classes=3)
resnet_history = resnet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
resnet_loss, resnet_accuracy, resnet_recall = resnet_model.evaluate(val_dataset)
print(f"ResNet50 - Accuracy: {resnet_accuracy:.4f} - Recall: {resnet_recall:.4f}")

# VGG16
print("=== Training VGG16 ===")
vgg_model = build_vgg16(num_classes=3)
vgg_history = vgg_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
vgg_loss, vgg_accuracy, vgg_recall = vgg_model.evaluate(val_dataset)
print(f"VGG16 - Accuracy: {vgg_accuracy:.4f} - Recall: {vgg_recall:.4f}")

# Comparaison finale
results = {
    "Baseline CNN": baseline_recall,
    "ResNet50": resnet_recall,
    "VGG16": vgg_recall,
}

best_model = max(results, key=results.get)
print(f"\n=== Meilleur modèle selon le recall : {best_model} (recall={results[best_model]:.4f}) ===")