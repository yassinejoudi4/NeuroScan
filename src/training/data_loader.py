"""
data_loader.py

Charge les images IRM depuis Data/train|val|test/<classe>/ et construit
des tf.data.Dataset prêts pour l'entraînement.

- Train : chaque classe est chargée séparément. Meningioma et pituitary
  (classes minoritaires) sont répétées + augmentées online (flip, rotation,
  zoom, contraste) jusqu'à atteindre TARGET_COUNT (1140, taille de glioma),
  afin d'équilibrer les classes sans dupliquer physiquement des fichiers
  sur disque. Les 3 classes sont ensuite concaténées, mélangées, et batchées.

- Val/Test : chargement brut des mêmes classes, sans augmentation ni
  rééquilibrage, pour une évaluation fidèle aux données réelles.

Labels en one-hot (glioma=0, meningioma=1, pituitary=2), cohérent avec
label_mode="categorical" et la loss categorical_crossentropy.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TARGET_COUNT = 1140

CLASSES = ["glioma", "meningioma", "pituitary"]
NUM_CLASSES = len(CLASSES)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


def load_class_images(class_dir, label_index, augment, target_count=None):
    file_pattern = os.path.join(class_dir, "*")
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True, seed=42)

    def load_and_preprocess(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        label = tf.one_hot(label_index, NUM_CLASSES)
        return image, label

    dataset = dataset.map(load_and_preprocess)

    if augment:
        dataset = dataset.repeat()
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))

    if target_count is not None:
        dataset = dataset.take(target_count)

    return dataset


def get_train_dataset():
    glioma = load_class_images("Data/train/glioma", 0, augment=False, target_count=TARGET_COUNT)
    meningioma = load_class_images("Data/train/meningioma", 1, augment=True, target_count=TARGET_COUNT)
    pituitary = load_class_images("Data/train/pituitary", 2, augment=True, target_count=TARGET_COUNT)

    dataset = glioma.concatenate(meningioma).concatenate(pituitary)
    dataset = dataset.shuffle(3 * TARGET_COUNT, seed=42)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def load_eval_class_images(class_dir, label_index):
    return load_class_images(class_dir, label_index, augment=False, target_count=None)


def get_val_dataset():
    datasets = [load_eval_class_images(f"Data/val/{cls}", i) for i, cls in enumerate(CLASSES)]
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset = dataset.concatenate(d)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_test_dataset():
    datasets = [load_eval_class_images(f"Data/test/{cls}", i) for i, cls in enumerate(CLASSES)]
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset = dataset.concatenate(d)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset