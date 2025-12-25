import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import pandas as pd
import os

#Path
MODEL_PATH = r"model\brain_tumor_multiclass_finetuned.h5"
DATASET_PATH = r"C:\Users\Asus\Desktop\Brain Tumor Detection\dataset\val"


ENABLE_SAMPLING = True      
SAMPLE_SIZE = 500           # Only verify through a sample of 500 images


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")


print("Loading dataset...")
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_indices = generator.class_indices
num_classes = len(class_indices)
print(f"Detected classes: {list(class_indices.keys())}")


if ENABLE_SAMPLING:
    print(f"\nSampling {SAMPLE_SIZE} images evenly from all {num_classes} classes...")

    sample_per_class = SAMPLE_SIZE // num_classes
    sampled_filepaths = []
    sampled_labels = []

    # Collecting equal files from all classes
    for class_name, class_index in class_indices.items():
        class_files = [fp for fp, lbl in zip(generator.filepaths, generator.classes) if lbl == class_index]
        sampled_files = class_files[:sample_per_class]
        sampled_filepaths.extend(sampled_files)
        sampled_labels.extend([class_index]*len(sampled_files))


    df = pd.DataFrame({
        'filename': sampled_filepaths,
        'class': sampled_labels
    })
    df['class_name'] = df['class'].map({v: k for k, v in class_indices.items()})

  
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='class_name',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
else:
    print("\nUsing full dataset for evaluation...")
    test_generator = generator

print("\nGenerating predictions...")
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes


print("\nCreating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
labels = list(test_generator.class_indices.keys())

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy Score:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

print("\nDone!")
