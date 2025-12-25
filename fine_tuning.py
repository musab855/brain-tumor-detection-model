import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # pyright: ignore[reportMissingImports]
import os

MODEL_PATH = r"model\brain_tumor_multiclass.keras"  # old model
TRAIN_DIR = r"C:\Users\Asus\Desktop\Brain Tumor Detection\Extra Dataset\Brain_Tumor_Detection_Dataset\Training"
VAL_DIR = r"C:\Users\Asus\Desktop\Brain Tumor Detection\Extra Dataset\Brain_Tumor_Detection_Dataset\Testing"
SAVE_PATH = r"model\brain_tumor_multiclass_finetuned.h5"  #new model

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

for layer in model.layers[:-3]:
    layer.trainable = False

for layer in model.layers[-3:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    SAVE_PATH,
    monitor='val_loss',
    save_best_only=True
)


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\nFine-tuned model saved at: {SAVE_PATH}")
