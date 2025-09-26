# =========================================================
# convertputra_fixed.py
# Fixed Putra Bird Classifier for Arduino IDE (consistent preprocessing)
# =========================================================

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib

# -------------------------------
# 1️⃣ Dataset path
# -------------------------------
DATASET_PATH = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\putra_dataset"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

class_names = sorted([d for d in os.listdir(DATASET_PATH)
                      if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')])

classes_with_files = {}
for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)
    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    if len(wav_files) > 0:
        classes_with_files[class_name] = len(wav_files)

class_names_for_report = sorted(list(classes_with_files.keys()))
print(f"Classes: {class_names_for_report}")

# -------------------------------
# 2️⃣ Audio → Mel-spectrogram
# -------------------------------
IMG_SIZE = (96, 96)
N_MELS = 96

def audio_to_mel(path, sr=44100, duration=3.0, img_size=IMG_SIZE):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)

    # Viridis colormap (3 channels) + resize
    S_rgb = matplotlib.colormaps['viridis'](S_norm)[..., :3]
    S_rgb_resized = tf.image.resize(S_rgb, img_size).numpy()
    return S_rgb_resized.astype(np.float32)

# -------------------------------
# 3️⃣ Prepare datasets
# -------------------------------
from keras.applications.mobilenet import preprocess_input

def preprocess_file(file_path, label):
    file_path_str = file_path.numpy().decode('utf-8')
    mel = audio_to_mel(file_path_str)
    mel = preprocess_input((mel * 255).astype(np.float32))
    return mel, label

file_paths, labels = [], []
class_name_to_idx = {name: idx for idx, name in enumerate(class_names_for_report)}
for class_name in class_names_for_report:
    class_folder = os.path.join(DATASET_PATH, class_name)
    for f in os.listdir(class_folder):
        if f.endswith(".wav"):
            file_paths.append(os.path.join(class_folder, f))
            labels.append(class_name_to_idx[class_name])

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

def create_dataset(paths, labels, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(preprocess_file, [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, IMG_SIZE+(3,)), tf.ensure_shape(y, ())))
    if shuffle:
        dataset = dataset.shuffle(500)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_paths, train_labels)
val_dataset = create_dataset(val_paths, val_labels, shuffle=False)
test_dataset = create_dataset(test_paths, test_labels, shuffle=False)

# -------------------------------
# 4️⃣ Build tiny MobileNetV1 model
# -------------------------------
from keras.applications import MobileNet

base_model = MobileNet(input_shape=IMG_SIZE+(3,), alpha=0.25, include_top=False, weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(class_names_for_report), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 5️⃣ Train model
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)

# -------------------------------
# 6️⃣ Evaluate
# -------------------------------
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.2%}")

# -------------------------------
# 7️⃣ Quantize to int8 TFLite
# -------------------------------
def representative_dataset():
    for images, _ in train_dataset.take(50):
        yield [images]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = "putra_model_int8_fixed.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Quantized TFLite saved: {tflite_path}, size = {len(tflite_model)/1024:.1f} KB")

model.save("putra_bird_model_2.h5")
print("✅ Keras model saved for float TFLite conversion")
