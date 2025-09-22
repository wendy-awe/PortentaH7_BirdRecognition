# Module 5: Training a Species Classifier Full Pipeline

## 1. Data Preparation
Raw bird call audio recordings in `.wav` format were converted into spectrograms, normalised between 0 and 1 for numerical stability during model training.

## 2. Preprocessing for MobileNetV3
Spectrograms were scaled to 0–255, then normalised using Keras' `preprocess_input()`. Images were resized to 224×224×3, the standard input for MobileNetV3.

## 3. Model Architecture
A pre-trained `MobileNetV3Small` (ImageNet) was used as a frozen base (`include_top=False`). Custom layers added on top:
- `GlobalAveragePooling2D` to vectorize features  
- `Dropout(0.2)` to reduce overfitting  
- `Dense` with softmax activation for species classification  

## 4. Model Compilation and Training
Compiled with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric. Training used 80% of the data with 10–20% for validation, for 20 epochs.

## 5. Evaluation
Model was evaluated on the unseen test set (~10% of data) to measure generalization performance.

## 6. Classification Report
Generated a report with:
- Precision: correct positive predictions  
- Recall: correctly identified positives  
- F1-score: balance of precision and recall  
- Support: sample count per class  

## 7. Confusion Matrix
Provided a visual summary of true vs predicted labels. Diagonal = correct predictions, off-diagonal = misclassifications.

## 8. Sample Predictions
Test samples were shown with true labels, predicted labels, and confidence scores to highlight correct classifications and errors.

## 9. Training History
Accuracy and loss curves across epochs were plotted to detect underfitting, overfitting, or steady improvement.

---

# Exercises

## 1. Replace Dataset
 **Replaced the Seabird dataset with the Putra dataset from [UPM Drive](https://drive.google.com/drive/folders/1HvR_sbEx5rzWjgTWkyjplUXgzsQkEgzH?usp=sharing).**

### 1. Dataset Overview
- **Seabird dataset**: ~8,500 samples, 10 classes, balanced distribution.
  <img width="453" height="311" alt="image" src="https://github.com/user-attachments/assets/f4d9cbea-ded7-4297-b670-80b1ddde19ab" />
  <img width="344" height="303" alt="image" src="https://github.com/user-attachments/assets/054ac111-a60b-4540-8bc7-26a16454c4e1" />
  <img width="940" height="392" alt="image" src="https://github.com/user-attachments/assets/118812a7-c3dd-4e21-9437-f64497a69656" />

- **Putra dataset**: ~1,000 samples, 10 classes (one empty folder ignored), smaller size increases overfitting risk.
  <img width="499" height="303" alt="image" src="https://github.com/user-attachments/assets/ac7ac078-9a76-4c0b-bb6b-34bb0a13a2f1" />
  <img width="369" height="320" alt="image" src="https://github.com/user-attachments/assets/53b5d407-88c3-47a0-bb87-8acb7a5f9136" />
  <img width="916" height="371" alt="image" src="https://github.com/user-attachments/assets/ea11bba1-96b1-4bee-bb21-530d917fe335" />

### 2. Model Training
- MobileNetV3Small used as frozen feature extractor with Dense classification layer.  
- Seabird: training & validation curves gradually overlapped → good generalization.  
- Putra: curves had a persistent gap → high validation accuracy but overfitting observed.

### 3. Classification Performance
- **Seabird**: ~81% training & validation accuracy, 82.7% test accuracy, macro F1 ≈ 0.83. Best class F1 = 0.988, weakest ≈ 0.71–0.75.  
- **Putra**: ~95% training, 98% validation, 89.1% test accuracy, macro F1 ≈ 0.888. Strongest classes F1 ≈ 0.95, weakest ≈ 0.70.

### 4. Sample Predictions
- Seabird: mostly correct, moderate confidence (0.5–0.9), some confusion between similar calls.  
- Putra: high confidence (0.8–0.99), but confusion remained between acoustically similar species.

### 5. Discussion
- **Seabird**: larger, diverse → better generalization, overlapping curves, slightly lower test accuracy (~82.7%).  
- **Putra**: smaller, cleaner → faster training, higher peak test accuracy (~89.1%), more prone to overfitting.  
- Reducing epochs from 20 → 15 decreased overfitting slightly but also reduced accuracy.  
- Overall: Seabird preferable for robust generalization, Putra achieves higher peak accuracy but less generalizable.

## 2. Copy the code and dataset to your laptop, convert script to regular Python (.py)
## 3. Convert the trained model to TensorFlow Lite (.tflite format)

### 1. Put code in local, name: correctionconvertputra_small.py (including convert to TFLite & quantization) and run module in IDLE (will see graphs & tables, predictions)
```python
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
```
Command:
```bash
python correctionconvertputra_small.py
```
  > **putra_model_int8_fixed.tflite** is created.

### 2. Put code in local, name: correctionputra_convert_tflite_h.py (convert TFLite → Arduino header (H file) )
```python
# =========================================================
# correctionputra_convert_tflite_h.py
# Convert a TensorFlow Lite model (.tflite) into a C header (.h)
# suitable for Arduino / STM32 (uint8_t + 8-byte alignment)
# =========================================================

import sys
import os

def convert_tflite_to_header(tflite_path, header_path):
    if not os.path.exists(tflite_path):
        print(f"❌ ERROR: File not found -> {tflite_path}")
        return

    # Automatically create header guard and array name from filename
    header_guard = os.path.basename(header_path).replace('.', '_').upper()
    array_name = os.path.splitext(os.path.basename(header_path))[0]

    # Step 1: Read the TFLite model
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # Step 2: Write to C header file
    with open(header_path, "w") as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        f.write("#include <stdint.h>\n\n")

        # Declare array with 8-byte alignment
        f.write(f"alignas(8) const uint8_t {array_name}[] = {{")
        for i, b in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("\n ")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n\n")

        # Store model length
        f.write(f"const unsigned int {array_name}_len = {len(tflite_model)};\n\n")

        f.write(f"#endif // {header_guard}\n")

    print(f"✅ Conversion complete: {tflite_path} → {header_path}")
    print(f"Model size: {len(tflite_model)/1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correctionputra_convert_tflite_h.py <input_model.tflite> <output_model.h>")
    else:
        convert_tflite_to_header(sys.argv[1], sys.argv[2])
```
Command:
```bash
python correctionputra_convert_tflite_h.py putra_model_int8_fixed.tflite corre20_putramodel.h
```
  > **corre20_putramodel.h** is created.

### 3. Put code in local, name: sample_mel_Crested-serpent_Eagle.py (generate .npy sample Mel)
```python
import os
import numpy as np
import librosa
import tensorflow as tf

# -------------------------------
# 1️⃣ Exact WAV file path
# -------------------------------
wav_file = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\putra_wav\Crested-serpent_Eagle\20200302_090000_KSNP_0_00295_472.wav"
bird_class = os.path.basename(os.path.dirname(wav_file))
wav_name = os.path.basename(wav_file)

print(f"Selected WAV file: {wav_name}")
print(f"Bird class: {bird_class}")

# -------------------------------
# 2️⃣ Convert audio to Mel-spectrogram (96x96x3)
# -------------------------------
def audio_to_mel_rgb(path, sr=44100, n_mels=96, duration=3.0, img_size=(96, 96)):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    # Convert power to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Normalize to [0,1]
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
    # Resize to 96x96 (still grayscale)
    S_resized = tf.image.resize(S_norm[np.newaxis, ..., np.newaxis], img_size).numpy()[0, ..., 0]
    # ✅ Stack grayscale to 3 channels → (96,96,3)
    S_rgb = np.stack([S_resized] * 3, axis=-1)
    return S_rgb.astype(np.float32)

mel = audio_to_mel_rgb(wav_file)
print(f"✅ Mel shape: {mel.shape}")  # should be (96,96,3)

# -------------------------------
# 3️⃣ Save as .npy
# -------------------------------
output_npy = f"sample_mel_{bird_class}.npy"
np.save(output_npy, mel)
print(f"✅ Saved sample Mel-spectrogram: {output_npy}")
```
Command:
```bash
python sample_mel_Crested-serpent_Eagle.py
```
  > **sample_mel_Crested-serpent_Eagle.npy** is created.

### 4. Put code in local, name: eaglemel_to_h.py (convert sample .npy → Arduino header)
```python
import numpy as np
import tensorflow as tf
import sys

# === Load mel.npy ===
mel_npy = "sample_mel_Crested-serpent_Eagle.npy"
mel = np.load(mel_npy)   # should be (96,96,3)

# === Load TFLite model to read quantization params ===
tflite_model = "putra_model_int8_fixed.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
scale, zero_point = input_details['quantization']

print(f"Model expects input scale={scale}, zero_point={zero_point}")
print("Input shape:", input_details['shape'])

# === Quantize using model's params ===
mel_int8 = np.round(mel / scale + zero_point).astype(np.int8)

print("Mel shape:", mel_int8.shape, "dtype:", mel_int8.dtype)
print("First 10 values:", mel_int8.flatten()[:10])

# === Save to Arduino header ===
header_file = "sample_mel_Crested_serpent_Eagle.h"
with open(header_file, "w") as f:
    f.write("#ifndef SAMPLE_MEL_CRESTED_SERPENT_EAGLE_H\n")
    f.write("#define SAMPLE_MEL_CRESTED_SERPENT_EAGLE_H\n\n")
    f.write("const signed char sample_mel_Crested_serpent_Eagle[] = {\n")
    mel_flat = mel_int8.flatten()
    for i, val in enumerate(mel_flat):
        f.write(f"{val},")
        if (i + 1) % 20 == 0:
            f.write("\n")
    f.write("};\n\n")
    f.write("#endif // SAMPLE_MEL_CRESTED_SERPENT_EAGLE_H\n")

print(f"Header file written: {header_file}, total values = {mel_int8.size}")
```
Command:
```bash
python eaglemel_to_h.py
```
  > **sample_mel_Crested_serpent_Eagle.h** is created.

### 5. Arduino project file: Bird_Classifier_putra.ino (Include corre20_putramodel.h & sample_mel_Crested_serpent_Eagle.h in folder)
```cpp
#include "Arduino.h"

#ifdef abs
#undef abs
#endif
#include <cmath>

// TensorFlow Lite Micro headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your converted model and sample Mel input
#include "corre20_putramodel.h"                // quantized TFLite model
#include "sample_mel_Crested_serpent_Eagle.h"  // generated sample input Mel spectrogram (96x96x3)

// -------------------------
// Bird names
// -------------------------
const char* bird_names[] = {
  "Asian Koel",
  "Black-naped Oriole",
  "Cinereous Tit",
  "Collared Kingfisher",
  "Common Lora",
  "Crested-serpent Eagle",
  "Large-tailed Nightjar",
  "Original",
  "Pied Fantail",
  "Spotted Dove",
  "Zebra Dove"
};

// -------------------------
// Globals
// -------------------------
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(corre20_putramodel);
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena
  constexpr int kTensorArenaSize = 200 * 1024;
  __attribute__((section(".ext_ram"))) uint8_t tensor_arena[kTensorArenaSize];
}

// -------------------------
// Setup
// -------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ Portenta H7 Putra Bird Classifier ready!");
}

// -------------------------
// Loop
// -------------------------
void loop() {
  // Fill input tensor
  for (int i = 0; i < input->bytes; i++) {
    input->data.int8[i] = sample_mel_Crested_serpent_Eagle[i];
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Dequantize output and apply softmax
  int output_size = output->dims->data[output->dims->size - 1];
  float dequant[output_size];
  float max_val = -1000.0;

  for (int i = 0; i < output_size; i++) {
    dequant[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    if (dequant[i] > max_val) max_val = dequant[i];
  }

  float sum_exp = 0.0;
  for (int i = 0; i < output_size; i++) {
    dequant[i] = exp(dequant[i] - max_val);
    sum_exp += dequant[i];
  }
  for (int i = 0; i < output_size; i++) dequant[i] /= sum_exp;

  // Get top-3 predictions
  int top_idx[3] = {0, 0, 0};
  float top_prob[3] = {0.0, 0.0, 0.0};

  for (int i = 0; i < output_size; i++) {
    if (dequant[i] > top_prob[0]) {
      top_prob[2] = top_prob[1]; top_idx[2] = top_idx[1];
      top_prob[1] = top_prob[0]; top_idx[1] = top_idx[0];
      top_prob[0] = dequant[i]; top_idx[0] = i;
    } else if (dequant[i] > top_prob[1]) {
      top_prob[2] = top_prob[1]; top_idx[2] = top_idx[1];
      top_prob[1] = dequant[i]; top_idx[1] = i;
    } else if (dequant[i] > top_prob[2]) {
      top_prob[2] = dequant[i]; top_idx[2] = i;
    }
  }

  // Print results
  Serial.println("Top 3 predictions:");
  for (int i = 0; i < 3; i++) {
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(bird_names[top_idx[i]]);
    Serial.print(" - ");
    Serial.println(top_prob[i], 4);
  }

  delay(2000);
}
```
### 6. Result
```bash
12:27:52.669 -> Top 3 predictions:
12:27:52.669 -> 1: Crested-serpent Eagle - 0.2080
12:27:52.669 -> 2: Black-naped Oriole - 0.0967
12:27:52.669 -> 3: Common Lora - 0.0874
```
- The numbers like 0.2080 are probabilities from the model.
- Multiply by 100 → percentage chance of each bird: 0.2080 × 100 ≈ 21%.
- The highest percentage is the model’s prediction.
- Showing low percentage due to:
   - Small & quantized model – Tiny MobileNet and int8 quantization reduce confidence.
   - Many classes & short clip – Probability is spread over 11 birds from just 3 seconds of audio.
   - Noise & variation – Background sounds or differences in bird calls lower certainty.
 
