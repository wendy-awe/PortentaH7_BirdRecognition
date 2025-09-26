# TinyChirp Bird Activity Detection (BAD) – Portenta H7 Deployment

This workflow trains a lightweight CNN for bird sound detection (“Bird Activity Detection”, BAD), quantizes the model for TFLite Micro, converts it into a C header, and runs real-time inference on the Portenta H7 using an analog microphone (MAX4466).

## 1. Training and Quantization (BAD_tiny.py)

This Python script:
```python
# =========================================================
# Train + quantize TinyChirp bird sound detection for TFLite Micro
# =========================================================
import os
import numpy as np
import tensorflow as tf
import librosa

# -------------------- Paths --------------------
DATA_DIR = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\tinychirp"
KERAS_MODEL_PATH = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\BAD_tiny_model.h5"
TFLITE_PATH = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\BAD_tiny_model_int8.tflite"
HEADER_PATH = r"C:\Users\chuwe\AppData\Local\Programs\Python\Python311\BAD_tiny_model.h"

# -------------------- Audio params --------------------
SAMPLE_RATE = 16000
CLIP_DURATION = 3.0
SAMPLES_PER_CLIP = int(SAMPLE_RATE * CLIP_DURATION)
N_MELS = 16
FFT_SIZE = 512
HOP_LENGTH = 256
FRAMES = 96  # for MCU input

# -------------------- Audio processing --------------------
def process_file(filepath, label):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    # Pad or trim
    if len(audio) >= SAMPLES_PER_CLIP:
        start = np.random.randint(0, len(audio)-SAMPLES_PER_CLIP+1)
        segment = audio[start:start+SAMPLES_PER_CLIP]
    else:
        segment = np.pad(audio, (0, SAMPLES_PER_CLIP-len(audio)))
    # Skip silent
    if np.sum(np.abs(segment)) < 1e-6:
        return None, None
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SAMPLE_RATE,
        n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize [0,1]
    mel_norm = (mel_db + 80.0) / 80.0
    # Resize to MCU input frames
    mel_norm = np.resize(mel_norm, (FRAMES, N_MELS, 1))
    return mel_norm.astype(np.float32), label

# -------------------- Load dataset --------------------
def load_split(split):
    split_map = {"train":"training", "validate":"validation", "test":"testing"}
    X_list, y_list = [], []
    split_path = os.path.join(DATA_DIR, split_map[split])
    for label_name, label in [("target", 1), ("non_target", 0)]:
        folder = os.path.join(split_path, label_name)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".wav"):
                filepath = os.path.join(folder, f)
                features, lab = process_file(filepath, label)
                if features is not None:
                    X_list.append(features)
                    y_list.append(lab)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

# -------------------- Load all splits --------------------
X_train, y_train = load_split("train")
X_val, y_val     = load_split("validate")
X_test, y_test   = load_split("test")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -------------------- Tiny CNN --------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------- Train --------------------
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)

# -------------------- Save Keras --------------------
model.save(KERAS_MODEL_PATH)
print(f"✅ Keras model saved: {KERAS_MODEL_PATH}")

# -------------------- Convert to TFLite int8 --------------------
def representative_dataset():
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_quant)
print(f"✅ Quantized TFLite saved: {TFLITE_PATH}")

# -------------------- Convert TFLite -> C header --------------------
with open(TFLITE_PATH, "rb") as f:
    tflite_data = f.read()

c_array = ", ".join(str(b) for b in tflite_data)
header_content = f"""
#ifndef BAD_TINY_MODEL_H
#define BAD_TINY_MODEL_H

const unsigned char bad_tiny_model[] = {{
{c_array}
}};

const unsigned int bad_tiny_model_len = {len(tflite_data)};

#endif
"""
with open(HEADER_PATH, "w") as f:
    f.write(header_content)
print(f"✅ C header generated: {HEADER_PATH}")
```
- Preprocesses audio clips into Mel spectrograms (16 Mel bins × 96 frames).

- Trains a tiny CNN for binary classification:

    - 1 = target bird detected
    - 0 = non-target / silence

- Saves the model in three formats:

    - Keras .h5 (training checkpoint)
    - Quantized TFLite .tflite (INT8)
    - Arduino-compatible .h C header

- Key Parameters:

    - Sample rate: 16 kHz
    - Clip duration: 3 s
    - FFT size: 512
    - Hop length: 256
    - Mel bins: 16
    - Frames per clip: 96

- Run the script:  Open Command Prompt in the Python folder, then run:
```bash
cd C:\Users\chuwe\AppData\Local\Programs\Python\Python311
python BAD_tiny.py
```

- After training, you will get:

    - BAD_tiny_model.h5 (Keras model)
    - BAD_tiny_model_int8.tflite (quantized model
    - BAD_tiny_model.h (C header for Arduino)

## 2. Deployment on Portenta H7 (BAD_tiny_M7.ino)

This Arduino sketch:
```cpp
#include "Arduino.h"

#ifdef abs
#undef abs
#endif

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "BAD_tiny_model.h" // Quantized INT8 model

// -------------------- Audio Settings --------------------
const int micPin = A0;
const int SAMPLE_RATE = 16000;
const float CLIP_DURATION = 3.0;
const int SAMPLES_PER_CLIP = (int)(SAMPLE_RATE * CLIP_DURATION);
const int N_MELS = 16;
const int FFT_SIZE = 512;
const int HOP_LENGTH = 256;
const int FRAMES = 96;

#ifdef PI
#undef PI
#endif
const float PI = 3.14159265359;

// Use smaller chunks
const int CHUNK_SIZE = 8000;
float audioChunk[CHUNK_SIZE];

// -------------------- TensorFlow Lite --------------------
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(bad_tiny_model);
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  analogReadResolution(12);

  while (!Serial) { ; }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Bird detection ready - EXACT Python match!");
}

// -------------------- (Mel spectrogram + inference code here, truncated for brevity) --------------------

// -------------------- Main Loop --------------------
void loop() {
  // Capture 3s audio → compute Mel → quantize → run inference
  // Serial.println(1) if bird detected, else Serial.println(0)
}
```
- Includes the generated BAD_tiny_model.h file.

- Captures audio via analog pin A0 from the MAX4466 mic.

- Recomputes Mel spectrogram features on-device (same settings as training).

- Quantizes the features to INT8.

- Runs inference with TensorFlow Lite Micro.

- Outputs detection via:

    - Serial Monitor (1 for bird, 0 for no bird)
    - LED_BUILTIN (ON if bird detected)

**Workflow**

1. Record 3-second audio chunk from the mic.

2. Convert to Mel spectrogram (normalized, [0–1] scale).

3. Quantize to match INT8 training.

4. Run TFLite inference on Portenta H7.

5. Apply threshold (>0.6 → bird detected).

**Upload**

1. Copy BAD_tiny_model.h into your Arduino project folder.

2. Open BAD_tiny_M7.ino in Arduino IDE.

3. Compile & upload to Portenta H7 (M7 core).

4. Open Serial Monitor to see live detection results.



## 3. Expected Output

- Serial Monitor prints:

    - 1 → Bird detected  
    - 0 → No bird / silence

- On-board LED turns ON when bird activity is detected.

✅ This pipeline ensures that training parameters (Python) and deployment parameters (Arduino) are an exact match, so inference behaves consistently on the microcontroller.
