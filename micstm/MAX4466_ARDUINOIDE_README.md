# MAX4466 on Portenta H7 — CMSIS-DSP & TensorFlow Lite Micro Workflow

## Table of Contents
1. [Hardware & Software Needed](#hardware--software-needed)
2. [Wiring the MAX4466](#wiring-the-max4466)
3. [Install Arduino Core for Portenta H7](#install-arduino-core-for-portenta-h7)
4. [Analog Mic Test (Serial Plotter)](#analog-mic-test-serial-plotter)
5. [CMSIS-DSP FFT Example](#cmsis-dsp-fft-example)
6. [TensorFlow Lite Micro Inference Example](#tensorflow-lite-micro-inference-example)
7. [Tips & Common Pitfalls](#tips--common-pitfalls)

---
### 1️⃣ Hardware & Software Needed

**Hardware:**

* Portenta H7
* MAX4466 microphone module
* Jumper wires

**Software:**

* Arduino IDE (latest)
* Portenta H7 Arduino Core
* CMSIS-DSP library (comes with ArduinoCore STM32)
* TensorFlow Lite Micro library (Arduino\_TensorFlowLite)

### 2️⃣ Wiring the MAX4466

| MAX4466 Pin | Portenta H7         |
| ----------- | ------------------- |
| VCC         | 3.3V                |
| GND         | GND                 |
| OUT         | A0 (or any ADC pin) |

⚠️ Do not connect MAX4466 to I²S pins — it’s analog.

### 3️⃣ Install Arduino Core for Portenta H7

* Arduino IDE → File → Preferences → Additional Boards Manager URLs

```
https://downloads.arduino.cc/packages/package_index.json
```

* Tools → Board → Board Manager, search Portenta H7, install.
* Install Arduino\_TensorFlowLite library from Library Manager.

### 4️⃣ Analog Mic Test (Serial Plotter)
```cpp
const int micPin = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int sample = analogRead(micPin);
  Serial.println(sample); // raw ADC value
  delay(1);
}
```
* Open Serial Plotter to see live waveform.
* Normalize values: `(analogRead() - 2048)/2048.0` → -1.0 … 1.0

### 5️⃣ CMSIS-DSP FFT Example
```cpp
#include "arm_math.h"

#define SAMPLES 256
#define MIC_PIN A0

float32_t adcBuffer[SAMPLES];
float32_t fftOutput[SAMPLES];
float32_t magOutput[SAMPLES];

arm_rfft_fast_instance_f32 fft;

void setup() {
  Serial.begin(115200);
  arm_rfft_fast_init_f32(&fft, SAMPLES);
}

void loop() {
  // Collect samples
  for(int i=0; i<SAMPLES; i++) {
    adcBuffer[i] = (analogRead(MIC_PIN) - 2048) / 2048.0f; // normalize
  }

  // Perform FFT
  arm_rfft_fast_f32(&fft, adcBuffer, fftOutput, 0);
  arm_cmplx_mag_f32(fftOutput, magOutput, SAMPLES/2);

  // Print magnitude spectrum
  for(int i=0; i<SAMPLES/2; i++) {
    Serial.println(magOutput[i]);
  }
  delay(500);
}
```
* Visualize frequency content → can detect peaks (like bird calls).

### 6️⃣ TensorFlow Lite Micro Inference Example
```cpp
#include <TensorFlowLite.h>
#include "model.h" // Your TFLite Micro model header
#include "arduino_tflite_micro.h" // Arduino wrapper

#define MIC_PIN A0
#define SAMPLE_RATE 16000
#define AUDIO_BUFFER_SIZE 16000

float audioBuffer[AUDIO_BUFFER_SIZE];

void collectAudio() {
  for(int i=0; i<AUDIO_BUFFER_SIZE; i++) {
    audioBuffer[i] = (analogRead(MIC_PIN) - 2048) / 2048.0f; // normalize
  }
}

void setup() {
  Serial.begin(115200);
  tfliteSetup(); // Initialize TFLite interpreter
}

void loop() {
  collectAudio();              // ADC → normalized audio
  tfliteRunInference(audioBuffer, AUDIO_BUFFER_SIZE); // run model
  tflitePrintResults();        // print probabilities / predictions
  delay(1000);
}
```
**Notes:**

* Replace `model.h` with your exported TFLite Micro model (Edge Impulse or TFLite converter).
* `tfliteRunInference()` and `tflitePrintResults()` are Arduino helper functions wrapping TFLite Micro calls.

### 7️⃣ Tips & Common Pitfalls

* MAX4466 is analog → use ADC, not I²S.
* Sample Rate: ADC is software-controlled; 16 kHz is reasonable.
* Normalization: Convert raw 12-bit ADC to -1.0 … 1.0 before DSP / ML.
* FFT & ML: Both work identically as with I²S; source only affects sampling code.
* Noise: MAX4466 may need software filtering (moving average) for stable results.
