# Portenta H7 + I²S Microphone (INMP441) — End-to-End Setup (Arduino IDE)

This README.md is a **practical, step-by-step guide** to connect, configure, and test a digital I²S microphone (**INMP441**) on the **Arduino Portenta H7 (STM32H747XI)** using the **Arduino IDE**.  
It walks you through **installing board support, wiring the mic, basic audio capture, FFT processing with CMSIS-DSP, and moving toward ML inference with CMSIS-NN / TensorFlow Lite Micro / Edge Impulse**.

---

## Table of Contents

1. [Hardware & Software Needed](#1-hardware--software-needed)  
2. [Install Arduino Core for Portenta H7](#2-install-arduino-core-for-portenta-h7)  
3. [Wire the Microphone (INMP441)](#3-wire-the-microphone-inmp441)  
4. [Test Mic with Arduino I²S Library](#4-test-mic-with-arduino-i²s-library)  
5. [Visualize Audio in Serial Plotter](#5-visualize-audio-in-serial-plotter)  
6. [Integrate CMSIS-DSP (FFT / Filtering)](#6-integrate-cmsis-dsp-fft--filtering)  
7. [Minimal FFT Test on Mic Samples](#7-minimal-fft-test-on-mic-samples)  
8. [Move Towards ML (CMSIS-NN / TFLite / Edge Impulse)](#8-move-towards-ml-cmsis-nn--tflite--edge-impulse)  
9. [Verify and Tune](#9-verify-and-tune)  
10. [Common Pitfalls Checklist](#10-common-pitfalls-checklist)  
11. [Summary](#11-summary)

---

## 1) Hardware & Software Needed

- **Board:** Arduino Portenta H7 (STM32H747XI MCU) with headers  
- **Mic:** Digital I²S MEMS mic (**INMP441**) with header pins  

### Microphone (Mic)

- **Type:** Digital I²S MEMS microphone  
- **Model:** INMP441  
- **Connection:** Header pins  
- **Required Signal Lines:**  
  - **Word Select (LRCK/WS):** Left/right channel select  
  - **Bit Clock (SCK/BCLK):** Clock signal for data transfer  
  - **Serial Data (SD):** Audio data output  

- **Cables:** USB-C data cable, female-to-female jumper wires (0.1")  
- **Optional:** Oscilloscope or USB logic analyzer to check I²S signals  

> Note: No ST-LINK required for Arduino IDE. All programming and debugging uses **USB-C**.

---

## 2) Install Arduino Core for Portenta H7

1. Install the latest **Arduino IDE**.  
2. Go to **Boards Manager** → search for **Arduino Mbed OS Boards**.  
3. Install the package that includes **Arduino Portenta H7**.  
4. Select your board: **Tools → Board → Arduino Portenta H7 (M7 core)**.  
5. Select the correct COM port under **Tools → Port**.

---

## 3) Wire the Microphone (INMP441)

Use **female-to-female jumper wires**:

| Mic Pin        | Portenta Pin (Arduino I²S default) | Notes                        |
|----------------|----------------------------------|------------------------------|
| VDD            | 3.3 V                             | Confirm mic voltage          |
| GND            | GND                               | Ground                       |
| SD (DATA)      | DMIC DO (J1)                      | I²S data input               |
| SCK (BCLK)     | DMIC CK (J1)                      | Bit clock                    |
| WS / LR (LRCK) | I2SWS (J1)                        | Word select                  |
| L/R            | GND (Left)                        | Choose left channel               |

---
## 4) Install Arduino I²S Library
- Install Arduino_AdvancedAnalog from Library Manager

## 5) Test Mic with Arduino I²S Library

```cpp
#include <Arduino_AdvancedAnalog.h>

// Correct pins for Portenta H7 J1 connector
// WS = I2SWS, CK = DMIC CK, SDI = DMIC DO
AdvancedI2S i2s(PB_9, PE_3, PB_2, NC, NC);  

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== INMP441 Mic Test ===");

  // Resolution, sample rate, buffer size, queue depth
  if (!i2s.begin(AN_I2S_MODE_IN, 16000, 256, 32)) {
    Serial.println("❌ Failed to start I2S!");
    while (1);
  }

  Serial.println("✅ I2S started, listening to INMP441...");
  Serial.println("Open Serial Plotter to view waveform");
}

void loop() {
  if (i2s.available()) {
    SampleBuffer buf = i2s.read();
    for (int i = 0; i < buf.size(); i++) {
      Serial.println(buf[i]);   // Raw waveform → Serial Plotter
    }
    buf.release();
  }
}

```
- Open Tools → Serial Plotter to visualize the waveform.
- Clap or speak → you should see waveform changes.

## 6) Test Mic with Arduino I²S Library

1. Visualize Audio in Serial Plotter
2. Go to Tools → Serial Plotter (baud 115200).
3. Observe real-time waveform changes when audio is detected.
4. If nothing appears: check wiring, swap WS/BCLK pins, or lower sample rate (e.g., 8000 Hz).
   
## 7) Integrate CMSIS-DSP (FFT / Filtering)

1. Open Library Manager → install Arduino_CMSIS-DSP.
2. Use it for filtering, FFT, RMS, or other signal processing.

- Example FFT:
```cpp
#include <Arduino_AdvancedAnalog.h>
#include <arm_math.h>

// Correct pins for Portenta H7 J1 connector
// WS = I2SWS, CK = DMIC CK, SDI = DMIC DO
AdvancedI2S i2s(PB_9, PE_3, PB_2, NC, NC);  

#define FFT_SIZE 256
float32_t input[FFT_SIZE];
float32_t output[FFT_SIZE];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== INMP441 FFT Test ===");

  if (!i2s.begin(AN_I2S_MODE_IN, 16000, 256, 32)) {
    Serial.println("❌ Failed to start I2S!");
    while (1);
  }
}

void loop() {
  // Collect FFT_SIZE samples
  for (int i = 0; i < FFT_SIZE; i++) {
    while (!i2s.available());
    SampleBuffer buf = i2s.read();
    input[i] = (float32_t)buf[0];  // Mono input
    buf.release();
  }

  arm_rfft_fast_instance_f32 fft;
  arm_rfft_fast_init_f32(&fft, FFT_SIZE);
  arm_rfft_fast_f32(&fft, input, output, 0);

  for (int i = 0; i < FFT_SIZE/2; i++) {
    float mag = sqrtf(output[2*i]*output[2*i] + output[2*i+1]*output[2*i+1]);
    Serial.println(mag);
  }
}
```
- Plot frequency bins in Serial Plotter instead of waveform.
 > [https://www.pschatzmann.ch/home/2023/07/10/arduino-uno-r4-fft-using-cmsis-dsp/]
 > [https://www.pschatzmann.ch/home/2025/02/21/reverse-fft/]

## 8) Minimal FFT Test on Mic Samples

1. Clap → strong low-frequency bins.
2. Voice → multiple harmonics.

## 9) Move Towards ML (CMSIS-NN / TFLite / Edge Impulse)

1. CMSIS-NN directly → low-level NN operators
2. TensorFlow Lite Micro → load .tflite model for inference
3. Edge Impulse Arduino library → easiest for end-to-end workflow

 > All three automatically use CMSIS-NN under the hood on Cortex-M7.
 > [https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/]

## 10) Verify and Tune

1. If mic values are always zero → check WS/SCK/SD wiring
2. If only noise → confirm I2S.begin() sample rate and bit depth match INMP441
3. If Serial Plotter too fast → print every Nth sample to slow down output

## 11) Common Pitfalls Checklist

1. Wrong pins
2. Mic voltage: must be 3.3V
3. Serial baud mismatch: ensure 115200
4. Buffer overruns: lower FFT size (128 or 64) if needed

  > USB cable: must be data-capable

## 12) Summary

1. Install Arduino IDE & Portenta H7 board support
2. Wire INMP441 → Portenta H7 using jumper wires
3. Run I²S test sketch → check waveform in Serial Plotter
4. Add CMSIS-DSP → FFT, filtering, energy detection
5. Move to CMSIS-NN / TFLite Micro / Edge Impulse for ML classification


