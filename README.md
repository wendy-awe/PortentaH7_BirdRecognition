# PortentaH7_BirdRecognition
Bird sound detection project using Portenta H7 + Arduino IDE.
This project aims to implement a bird recognition system on the Portenta H7 microcontroller.
It is developed using Arduino IDE and will include source code, documentation, models, and test. 

## Repository Structure
- `src/` : Source code for firmware and application
- `docs/` : Documentation, design notes, and reports
- `models/` : Machine learning models (trained or converted for MCU deployment)
- `tests/` : Unit tests, hardware tests, and validation scripts

## Portenta H7 + STM32CubeIDE Project Flow
1. **Collect data**: Gather pictures / audio / sensor data on your laptop.

2. **Train ML model**: Use Edge Impulse (online) to train the model.

3. **Convert model to Arduino C/C++ code**: Use Edge Impulse → Arduino Library export (model → .h + .cpp files).

4. **Create Arduino project**: Install required libraries via Library Manager (e.g., CMSIS-DSP, Edge Impulse SDK, I²S for mic, camera library if used).

5. **Add drivers (peripherals)**: Configure GPIO, UART, I²C, SPI, camera, mic (depending on sensors you use).

6. **Integrate ML model**: Copy the exported model files (model.cpp, model.h) into your Arduino project.

7. **Build & Flash**: Compile in Arduino IDE and upload to Portenta H7.

8. **Test & Debug**: Use Serial Monitor or debugger to check results → adjust if needed.

*In short*: 
Data → Train in Edge Impulse → Export Arduino Library → Build project in Arduino IDE → Upload to H7 → Run & Debug.

# Bird Species Recognition with MATLAB + FFT + Arduino Portenta H7

## 1️) Goal

Recognize bird species from their chirps using frequency analysis (FFT) on Arduino Portenta H7 dual-core MCU:

- **M4 core** → Low-power audio capture & preprocessing  
- **M7 core** → Main classification  

Optional: Later switch to **CMSIS-NN** for neural-network-based classification.  

> **Note:** Frequency-based classification is simpler for a small dataset. Neural networks are better for large datasets or noisy environments.

---

## 2️) Hardware Details

| Component          | Notes / Tips |
|-------------------|--------------|
| Arduino Portenta H7 | Dual-core MCU, up to 480 MHz (M7). Can handle FFT on M7 or smaller FFTs on M4. |
| INMP441 I²S mic     | Digital MEMS mic. Connect to I²S pins on Portenta. Works directly with M4 core. |
| USB cable           | For flashing code & Serial Monitor output. |
| Optional: SD card   | Can store raw audio or classification logs. Useful for debugging or large datasets. |
| Speaker / headphones| For testing playback of chirps. |

---

## 3️) Software / Libraries

| Software / Library | Purpose | Notes |
|------------------|--------|------|
| MATLAB | Record, visualize, FFT, spectrogram, frequency extraction | You can simulate bird chirps using `chirp()` function instead of recording all birds initially. |
| Arduino IDE | Develop and flash dual-core firmware | Must have Arduino **mbed_portenta** core installed. |
| CMSIS-DSP library | FFT and magnitude calculations on MCU | Used for real-time frequency analysis. Compatible with both M4 and M7 cores. |
| CMSIS-NN (optional) | Neural network inference | Only needed for ML-based classification instead of frequency rules. |
| Python (optional) | Data preprocessing or batch analysis | Can preprocess audio, generate datasets, or visualize frequency features before porting to Arduino. |

> **Notes:**  
> - No need to install full TensorFlow on Arduino—it’s too large.  
> - TFLite Micro exists for microcontrollers, but **CMSIS-DSP** is sufficient for FFT-based projects.

---

## 4️) Workflow Step-by-Step

### Step 1: Collect Bird Chirps
- Record 2–3 species first (avoid 100+ for initial testing).  
- Save in `.wav` format.  
- Optionally, generate synthetic chirps with MATLAB:

```matlab
Fs = 44100; % Sample rate
t = 0:1/Fs:0.5; % 0.5s chirp
y = chirp(t, 1000, 0.5, 2000); % Frequency from 1kHz → 2kHz
audiowrite('birdA.wav', y, Fs);
**Tips:**  
- Keep recordings in a quiet environment.  
- Short chirps (0.3–0.5s) are enough for FFT analysis.
```
---

### Step 2: MATLAB Analysis

- Load and visualize audio:
  
  ```matlab
  [y, Fs] = audioread('bird1.wav');
  plot(y); % Visualize waveform
  spectrogram(y, 256, 200, 512, Fs, 'yaxis'); % View time-frequency
  ```
- Perform FFT:

  ```matlab
  Y = fft(y);
  f = (0:length(Y)-1)*Fs/length(Y);
  plot(f, abs(Y)); % visualize frequency spectrum
  ```
- Peaks in the FFT show dominant frequencies.

- Example: Bird A ~2 kHz; Bird B ~4.5 kHz.

- Save these frequency ranges for classification thresholds.

### Step 3: Build MATLAB Classifier

- Frequency-threshold logic:
  ```matlab
  if maxFreq >= 1.8e3 && maxFreq <= 2.2e3
      class = "Bird A";
  elseif maxFreq >= 4.3e3 && maxFreq <= 4.7e3
      class = "Bird B";
  else
      class = "Unknown";
  end
  ```
- Test on multiple chirps to avoid misclassification.

- FFT resolution depends on NFFT (1024 points typical). Larger NFFT → better frequency resolution but slower processing.

### Step 4: Port to Arduino (M4 + M7)
**M4 Core**
- Purpose: Audio capture + pre-processing
- Initialize INMP441 using I²S.
- Capture audio in buffer (e.g., 1024 samples).
- Run FFT using CMSIS-DSP (optional: small FFT).
- Send snippet to M7 only if bird chirp detected.
  ```cp
  void setupM4() {
    // Initialize I2S mic
  }
  
  void loopM4() {
    captureAudio(buffer);
    if(detectChirp(buffer)) {
      sendToM7(buffer);
    }
  }
  
  ```
- M4 reduces workload for M7.

- Detection can be simple: energy > threshold, or peak in certain frequency band.

**M7 Core**

- Purpose: Main classification

- Receive audio snippet from M4.

- Run FFT with CMSIS-DSP:
 ```cp
arm_rfft_fast_instance_f32 S;
arm_rfft_fast_init_f32(&S, 1024);
arm_rfft_fast_f32(&S, input_buffer, fft_output, 0);
arm_cmplx_mag_f32(fft_output, magnitudes, 512);

int maxIndex;
float maxValue;
arm_max_f32(magnitudes, 512, &maxValue, &maxIndex);
float peakFreq = (maxIndex * sampleRate) / 1024;
```

- Apply classification rules from MATLAB.

- Output detected bird:
```cp
Serial.println(birdName);
```
- M7 can handle heavier computations (larger FFT, neural network later).

- Optional filtering: moving average, bandpass filter for improved classification.

### Step 5: Test & Validate

- Play back chirps via speaker.

- Arduino should detect & classify in real-time.

- Observe Serial Monitor output: "Bird A detected".

- Start with 2–3 bird species, then scale to more.

- Tune buffer size, FFT points, threshold values for best accuracy.

- Optional: store results on SD card for offline analysis.

---

## 5️) Optional Upgrades

- CMSIS-NN: Small neural network for species with overlapping frequency bands.

- Python / MATLAB batch analysis: Preprocess hundreds of chirps, compute FFT features, generate thresholds, or train NN model.

- Display / Network output: Show detected bird species on a small screen or send via WiFi/Bluetooth.

---

## 6️) Notes & Tips

- FFT resolution: NFFT affects frequency resolution: freq_res = sampleRate / NFFT. Choose wisely.

- Sampling rate: INMP441 supports up to 44.1 kHz; 16 kHz is sufficient for most bird chirps.

- Dual-core usage: Keep M4 simple, M7 handles heavy computation.

- Start small: Record a few chirps, test end-to-end, then scale.

- MATLAB → Arduino logic: Port key logic only (thresholds, peak detection), not full MATLAB code.
