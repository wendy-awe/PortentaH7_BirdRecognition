NAME			: NG CHU WEN 

MATRIC		: A23KE0455

SUPERVISOR		: ASSOC PROF MUHAMMAD MUN’IM BIN AHMAD ZABIDI 

---

# Objectives
- Implement a bird sound detection system using Portenta H7 + STM32Cube-AI / Arduino IDE.
- Set up GitHub repository and local development environment for the project.
- Test onboard LEDs and microphone input on Portenta H7.
- Collect and review datasheets and documentation for Portenta H7 and peripherals.
- Plan integration of ML workflow using Edge Impulse, CMSIS-DSP, and CMSIS-NN.
- Prepare workflow for audio capture, FFT processing, and ML inference on the microcontroller.
  
---

## Day 1 – Environment Setup & Documentation

1. **Created GitHub repository PortentaH7_BirdRecognition, installed Git, and cloned the repository locally.**
   
2. **Organised folder structure:**
    - docs/ – planning, research, datasheets, weekly/day reports.
    - models/ – ML model files (.eim, .tflite), converted C headers from Cube.AI.
    - src/ – firmware, C/C++ code, driver integration (GPIO, UART, camera, mic).
    - tests/ – validation scripts (Python/C), hardware tests, ML accuracy checks.
   
3. **Collected and reviewed datasheets and documentation:**
     - Portenta H7 family datasheet – memory, connectivity, security.
     - Portenta H7 reference manual – STM32H747XI dual-core specs, memory architecture, peripherals, pinouts.
   
4. **Learned Git workflow: clone, edit locally, push updates, verify on GitHub**

## Day 2 – Arduino IDE Setup & LED Blink Test

1. **Installed the latest Arduino IDE and added the Portenta Mbed OS Board Core.**
   
2. **Configured IDE: selected Portenta H7, Main Core (Cortex-M7, 480 MHz), and the correct USB port.**
   
3. **Blinked onboard blue LED (LEDB) successfully.**
   
4. **Attempted DFU mode to flash via STM32CubeProgrammer:**
    - DFU not detected; driver issue observed.
    - Confirmed ST-Link required for flashing/debugging as DFU failed.
    - Purchase ST-Link online
     
5. **Communication with supervisor:**
    - Decided to continue development in the Arduino IDE first before the ST-Link arrives by week 2.
    - Supervisor provided guidance for next steps, which are to integrate CMSIS-DSP and CMSIS-NN, and use FFT/inverse FFT on audio samples.

## Day 3 – Microphone Survey & Workflow Planning
1. **Decided to use I2S digital microphone (INMP441) for the project.**
   
2. **Outlined hardware and pin requirements for connecting the microphone to Portenta H7.**
 
3. **Planned end-to-end setup: audio capture, data processing, and potential logging.**
 
4. **Studied how to link CMSIS-DSP and CMSIS-NN libraries in Arduino IDE for audio processing and ML inference.**
 
5. **Explored approach to use FFT for frequency analysis of captured audio to prepare for bird sound recognition.**

## Day 4 – Switching from CubeIDE to Arduino IDE
1. **Facing issue of needing breakout board or soldering for flashing and debugging purpose.**
   
2. **After discussion with supervisor, Arduino IDE is suggested as breakout board is out of budget while soldering is too risky for Portenta H7 board.**

3. **Relisted steps to do project and microphone testing using Arduino IDE**

4. **Successfully added CMSIS-DSP and CMSIS-NN libraries to Arduino IDE and verified them with a minimal demo on the Portenta H7.**
    - CMSIS-DSP : Navigate to <Board Manager> , type "CMSIS-DSP" and install.
    - CMSIS-NN : Go to [https://github.com/ARM-software/CMSIS-NN.git], navigate to Code, Download zip, open the file, find file Include and Source , and copy them . Go to Document, Arduino, and go to Library, create new file named "CMSIS-NN", and paste <nclude and Source into the file. Refresh Arduino by restarting it, do code demo to make sure library CMSIS-NN is added. If not detected, copy these into the same folder as the .ino: arm_fully_connected_s8.c, arm_nn_vec_mat_mult_t_s8.c, arm_nntables.c.

## Day 5 – Deep Learning Basics & Python Set Up
1. **Completed Module 1 – Deep Learning Basics (theory + coding exercises).**
    - Learned differences between traditional programming, ML, and DL.
    - Understood neural network structure, activation functions (ReLU, sigmoid, tanh), and CNNs.
    - Studied TinyML advantages: low power, privacy, latency, and cost efficiency.
    - Ran first neural network on MNIST in Google Colab.

2. **Did Module 1 exercises and reflection.**
    - Changed neurons, activations, and training epochs to observe accuracy trade-offs.
    - Reflected on model complexity vs. training speed and overfitting.
    - Related trade-offs to microcontrollers: smaller models = faster + efficient, but less accurate; larger models = higher accuracy but more resource use.

3. **Compiled notes and reflections into a study log for future reference and supervisor updates.**

4. **Set up Python environment with TensorFlow and NumPy (Python 3.11.9).**
    - Installed required packages via Command Prompt (pip install tensorflow numpy).
    
## Day 6 – Model Conversion & Portenta Deployment
1. **Trained a simple CNN on MNIST dataset for digit classification.**
    - Achieved ~97–98% accuracy after several epochs.
    - Exported trained model as mnist_portenta.tflite for microcontroller deployment.

2. **Quantized model to INT8 for compatibility with Portenta H7.**
    - Verified quantization parameters (scale, zero_point) for generating test digits.
    - Ensured values matched the TFLite model to maintain accuracy.

3. **Generated representative test samples (digits 1–5) and saved them as digits.h.**
    - Quantized pixel values to int8 format for direct use in Arduino inference.
  
4. **Installed TensorFlow Lite Micro library for Arduino Portenta H7**
    - Add both header files (model_data.h and digits.h) into the Arduino sketch folder.

5. **Wrote Arduino sketch to initialize TensorFlow Lite interpreter, allocate tensors, and perform inference on the test digits.**

6. **Successfully uploaded the sketch to the Portenta H7 and confirmed inference results in Serial Monitor:**
    - Model initialized and ran inference correctly.
    - Predicted digit matched test input (e.g., digit1 → 1, digit2 → 2).
    - Confidence values were consistently ~0.9961 due to quantization saturation, but predictions were accurate.
