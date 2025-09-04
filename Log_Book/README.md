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


