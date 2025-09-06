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

