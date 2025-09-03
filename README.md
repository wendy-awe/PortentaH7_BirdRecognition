# PortentaH7_BirdRecognition
Bird sound detection project using Portenta H7 + STM32Cube-AI.
This project aims to implement a bird recognition system on the Portenta H7 microcontroller.  
It is developed using STM32CubeIDE and will include source code, documentation, models, and tests.

## Repository Structure
- `src/` : Source code for firmware and application
- `docs/` : Documentation, design notes, and reports
- `models/` : Machine learning models (trained or converted for MCU deployment)
- `tests/` : Unit tests, hardware tests, and validation scripts

## Portenta H7 + STM32CubeIDE Project Flow
1. **Collect data**: Gather pictures / audio / sensor data on your laptop.

2. **Train ML model**: Use Edge Impulse (online) to train the model.

3. **Convert model to STM32 code**: Use STM32Cube.AI (inside STM32CubeMX/IDE) to change ML model → C code.

4. **Create STM32 project**: Open STM32CubeIDE, start a new project for Portenta H7 (board config).

5. **Add drivers (peripherals)**: Configure GPIO, UART, I²C, SPI, camera, mic (depending on sensors you use).

6. **Integrate ML model**: Copy the C code from Cube.AI into your STM32 project.

7. **Build & Flash**: Compile in STM32CubeIDE and upload to Portenta H7.

8. **Test & Debug**: Use serial monitor or debugger to check results → adjust if needed.

*In short*: 
Data → Train in Edge Impulse → Convert in STM32Cube.AI → Build project in STM32CubeIDE → Upload to H7 → Run & Debug.

### Week 1
- [x] Installed STM32CubeIDE
- [ ] Created Portenta H7 project
- [ ] Blink onboard LED confirmed
- [ ] UART "Hello World" working
- [x] Push to GitHub repository


