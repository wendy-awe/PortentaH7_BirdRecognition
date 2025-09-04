# Portenta H7 + I²S Microphone (INMP441) — End-to-End Setup 

This README.md is a **practical, step-by-step guide** to connect, detect, configure, and test a digital I²S microphone (**INMP441**) on the **Arduino Portenta H7 (STM32H747XI)**.  
It walks you through **board detection (DFU & ST-LINK), CubeIDE project creation, I²S + clocks + DMA setup, a minimal M4 test (DMA + LED on sound), flashing, and troubleshooting**, updated for **header pin connections with female-to-female jumpers**.  

---

## Table of Contents

1. Hardware & Software Needed  
2. Connect & Detect the Board (DFU & ST-LINK)  
3. Wire the Microphone (INMP441 — I²S via headers)  
4. Create the STM32CubeIDE Project  
5. Configure Peripherals (I²S, Clocks, DMA, GPIO, UART)  
6. Generate Code  
7. Minimal M4 Mic Test (DMA + LED on Sound)  
8. Build, Flash, and Run  
9. Verify and Tune  
10. Further: Move Towards Frequency Detection / Bird Calls  
11. Common Pitfalls Checklist  

---

## 1) Hardware & Software Needed

* **Board:** Arduino Portenta H7 (STM32H747XI MCU) with headers.  
* **Mic:** Digital I²S MEMS mic (**INMP441**) with header pins.  
* **Programmer:**

  * **Preferred:** ST-LINK (ST-LINK V3-MINI recommended; ST-LINK V2 clone often works).  
  * **Alternative:** USB DFU via Portenta's USB‑C (no external programmer required, but limited).  
* **Cables:** USB‑C data cable, female-to-female jumper wires (0.1").  
* **Optional:** Oscilloscope or USB serial monitor for debugging signals.  

 **Note:** Breakout boards are optional. You can directly use jumper wires if **SAI and SWD pins are accessible on headers**.

---

## 2) Connect & Detect the Board

### Option A — USB DFU (no ST-LINK)

1. Plug Portenta H7 into PC using a **data-capable USB-C cable**.  
2. Double-tap the **RESET button** to enter DFU mode.  
3. On Windows Device Manager, check for **"STM32 BOOTLOADER"**.  
4. Open **STM32CubeProgrammer → USB → Refresh → Connect**.  

- DFU works for flashing but **cannot do live debugging**. Prefer ST-LINK for development.

#### [Option A Failed]  

### Option B — ST-LINK SWD (recommended)

1. Connect **ST-LINK** to Portenta headers (jumper wires or breakout):  

| ST-LINK | Portenta Header Pin | Notes |
|---------|-------------------|-------|
| SWDIO   | SWDIO              | Data line |
| SWCLK   | SWCLK              | Clock line |
| NRST    | NRST               | Reset line |
| GND     | GND                | Ground |
| 3V3     | Vtarget            | Sense target voltage |

2. Connect ST-LINK to PC via USB.  
3. Open **STM32CubeProgrammer → ST-LINK → Connect**.  

> If ST-LINK connects, CubeIDE can debug with breakpoints. If not, DFU is the fallback.  

---

## 3) Wire the Microphone (INMP441, I²S via headers)

Use **female-to-female jumper wires**:

| Mic Pin        | Portenta Pin (I²S2 / SAI2_A) | Notes |
| -------------- | ---------------------------- | ----- |
| VDD            | 3.3 V                        | Confirm mic voltage |
| GND            | GND                          | Ground |
| SD (DATA)      | PC3 / I²S2_SD                | I²S data input |
| SCK (BCLK)     | PI1 / I²S2_CK                | Bit clock (MCU provides Master) |
| WS / LR (LRCK) | P10 / I²S2_WS                | Word/frame select |
| L/R            | GND (Left) or 3.3V (Right)  | Single channel selection |

- Ensure pins **P10, PI1, PC3** are accessible on headers. Otherwise, solder to high-density connector.

---

## 4) Create the STM32CubeIDE Project

1. **File → New → STM32 Project**.  
2. MCU/MPU selector → **STM32H747XIHx** → Next.  
3. Project Manager → **Advanced Settings** → enable **Cortex-M4 project** (M4 handles mic capture).  
4. Finish → CubeMX configurator opens.  

> M7 core can be added later for ML / bird call classification.

---

## 5) Configure Peripherals (CubeMX)

### 5.1 I²S (SAI/I²S2) Receiver

* Peripherals → **SAI2 / I²S2**  
  - Mode: **I²S Master Receiver**  
  - Standard: **I²S Philips**  
  - Data size: 16-bit or 24-bit (INMP441 outputs 24-bit, can capture as 32-bit)  
  - Frame length: 32 bits  

* Map pins:  
  - WS → P10  
  - CK → PI1  
  - SD → PC3  

### 5.2 Clocks

* Clock tab → Configure **PLLI2S** or PLL2/3 to generate I²S clock.  
* Sample rate: **16 kHz** (test) or **48 kHz** (preferred for audio).  

### 5.3 DMA for I²S RX

* DMA request → RX (Circular mode)  
* Memory increment → Enabled  
* Peripheral increment → Disabled  
* Data width → Match SAI/I²S data size (16-bit or 32-bit)  
* FIFO → Off (keep it simple)  

### 5.4 GPIO LED

* Configure **accessible GPIO** as **Output Push-Pull**, low speed.  

### 5.5 UART (optional)

* Enable **USART/LPUART** for printf debugging  
* Settings: 115200 8-N-1  

---

## 6) Generate Code

* Project → Generate Code (gear icon)  
* CubeIDE generates:  
  - `MX_I2S2_Init()`  
  - `MX_DMA_Init()`  
  - `MX_GPIO_Init()`  
  - UART init if enabled  

---

## 7) Minimal M4 Mic Test (DMA + LED)

Paste into **`main.c`** (user code region):

```c
#include "main.h"
#include <stdlib.h>

extern I2S_HandleTypeDef hi2s2;

#define AUDIO_SAMPLES 1024
static int16_t audio_buffer[AUDIO_SAMPLES];
static volatile uint8_t buffer_ready = 0;

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s) {
    if (hi2s->Instance == SPI2) { // I²S2 uses SPI2
        buffer_ready = 1;
    }
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_I2S2_Init();

    HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)audio_buffer, AUDIO_SAMPLES);

    while (1) {
        if (buffer_ready) {
            buffer_ready = 0;

            uint32_t sum = 0;
            for (int i = 0; i < AUDIO_SAMPLES; i++) sum += abs(audio_buffer[i]);

            if (sum > 50000) { // adjust threshold
                HAL_GPIO_TogglePin(GPIOx, GPIO_PIN_y); // LED
            }

            // Optional debug
            // printf("sum=%lu, sample0=%d\r\n", sum, audio_buffer[0]);
        }
    }
}
```

## 8) Build, Flash, and Run

### Using ST-LINK (recommended)

1. Debug → M4 project → ST-LINK  
2. Interface → **SWD**, Frequency 2–4 MHz  
3. Reset → **Connect under reset**  
4. Debug → CubeIDE flashes → Resume (F8)  

### Using DFU (optional)

1. Double-tap **RESET** → DFU mode  
2. STM32CubeProgrammer → USB → Connect  
3. Download `.hex` or `.elf` → Reset  

---

## 9) Verify and Tune

* Clap or speak → LED should toggle  
* If not:  
  - Check wiring (`SD`, `SCK`, `WS`, `VDD`, `GND`)  
  - Confirm **I²S Master** configuration  
  - Lower threshold (e.g., `20000`)  
  - Inspect signals with oscilloscope / logic analyzer  
  - Try different USB cable / port  

---

## 10) Further: Frequency Detection / Bird Calls

* Apply **CMSIS-DSP FFT** on DMA buffer  
* Detect dominant frequency → toggle LED if in bird band (2–8 kHz)  
* Compute spectrogram → send to **M7 core** for ML inference  

---

## 11) Common Pitfalls

* Wrong pins (`SD`/`SCK`/`WS`) must map to **I²S2 / SAI2** pins  
* Mic voltage incorrect (must be **3.3 V**)  
* I²S not set to **Master** (MCU must supply clock)  
* DMA not **Circular** → streaming stops  
* Debug via DFU → no breakpoints / SWO → use **ST-LINK**  
* Headers hard to reach → breakout optional  

---

## Summary

1. Wire **INMP441 → Portenta** via headers & female jumpers  
2. Configure **I²S2 + DMA** in CubeIDE  
3. Generate code, start DMA reception  
4. LED or UART debug → values should change when sound is present  
5. Once confirmed → move to **FFT & ML** for bird call detection  
