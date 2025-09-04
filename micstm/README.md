# Portenta H7 + I²S Microphone (INMP441) — End-to-End Setup

This README.md is a step-by-step, practical guide to connect, detect, configure, and test a digital I²S microphone (example: **INMP441**) on the **Arduino Portenta H7 (STM32H747XI)**. It walks you through board detection (DFU & ST-LINK), CubeIDE project creation, SAI/I²S + clocks + DMA setup, a minimal M4 test (DMA + LED on sound), flashing, and troubleshooting.

---

## Table of Contents

1. What you need
2. Detect the board (DFU & ST-LINK)
3. Wire the microphone (INMP441 — I²S)
4. Create the right project in STM32CubeIDE
5. Configure peripherals (SAI/I²S, Clocks, DMA, GPIO, UART)
6. Generate code
7. Minimal M4 mic test (DMA + LED on sound)
8. Build, flash, and run
9. Verify and tune
10. Optional: move towards frequency detection / birds
11. Quick glossary
12. Common pitfalls checklist

---

## 1) What you need

* **Board:** Arduino Portenta H7 (STM32H747XI MCU).
* **Mic:** Digital I²S MEMS mic (suggested: **INMP441** module).
* **Programmer (choose one):**

  * **Preferred:** ST-LINK (ST-LINK V3-MINI recommended; ST-LINK V2 clone often works).
  * **Alternative:** USB DFU via Portenta's USB‑C (no external programmer required).
* **Optional:** Breakout board for Portenta (to access SAI / SWD pins easily).
* **Cables:** Good USB‑C data cable, 3.3 V jumper wires.

---

## 2) Connect & “Detect the board”

### Option A — USB DFU (no ST-LINK)

1. Plug Portenta H7 into your PC using a data-capable USB-C cable.
2. Enter DFU mode: **double‑tap the RESET button quickly**.
3. On Windows Device Manager you should see **"STM32 BOOTLOADER"** (or similar).
4. Open **STM32CubeProgrammer** → select **USB** → click **Refresh** → **Connect**.

> If it does not appear: try a different cable/port, install the STM32 USB (WinUSB) driver, and repeat the double-tap.

**Note:** DFU is good for flashing, but you cannot live-debug with breakpoints. For debugging use ST‑LINK.

### Option B — ST-LINK SWD (recommended)

1. Connect your **ST-LINK** to the Portenta (use a breakout board if available):

   * `SWDIO -> SWDIO`
   * `SWCLK -> SWCLK`
   * `NRST -> NRST`
   * `GND -> GND`
   * `3V3 -> Vtarget` (so ST-LINK can sense the target voltage)
2. Connect ST-LINK to PC via USB.
3. Open **STM32CubeProgrammer** → choose **ST-LINK** → **Connect**.

> If ST-LINK connects in the programmer, CubeIDE will also be able to debug.

**Tip:** If you don’t have a breakout board, DFU may be simpler because accessing SWD pins on the Portenta headers is fiddly.

---

## 3) Wire the microphone (INMP441, I²S)

Wire the microphone module to the Portenta using the SAI1 Block A pins (or equivalent I²S pins on your breakout):

| Mic pin        | Portenta (SAI1 Block A) | Notes                                        |
| -------------- | ----------------------: | -------------------------------------------- |
| VDD            |                   3.3 V | Power (confirm mic voltage)                  |
| GND            |                     GND | Ground                                       |
| SD (DATA)      |             `SAI1_SD_A` | I²S data input                               |
| SCK (BCLK)     |            `SAI1_SCK_A` | Bit clock — MCU provides this in Master mode |
| WS / LR (LRCK) |             `SAI1_FS_A` | Word/frame select (frame sync)               |

> Many INMP441 modules output on a single channel; if the module has an L/R select pin, follow the datasheet.

**Important:** The exact physical pin numbers depend on your breakout/carrier. Use the Portenta H7 pinout and map to **SAI1 Block A** pins.

---

## 4) Create the right project in STM32CubeIDE

1. **File → New → STM32 Project**.
2. In the MCU/MPU selector search for **STM32H747XIHx** → Next.
3. In Project Manager → **Advanced Settings**: enable the **Cortex‑M4** project (you will see two projects: M7 and M4). For mic capture, we will work in the **M4** project.
4. Finish to open the CubeMX configurator.

> You can do the complete mic test in the **M4** project. Add the M7 project later for ML / bird identification.

---

## 5) Configure peripherals (CubeMX inside CubeIDE)

### 5.1 SAI (I²S) as Receiver

* Peripherals → **SAI1**

  * **Block A** → **Mode:** I²S, **Master Receiver**
  * **Protocol/Standard:** I²S Philips
  * **Data size:** 16‑bit or 24‑bit (INMP441 typically 24‑bit; capturing in 32‑bit frames is common)
  * **Frame length:** 32 bits (common)
* Confirm `SAI1_SD_A`, `SAI1_SCK_A`, `SAI1_FS_A` are mapped to accessible pins (Pinout tab).

### 5.2 Clocks

* Clock Configuration tab:

  * Set a PLL (PLL2 or PLL3) to generate a clock source for SAI1 that can produce audio sample rates (e.g., 48 kHz or 44.1 kHz).
  * For bird audio, **48 kHz** is a good default. For simple tests, **16 kHz** also works.

### 5.3 DMA for SAI1 Block A (RX)

* In **SAI1 → DMA Settings** add an RX DMA request.

  * **Mode:** Circular (continuous streaming)
  * **Memory increment:** Enabled
  * **Peripheral increment:** Disabled
  * **Data width:** Halfword (16‑bit) or Word (32‑bit) — match your SAI data size
  * **FIFO:** Off (keep it simple)

### 5.4 GPIO for LED (visual test)

* Choose an accessible GPIO pin (for example a carrier board user LED) and configure it as **GPIO Output** (Push‑pull, no pull, low speed).

### 5.5 (Optional) UART for `printf` debugging

* Enable a **USART** or **LPUART** on pins you can access with a USB‑TTL serial adapter (3.3 V).
* Set **115200 8‑N‑1**.

---

## 6) Generate code

* Click **Project → Generate Code** (or the gear icon).
* CubeIDE generates `MX_SAI1_Init()`, `MX_DMA_Init()`, `MX_GPIO_Init()`, any UART init, and other skeletons inside the **M4** project.

---

## 7) Minimal M4 mic test (DMA + LED on sound)

**Paste this into the M4 project `Core/Src/main.c` inside the user code regions** (replace `GPIOx` / `GPIO_PIN_y` with the actual LED port/pin you configured):

```c
#include "main.h"
#include <stdlib.h> // for abs

extern SAI_HandleTypeDef hsai_BlockA1;

#define AUDIO_SAMPLES 1024  // one DMA buffer "plate"
static int16_t audio_buffer[AUDIO_SAMPLES];
static volatile uint8_t buffer_ready = 0;

void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai) {
  if (hsai->Instance == SAI1_Block_A) {
    buffer_ready = 1; // DMA finished filling the buffer
  }
}

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SAI1_Init();
  // If you enabled UART: MX_USARTx_UART_Init();

  // Start streaming audio into the buffer via DMA
  HAL_SAI_Receive_DMA(&hsai_BlockA1, (uint8_t*)audio_buffer, AUDIO_SAMPLES);

  while (1) {
    if (buffer_ready) {
      buffer_ready = 0;

      // Simple "any sound?" test: sum absolute values
      uint32_t sum = 0;
      for (int i = 0; i < AUDIO_SAMPLES; i++) {
        sum += (uint32_t)abs(audio_buffer[i]);
      }

      // Basic threshold — adjust after testing in your room
      if (sum > 50000) {
        // Toggle LED to show mic activity
        HAL_GPIO_TogglePin(GPIOx, GPIO_PIN_y); // <-- replace with your LED port/pin
      }

      // Optional: print a sample or the sum for debugging
      // printf("sum=%lu, sample0=%d\r\n", sum, audio_buffer[0]);
    }
  }
}
```

**What this does:**

* DMA continuously fills `audio_buffer`.
* When the buffer is full, the `HAL_SAI_RxCpltCallback` sets `buffer_ready`.
* The main loop sums absolute sample values and toggles the LED when the sum passes a threshold.
* Clap/speak near the mic — the LED should react.

---

## 8) Build, flash, and run

### If using **ST-LINK** (recommended for debugging)

1. Click the **Debug** (bug) icon. Choose the **M4** project debug configuration.
2. In Debug Configs → ST-LINK settings:

   * Interface: **SWD**
   * Frequency: **2–4 MHz** (start safe)
   * Reset: **Connect under reset** (often safer on H7)
3. Click **Debug** — CubeIDE will flash and halt at `main()`; press **Resume** (F8) to run.

### If using **DFU**

1. Put Portenta into **DFU** (double-tap RESET).
2. Open **STM32CubeProgrammer** → USB → Connect.
3. **Open** your built M4 `.hex` or `.elf` file → **Download**.
4. Reset the board to run.

---

## 9) Verify and tune

* Clap or speak near the mic — the LED should toggle.
* If nothing happens:

  * Re-check wiring: `SD ↔ SAI1_SD_A`, `SCK ↔ SAI1_SCK_A`, `FS ↔ SAI1_FS_A`, `3.3V`, `GND`.
  * Confirm **SAI1 Block A** is configured as **I²S Master Receiver**.
  * Lower the threshold value (try `20000`) and retest.
  * Use an oscilloscope/logic analyzer to inspect `SCK` and `SD` signals.
  * Try a different USB cable/port or reinstall drivers if DFU fails.

---

## 10) (Optional) Move towards frequency detection / bird calls

1. Add **CMSIS‑DSP** FFT on the filled buffer, compute the dominant frequency for each buffer.
2. Only toggle the LED when the dominant frequency is within a bird frequency band (e.g., 3–8 kHz).
3. For species classification: compute spectrogram frames on M4 and send them to M7 (or send raw buffers) for ML inference on the M7 core.

---

## 11) Quick glossary ("sandwich" style)

* **Buffer** = the plate that holds a batch of audio samples.
* **DMA** = the conveyor belt that fills the plate from the mic without the CPU copying samples.
* **M4** = kitchen staff steadily filling plates (real-time capture).
* **M7** = the head chef doing heavy analysis (FFT/ML).

---

## 12) Common pitfalls checklist

* Wrong pins (SD/SCK/FS must be mapped to SAI1 Block A pins).
* Mic not powered with the correct voltage (most INMP441 modules are 3.3 V — confirm).
* SAI not set as **Master** (many mics expect the MCU to supply the bit clock).
* DMA not set to **Circular** mode (without circular, streaming will stop after one buffer).
* Trying to debug over DFU (breakpoints/printf via SWO not supported) — prefer ST‑LINK for development.
* No breakout board → SWD pins hard to reach. If you lack a breakout, use DFU for flashing or obtain a breakout.

---

If you want, I can also:

* Generate a ready‑to‑paste **M4 project skeleton** (main.c, stm32 init stubs) matching this setup.
* Add a **CMSIS‑DSP FFT example** and a small UART `printf` helper to view values in a serial terminal.

---

*End of README*
