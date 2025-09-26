# Portenta H7 Dual-Core Shared Memory Communication

## Principle
- The Portenta H7 has a dual-core architecture:

    - M7 (Cortex-M7): main core, handles main logic (e.g., writing data / triggering tasks).
    
    - M4 (Cortex-M4): secondary core, handles co-processing (e.g., receiving data / performing computation).

- The two cores communicate using shared memory (SRAM4) plus synchronisation flags.
  
- Data is stored in shared memory, while flags coordinate read/write operations to avoid conflicts.

## Shared Memory Base Address
Select a region both M7 and M4 can access:
```cpp
#define SHARED_BASE 0x38001000  // Start of SRAM4
```

## Shared Memory & Flags
- Instead of using extern/volatile variables (which can cause multiple-definition errors), we map directly to memory addresses using macro.
- **SharedMemory.h**:
```cpp
// SharedMemory.h
#pragma once

#define SHARED_BASE   0x38001000

// Data storage (integer)
#define sharedData   (*(volatile int*)(SHARED_BASE + 0))

// Flags
#define flagM7toM4   (*(volatile bool*)(SHARED_BASE + 4)) // M7 → M4
#define flagM4toM7   (*(volatile bool*)(SHARED_BASE + 8)) // M4 → M7
```

## M7 Logic (BirdDualCore_M7.ino)
 M7 writes data, sets flagM7toM4 = true, then waits until M4 replies with flagM4toM7 = true.
 ```cpp
#include "SharedMemory.h"

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("M7 core started");

  // Reset shared memory
  sharedData = 0;
  flagM7toM4 = false;
  flagM4toM7 = false;

  // Start M4
  bootM4();
}

void loop() {
  delay(1000);

  // Write data to M4
  sharedData++;
  Serial.print("M7 wrote: ");
  Serial.println(sharedData);

  flagM7toM4 = true;  // notify M4

  // Wait for M4 reply
  while (!flagM4toM7) {
    delay(1);
  }

  Serial.print("M7 got reply from M4: ");
  Serial.println(sharedData);

  flagM4toM7 = false;  // clear flag
}
```

## M4 Logic (BirdDualCore_M4.ino)
 M4 waits for flagM7toM4 = true, reads data, processes it (×2), writes back to shared memory, then sets flagM4toM7 = true.
 ```cpp
#include "SharedMemory.h"

void setup() {
  Serial.begin(115200);
  Serial.println("M4 core started");

  // Reset shared memory
  sharedData = 0;
  flagM7toM4 = false;
  flagM4toM7 = false;
}

void loop() {
  if (flagM7toM4) {
    int val = sharedData;

    // Process (×2)
    sharedData = val * 2;

    Serial.print("M4 processed: ");
    Serial.println(sharedData);

    flagM7toM4 = false;  // clear
    flagM4toM7 = true;   // reply
  }
}
```
## Upload Steps
1. Create BirdDualCore_M7.ino
2. Create BirdDualCore_M4.ino
3. Locate BirdDualCore_M7.ino folder & BirdDualCore_M4.ino folder in the same folder.
4. Create SharedMemory.h & locate it in both BirdDualCore_M7.ino and BirdDualCore_M4.ino respectively.
5. Upload BirdDualCore_M7.ino, then only upload BirdDualCore_M4.ino
   
## Output
```yaml
M7 core started
M7 wrote: 1
M7 got reply from M4: 2
M7 wrote: 3
M7 got reply from M4: 6
M7 wrote: 7
M7 got reply from M4: 14
M7 wrote: 15
M7 got reply from M4: 30
```

- Use memory-mapped macros, not global variables.

- Use two flags (M7→M4 and M4→M7) to avoid race conditions.

- Place shared memory in SRAM4 (accessible by both cores).

- Always clear flags in the loop, otherwise execution stalls.
