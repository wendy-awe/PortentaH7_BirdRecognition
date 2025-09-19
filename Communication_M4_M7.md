# Communication between M4 & M7 on Portenta H7

This project demonstrates a simple communication setup between the two cores (Cortex-M7 and Cortex-M4) of the Arduino Portenta H7 using the `RPC` library.  
The goal was to test whether messages can be passed reliably between cores, as a foundation for more complex tasks (e.g., bird activity detection on M4 and species classification on M7).

---

## Setup

- **Board**: Arduino Portenta H7  
- **IDE**: Arduino IDE  
- **Core Split**: `1MB M7 + 1MB M4`  
- **Serial Monitor**: Connected to M7 (Main Core)  

---

## Code

### M7 (Main Core)

```cpp
#include "RPC.h"

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("[M7] Ready, waiting for M4...");

  // Bind function to handle messages from M4
  RPC.bind("fromM4", [](String msg) {
    Serial.print("[M7] Got from M4: ");
    Serial.println(msg);
  });
}

void loop() {
  // Nothing to do here, just wait for M4 messages
  delay(1000);
}
```

### M4 (Communication Core)

```cpp
#include "RPC.h"

void setup() {
  delay(1000); // Give M7 time to boot
  RPC.begin();
  RPC.println("Hello from M4!"); // First greeting
}

void loop() {
  // Send periodic test messages
  RPC.call("fromM4", "Ping from M4...");
  delay(1000);
}
```

## Procedure

1. Prepare Split Flash:

    - Tools → Flash split → Select 1MB M7 + 1MB M4

2. Upload to M4:

    - Tools → Target core → CM4
  
    - Upload M4 sketch

3. Upload to M7:

    - Tools → Target core → Main Core (CM7)
    
    - Upload M7 sketch

4. Open Serial Monitor (baud 115200).

## Results

Serial Monitor output:

```cp
[M7] Ready, waiting for M4...
[M7] Got from M4: Hello from M4!
[M7] Got from M4: Ping from M4...
```

- The first line confirms M7 is initialized.

- The next lines show successful reception of messages sent from M4.

   Communication between M4 and M7 cores is working.

## Explanation

- RPC (Remote Procedure Call) is used as the communication layer.

- M4 and M7 can exchange strings, numbers, or more complex data structures.

- In this test, only basic strings were exchanged.

- This basic setup verifies the dual-core communication channel is functioning.

## Next Steps

- Replace "Ping from M4..." with bird activity detection results (from BAD model).

- Let M7 process classification when activity is detected.

- Extend communication protocol (e.g., M4 sends status codes, M7 responds with species names).
