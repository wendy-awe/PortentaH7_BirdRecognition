# blink_portentaH7_arduino
## Steps
1. **Download and install the latest version of Arduino IDE from the official Arduino website**
  - My version is Arduino IDE 2.3.6
    
2. **Add the Portenta board core**
  - Open the Arduino IDE.
  - Navigate to Tools > Board > Boards Manager...
  - Search for "Portenta".
  - Find Arduino Mbed OS Boards and click Install.
    
3. **Select the correct board and core**
  - Connect your Portenta H7 to your computer with a USB-C cable
  - Select the core: Navigate to Tools > Target core and select Main Core 
    [Main Core refers to the Cortex-M7 core, which is the primary processor running at a faster clock speed (480 MHz). It is the correct core to select for a simple, single-core sketch like blinking an LED.]
    [M4 Co-processor refers to the Cortex-M4 core, which runs at a lower clock speed (240 MHz) and is used for specific real-time tasks in dual-core applications.]
     
4. **Ensure you have the right port**
  - Go to Tools > Port and select the port your Portenta is connected to, and the name is “Arduino Portenta H7".
    
5. **Sketch to blink the green LED**
  - The Portenta H7 has built-in RGB LEDs.Predefined constants LEDR (red), LEDG (green), and LEDB (blue) can be used.
  - The following sketch blinks the green LED once per second.
  ```
  // the setup function runs once when you press reset or power the board
  void setup() {
    // initialize digital pin LEDB as an output.
    pinMode(LEDB, OUTPUT);
  }
  
  // the loop function runs over and over again forever
  void loop() {
    digitalWrite(LEDB, HIGH);  // turn the LED on (HIGH is the voltage level)
    delay(1000);                      // wait for a second
    digitalWrite(LEDB, LOW);   // turn the LED off by making the voltage LOW
    delay(1000);                      // wait for a second
  }
  ```

6. **Click the Verify & Upload button in the top-left corner of the IDE**
  - The IDE will compile the code and upload it to your Portenta H7. Once the upload is complete, the blue LED on your Portenta H7 will begin to blink on and off   with a one-second interval.
  
