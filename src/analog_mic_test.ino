const int microphonePin = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int mn = 1024;
  int mx = 0;

  // take 1000 samples (10k is a bit heavy for loop delay)
  for (int i = 0; i < 10000; i++) {
    int val = analogRead(microphonePin);
    mn = min(mn, val);
    mx = max(mx, val);
  }

  int delta = mx - mn;

  Serial.print("Min = ");
  Serial.print(mn);
  Serial.print("\tMax = ");
  Serial.print(mx);
  Serial.print("\tDelta = ");
  Serial.println(delta);
}
