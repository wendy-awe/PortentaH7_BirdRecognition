#include "Arduino.h"

#ifdef abs
#undef abs
#endif

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "BAD_tiny_model.h" // Quantized INT8 model

// -------------------- Mic Settings --------------------
const int micPin = A0;
const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = SAMPLE_RATE * 1; // 1-second buffer
int16_t audioBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// -------------------- TensorFlow Lite --------------------
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(bad_tiny_model);
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 150 * 1024;
  __attribute__((section(".ext_ram"))) uint8_t tensor_arena[kTensorArenaSize];
}

// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  while (!Serial) { ; }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ M7 INT8 TinyChirp detection ready!");
}

// -------------------- Helper: Simple Mel Approximation --------------------
void computeMel(int16_t* audio, int length, float* melOut, int melBins = 16, int frames = 96) {
  int hop = length / frames;
  for (int f = 0; f < frames; f++) {
    for (int m = 0; m < melBins; m++) {
      float sum = 0;
      int startIdx = f * hop + m * (hop / melBins);
      for (int j = startIdx; j < startIdx + hop / melBins; j++) {
        if (j < length) {
          float s = audio[j] / 32768.0f; // [-1,1]
          sum += s * s;
        }
      }
      melOut[f * melBins + m] = sqrt(sum / (hop / melBins));
    }
  }
}

// -------------------- Loop --------------------
void loop() {
  // --- 1. Sample audio ---
  for (int i = 0; i < BUFFER_SIZE; i++) {
    int val = analogRead(micPin);
    float sample = ((float)val - 512.0f) / 512.0f;
    audioBuffer[bufferIndex++] = (int16_t)(sample * 32767);
    if (bufferIndex >= BUFFER_SIZE) bufferIndex = 0;
  }

  // --- 2. Compute Mel spectrogram ---
  float mel[96 * 16];
  computeMel(audioBuffer, BUFFER_SIZE, mel, 16, 96);

  // --- 3. Quantize and fill input tensor ---
  for (int i = 0; i < 96*16; i++) {
    int8_t q = (int8_t)(mel[i] / input->params.scale + input->params.zero_point);
    input->data.int8[i] = q;
  }

  // --- 4. Run inference ---
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // --- 5. Check output with correct softmax interpretation ---
  int8_t class0 = output->data.int8[0]; // non-bird
  int8_t class1 = output->data.int8[1]; // bird

  float prob0 = (class0 - output->params.zero_point) * output->params.scale;
  float prob1 = (class1 - output->params.zero_point) * output->params.scale;

  // pick the class with higher probability
  int detection = (prob1 > prob0) ? 1 : 0;

  // --- 6. Temporal smoothing ---
  static int lastDetections[3] = {0,0,0};
  lastDetections[0] = lastDetections[1];
  lastDetections[1] = lastDetections[2];
  lastDetections[2] = detection;
  int sumDetections = lastDetections[0] + lastDetections[1] + lastDetections[2];

  if (sumDetections >= 3) {
    digitalWrite(LED_BUILTIN, LOW);  // LED ON when bird detected
    Serial.println(1);
  } else {
    digitalWrite(LED_BUILTIN, HIGH);   // LED OFF otherwise
    Serial.println(0);
  }
}
