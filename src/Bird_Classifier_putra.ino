#include "Arduino.h"

#ifdef abs
#undef abs
#endif
#include <cmath>

// TensorFlow Lite Micro headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your converted model and sample Mel input
#include "corre20_putramodel.h"                // quantized TFLite model
#include "sample_mel_Crested_serpent_Eagle.h"  // generated sample input Mel spectrogram (96x96x3)

// -------------------------
// Bird names
// -------------------------
const char* bird_names[] = {
  "Asian Koel",
  "Black-naped Oriole",
  "Cinereous Tit",
  "Collared Kingfisher",
  "Common Lora",
  "Crested-serpent Eagle",
  "Large-tailed Nightjar",
  "Original",
  "Pied Fantail",
  "Spotted Dove",
  "Zebra Dove"
};

// -------------------------
// Globals
// -------------------------
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(corre20_putramodel);
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena
  constexpr int kTensorArenaSize = 200 * 1024;
  __attribute__((section(".ext_ram"))) uint8_t tensor_arena[kTensorArenaSize];
}

// -------------------------
// Setup
// -------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ Portenta H7 Putra Bird Classifier ready!");
}

// -------------------------
// Loop
// -------------------------
void loop() {
  // Fill input tensor
  for (int i = 0; i < input->bytes; i++) {
    input->data.int8[i] = sample_mel_Crested_serpent_Eagle[i];
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Dequantize output and apply softmax
  int output_size = output->dims->data[output->dims->size - 1];
  float dequant[output_size];
  float max_val = -1000.0;

  for (int i = 0; i < output_size; i++) {
    dequant[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    if (dequant[i] > max_val) max_val = dequant[i];
  }

  float sum_exp = 0.0;
  for (int i = 0; i < output_size; i++) {
    dequant[i] = exp(dequant[i] - max_val);
    sum_exp += dequant[i];
  }
  for (int i = 0; i < output_size; i++) dequant[i] /= sum_exp;

  // Get top-3 predictions
  int top_idx[3] = {0, 0, 0};
  float top_prob[3] = {0.0, 0.0, 0.0};

  for (int i = 0; i < output_size; i++) {
    if (dequant[i] > top_prob[0]) {
      top_prob[2] = top_prob[1]; top_idx[2] = top_idx[1];
      top_prob[1] = top_prob[0]; top_idx[1] = top_idx[0];
      top_prob[0] = dequant[i]; top_idx[0] = i;
    } else if (dequant[i] > top_prob[1]) {
      top_prob[2] = top_prob[1]; top_idx[2] = top_idx[1];
      top_prob[1] = dequant[i]; top_idx[1] = i;
    } else if (dequant[i] > top_prob[2]) {
      top_prob[2] = dequant[i]; top_idx[2] = i;
    }
  }

  // Print results
  Serial.println("Top 3 predictions:");
  for (int i = 0; i < 3; i++) {
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(bird_names[top_idx[i]]);
    Serial.print(" - ");
    Serial.println(top_prob[i], 4);
  }

  delay(2000);
}
