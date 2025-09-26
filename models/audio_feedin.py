import os
import numpy as np
import librosa
import tensorflow as tf

# -------------------------------
# 1️⃣ Exact WAV file path
# -------------------------------
wav_file = r"C:/Users/chuwe/AppData/Local/Programs/Python/Python311/putra_wav/Collared_Kingfisher/20200719_110000_HSKT_0_01022_310.wav"
bird_class = os.path.basename(os.path.dirname(wav_file))
wav_name = os.path.basename(wav_file)

print(f"Selected WAV file: {wav_name}")
print(f"Bird class: {bird_class}")

# -------------------------------
# 2️⃣ Convert audio to Mel-spectrogram (96x96x3)
# -------------------------------
def audio_to_mel_rgb(path, sr=44100, n_mels=96, duration=3.0, img_size=(96, 96)):
    y, sr = librosa.load(path, sr=sr, duration=duration)
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    # Convert power to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Normalize to [0,1]
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
    # Resize to 96x96 (still grayscale)
    S_resized = tf.image.resize(S_norm[np.newaxis, ..., np.newaxis], img_size).numpy()[0, ..., 0]
    # ✅ Stack grayscale to 3 channels → (96,96,3)
    S_rgb = np.stack([S_resized] * 3, axis=-1)
    return S_rgb.astype(np.float32)

mel = audio_to_mel_rgb(wav_file)
print(f"✅ Mel shape: {mel.shape}")  # should be (96,96,3)

# -------------------------------
# 3️⃣ Save as .npy
# -------------------------------
output_npy = f"sample_mel_{bird_class}.npy"
np.save(output_npy, mel)
print(f"✅ Saved sample Mel-spectrogram: {output_npy}")

# -------------------------------
# 4️⃣ Load TFLite model to read quantization params
# -------------------------------
tflite_model = "putra_model_int8_fixed.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
scale, zero_point = input_details['quantization']

print(f"Model expects input scale={scale}, zero_point={zero_point}")
print("Input shape:", input_details['shape'])

# -------------------------------
# 5️⃣ Quantize using model's params
# -------------------------------
mel_int8 = np.clip(np.round(mel / scale + zero_point), -128, 127).astype(np.int8)

print("Mel shape:", mel_int8.shape, "dtype:", mel_int8.dtype)
print("First 10 values:", mel_int8.flatten()[:10])

# -------------------------------
# 6️⃣ Save to Arduino header
# -------------------------------
header_file = f"sample_mel_{bird_class}.h"
array_name = f"sample_mel_{bird_class}".replace("-", "_")

with open(header_file, "w") as f:
    f.write(f"#ifndef SAMPLE_MEL_{bird_class.upper().replace('-', '_')}_H\n")
    f.write(f"#define SAMPLE_MEL_{bird_class.upper().replace('-', '_')}_H\n\n")
    f.write(f"const signed char {array_name}[] = {{\n")
    mel_flat = mel_int8.flatten()
    for i, val in enumerate(mel_flat):
        f.write(f"{val},")
        if (i + 1) % 20 == 0:
            f.write("\n")
    f.write("};\n\n")
    f.write(f"#endif // SAMPLE_MEL_{bird_class.upper().replace('-', '_')}_H\n")

print(f"Header file written: {header_file}, total values = {mel_int8.size}")
