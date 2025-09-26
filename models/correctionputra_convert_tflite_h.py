# =========================================================
# correctionputra_convert_tflite_h.py
# Convert a TensorFlow Lite model (.tflite) into a C header (.h)
# suitable for Arduino / STM32 (uint8_t + 8-byte alignment)
# =========================================================

import sys
import os

def convert_tflite_to_header(tflite_path, header_path):
    if not os.path.exists(tflite_path):
        print(f"❌ ERROR: File not found -> {tflite_path}")
        return

    # Automatically create header guard and array name from filename
    header_guard = os.path.basename(header_path).replace('.', '_').upper()
    array_name = os.path.splitext(os.path.basename(header_path))[0]

    # Step 1: Read the TFLite model
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # Step 2: Write to C header file
    with open(header_path, "w") as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        f.write("#include <stdint.h>\n\n")

        # Declare array with 8-byte alignment
        f.write(f"alignas(8) const uint8_t {array_name}[] = {{")
        for i, b in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("\n ")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n\n")

        # Store model length
        f.write(f"const unsigned int {array_name}_len = {len(tflite_model)};\n\n")

        f.write(f"#endif // {header_guard}\n")

    print(f"✅ Conversion complete: {tflite_path} → {header_path}")
    print(f"Model size: {len(tflite_model)/1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correctionputra_convert_tflite_h.py <input_model.tflite> <output_model.h>")
    else:
        convert_tflite_to_header(sys.argv[1], sys.argv[2])
