import numpy as np
import subprocess
import sys
import os

# --- 1. Test Configuration ---
# The script will get its input data by running data.py
DATA_GENERATOR_SCRIPT = "data.py"
INPUT_FILE = "input_data.bin" # The file created by data.py
OUTPUT_FILE = "standalone_relu_output.bin"
EXECUTABLE = "./relu.exe"

# --- 2. Generate Input Data by running data.py ---
print(f"--- Running data generator script: {DATA_GENERATOR_SCRIPT} ---")
try:
    # Execute data.py as a separate process
    subprocess.run([sys.executable, DATA_GENERATOR_SCRIPT], check=True)
    print(f"Successfully created '{INPUT_FILE}'")
except FileNotFoundError:
    print(f"[ERROR] The script '{DATA_GENERATOR_SCRIPT}' was not found in this directory.")
    sys.exit(1)
except subprocess.CalledProcessError:
    print(f"[ERROR] '{DATA_GENERATOR_SCRIPT}' encountered an error during execution.")
    sys.exit(1)
print("-" * 20)


# --- 3. Build and Execute the Command ---
# This list contains all the arguments that will be passed to relu.exe
command = [
    EXECUTABLE,
    "-x", "build/final.xclbin",
    "-i", "build/insts.bin",
    "-k", "MLIR_AIE",
    "--input-file", INPUT_FILE,
    "--output-file", OUTPUT_FILE,
    "-v", "1",
     "--run-cpu" # Verbosity level 1 to see detailed logs
]

print("Executing command:")
# We join the list into a string for printing, which makes it easy to copy/paste
print(" ".join(command))
print("-" * 20 + "\n")

try:
    # Execute the command and check for errors
    subprocess.run(command, check=True)
except FileNotFoundError:
    print(f"[ERROR] Executable '{EXECUTABLE}' not found. Have you compiled the C++ code?")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] C++ program exited with an error (code {e.returncode}). Check the output above for details.")
    sys.exit(1)

print("\n" + "-" * 20)
print("--- Standalone test finished successfully! ---")

# --- 4. (Optional) Verify Output ---
if os.path.exists(OUTPUT_FILE):
    print(f"\n--- Verifying output file '{OUTPUT_FILE}' ---")
    output_data = np.fromfile(OUTPUT_FILE, dtype=np.float32)
    print(f"Read {len(output_data)} elements from output.")
    if len(output_data) > 0:
        print(f"First 5 output elements: {output_data[:5]}")
else:
    print(f"[WARNING] Output file '{OUTPUT_FILE}' was not created.")

