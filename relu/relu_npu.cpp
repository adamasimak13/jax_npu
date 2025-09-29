#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string>
#include <array>
#include <iostream>

extern "C" void relu_npu(float* input_array, float* output_array, int size, int shard_id) {

    char input_filename[64];
    char output_filename[64];

    sprintf(input_filename, "input_relu_shard%d.bin", shard_id);
    sprintf(output_filename, "output_relu_shard%d.bin", shard_id);

    FILE* f_in = fopen(input_filename, "wb");
    if (f_in) {
        fwrite(input_array, sizeof(float), (size_t)size, f_in);
        fclose(f_in);
    }

    char cmd[512];
    sprintf(cmd,
    "./relu.exe "
    "-x build/final.xclbin "
    "-i build/insts.bin "
    "-k MLIR_AIE "
    "--input-file %s "
    "--output-file %s "
    "-v 0"
    " --run-cpu", 
    input_filename,
    output_filename
);

    printf("[INFO] Shard %d is executing: %s\n", shard_id, cmd);
    fflush(stdout); // Ensure the log prints immediately

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "popen() failed!\n");
        return;
    }

    std::array<char, 128> buffer;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        printf("[NPU LOG Shard %d] %s", shard_id, buffer.data());
    }

    pclose(pipe);

    FILE* f_out = fopen(output_filename, "rb");
    if (f_out) {
        fread(output_array, sizeof(float), (size_t)size, f_out);
        fclose(f_out);
    } else {
        fprintf(stderr, "[ERROR] Shard %d: Failed to open result file %s\n", shard_id, output_filename);
    }

    remove(input_filename);
    remove(output_filename);

}
