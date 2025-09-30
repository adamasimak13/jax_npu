#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/file.h> // For flock

extern "C" void matmul_npu(float* A, float* B, float* C,
                           int M, int N, int K, int shard_id) {
    char input_A_file[64];
    char output_C_file[64];
    const char* input_B_file = "matrix_B.bin"; // B is common

    // Unique filenames per shard to avoid collisions
    sprintf(input_A_file, "matrix_A_%d.bin", shard_id);
    sprintf(output_C_file, "matrix_C_%d.bin", shard_id);

    // --- Save A shard ---
    FILE* f_in1 = fopen(input_A_file, "wb");
    if (!f_in1) {
        fprintf(stderr, "[ERROR] Shard %d: Failed to open %s for writing\n", shard_id, input_A_file);
        return;
    }
    fwrite(A, sizeof(float), (size_t)M*K, f_in1);
    fclose(f_in1);

    // --- Save B (common for all kernels), using a lock to prevent race condition ---
    // Shard 0 will write it, other shards will wait for it to be available.
    if (shard_id == 0) {
        FILE* f_in2 = fopen(input_B_file, "wb");
        if (!f_in2) {
            fprintf(stderr, "[ERROR] Shard %d: Failed to open %s for writing\n", shard_id, input_B_file);
            return;
        }
        // Lock the file to prevent others from reading a partial write
        int fd = fileno(f_in2);
        flock(fd, LOCK_EX);
        fwrite(B, sizeof(float), (size_t)K*N, f_in2);
        flock(fd, LOCK_UN); // Unlock after writing
        fclose(f_in2);
    } else {
        // Other shards wait until the file is available and unlocked
        while (access(input_B_file, F_OK) != 0) {
            usleep(10000); // Wait 10ms for the file to exist
        }
    }


    // Execute NPU kernel with shard-specific file arguments ---
    char cmd[512];
    sprintf(cmd,
        "./whole_array.exe "
        "-x build/final_256x768x768_32x32x32_4c.xclbin "
        "-i build/insts_256x768x768_32x32x32_4c.txt "
        "-k MLIR_AIE -M %d -K %d -N %d --b_col_maj 0 "
        "--a-file %s --b-file %s --c-file %s --no-verify", // Pass the unique filenames
        M, K, N, input_A_file, input_B_file, output_C_file
    );

    printf("[INFO] Shard %d executing command: %s\n", shard_id, cmd);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int ret = system(cmd);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (ret != 0) {
        fprintf(stderr, "[ERROR] Kernel %d failed with code %d\n", shard_id, ret);
        return;
    }

    
    // double elapsed = (end.tv_sec - start.tv_sec) +
    //                  (end.tv_nsec - start.tv_nsec) / 1e9;
    // printf("[TIMING] Kernel %d time: %.6f sec\n", shard_id, elapsed);

    // --- Load result C shard ---
    FILE* f_out = NULL;
    for (int i = 0; i < 10; ++i) {  // Retry in case file not ready
        f_out = fopen(output_C_file, "rb");
        if (f_out) break;
        usleep(10000);
    }

    if (!f_out) {
        fprintf(stderr, "[ERROR] Shard %d: Failed to open %s for reading\n", shard_id, output_C_file);
        return;
    }

    fread(C, sizeof(float), (size_t)M*N, f_out);
    fclose(f_out);
}