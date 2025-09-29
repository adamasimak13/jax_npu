#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>

// Ορισμοί τύπων δεδομένων
using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;

// --- Καθολικές μεταβλητές για τα ονόματα των αρχείων ---
std::string g_inputFile;
std::string g_outputFile;


// --- Function to get the size of a file in bytes ---
long get_file_size(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "FATAL ERROR: Could not open file to get size: " << filename << std::endl;
        return -1;
    }
    return file.tellg();
}

// --- Συνάρτηση αρχικοποίησης που χρησιμοποιεί την καθολική μεταβλητή g_inputFile ---
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE, const std::string& input_filename) {
    std::ifstream infile(input_filename, std::ios::binary);
      if (infile.is_open()) {
        std::vector<float> temp(SIZE);
        infile.read(reinterpret_cast<char *>(temp.data()), SIZE * sizeof(float));
        infile.close();
        for (int i = 0; i < SIZE; ++i) {
          bufIn1[i] = static_cast<DATATYPE_IN1>(temp[i]);
        }
      } else {
    // Κάντε το μήνυμα σφάλματος πιο χρήσιμο
        std::cerr << "Error opening input file: " << input_filename << "\n";
        exit(EXIT_FAILURE);
      }
    }

void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE * sizeof(DATATYPE_OUT));
}

void write_output_data(DATATYPE_OUT *bufOut, int SIZE, const std::string& output_filename) {
    std::ofstream outfile(output_filename, std::ios::binary);
    if (outfile.is_open()) {
        std::vector<float> temp(SIZE);
        for (int i = 0; i < SIZE; ++i) {
            temp[i] = static_cast<float>(bufOut[i]);
        }
        outfile.write(reinterpret_cast<const char*>(temp.data()), SIZE * sizeof(float));
        outfile.close();
    } else {
        std::cerr << "Error opening output file for writing: " << output_filename << "\n";
        exit(EXIT_FAILURE);
    }
}
// --- Συνάρτηση επαλήθευσης (παραμένει ίδια) ---
int verify_relu_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
  int errors = 0;
  for (uint32_t i = 0; i < SIZE; i++) {
    if (std::isnan(bufIn1[i])) continue;
    DATATYPE_OUT ref = (DATATYPE_OUT)0;
    if (bufIn1[i] > (DATATYPE_OUT)0) ref = bufIn1[i];
    if (!test_utils::nearly_equal(ref, bufOut[i])) {
      errors++;
    }
  }
  return errors;
}

// --- CPU ReLU implementation ---
void cpu_relu(const DATATYPE_IN1 *input, float *output, int size) {
    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(input[i]); // Convert bfloat16 to float
        output[i] = (val > 0.0f) ? val : 0.0f;
    }
}

int main(int argc, const char *argv[]) {
  // Ορισμός σταθερών στην αρχή της main
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  // --- Διαχείριση παραμέτρων ---
  cxxopts::Options options("XRT Test Wrapper", "Test harness for NPU kernels");
  options.allow_unrecognised_options();
  options.add_options()
    ("h,help", "produce help message")
    ("x,xclbin", "the input xclbin path", cxxopts::value<std::string>())
    ("k,kernel", "the kernel name in the XCLBIN", cxxopts::value<std::string>())
    ("v,verbosity", "the verbosity of the output", cxxopts::value<int>()->default_value("0"))
    ("i,instr", "path of file containing instructions", cxxopts::value<std::string>())
    ("verify", "whether to verify the AIE computed output", cxxopts::value<bool>()->default_value("true"))
    ("input-file", "Input data file for this run", cxxopts::value<std::string>())
    ("output-file", "Output data file for this run", cxxopts::value<std::string>());

  auto vm = options.parse(argc, argv);

  if (vm.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
  }

  // Δημιουργία και γέμισμα του struct 'args'
  args myargs;
  myargs.verbosity = vm["verbosity"].as<int>();
  myargs.do_verify = vm["verify"].as<bool>();
  myargs.instr = vm["instr"].as<std::string>();
  myargs.xclbin = vm["xclbin"].as<std::string>();
  myargs.kernel = vm["kernel"].as<std::string>();
  myargs.n_iterations = 20;
  myargs.n_warmup_iterations = 5;
  myargs.trace_size = 0;

  // **ΔΙΟΡΘΩΣΗ**: Ανάθεση των ονομάτων αρχείων στο myargs
  myargs.input_file = vm["input-file"].as<std::string>();
  myargs.output_file = vm["output-file"].as<std::string>();

  if (myargs.input_file.empty() || myargs.output_file.empty()) {
      std::cerr << "Error: --input-file and --output-file arguments are required." << std::endl;
      return 1;
  }
    // --- Load input data into host memory first ---
  std::vector<DATATYPE_IN1> host_input_buffer(IN1_VOLUME);
  // We'll use the existing initialize_bufIn1 to load data into our vector
  initialize_bufIn1(host_input_buffer.data(), IN1_VOLUME, myargs.input_file);

     // --- ADDED: CPU ReLU Calculation and Timing ---
  std::cout << "\n--- Running ReLU on CPU for comparison ---" << std::endl;
  std::vector<float> cpu_output_buffer(OUT_VOLUME);
  
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpu_relu(host_input_buffer.data(), cpu_output_buffer.data(), IN1_VOLUME);
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
  std::cout << "CPU execution time: " << cpu_duration << " us" << std::endl;
  std::cout << "-----------------------------------------\n" << std::endl;
  // --- END of CPU Section ---
    
  // Δημιουργία lambdas για να περάσουμε παραμέτρους στο wrapper
  // Use a lambda to pass the pre-loaded data to the AIE wrapper
  auto init_in_lambda = [&](DATATYPE_IN1* buf, int size) {
    memcpy(buf, host_input_buffer.data(), size * sizeof(DATATYPE_IN1));
  };

  std::vector<DATATYPE_OUT> result_buffer(OUT_VOLUME);
  
  auto verify_and_capture_lambda = [&](DATATYPE_IN1* b_in, DATATYPE_OUT* b_out, int size, int verbosity) {
      memcpy(result_buffer.data(), b_out, size * sizeof(DATATYPE_OUT));
      return verify_relu_kernel(b_in, b_out, size, verbosity);
  };

  // --- Κλήση του test wrapper με τα lambdas ---
  // **ΔΙΟΡΘΩΣΗ**: Χρησιμοποιούμε τα lambdas ως ορίσματα του template
 int aie_res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT>(
    IN1_VOLUME,
    OUT_VOLUME,
    myargs,
    init_in_lambda,                 // Pass the lambda directly
    initialize_bufOut,              // Pass the regular function directly
    verify_and_capture_lambda       // Pass the lambda directly
);
    

  if (aie_res != 0) {
      std::cerr << "[ERROR] AIE run failed.\n" << std::flush;
      return aie_res;
  }

  // Εγγραφή των αποτελεσμάτων στο αρχείο εξόδου
  write_output_data(result_buffer.data(), OUT_VOLUME, myargs.output_file);
  std::cout << "[INFO] AIE run completed successfully. Output written to " << myargs.output_file << std::endl;

  std::cout << ">>> AIE KERNEL PASSED <<<\n" << std::flush;
  return 0;
} // <-- Το σωστό σημείο για το κλείσιμο της main