#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"

int main(int argc, char *argv[])  {
  if (argc != 3) {
    printf("Usage: %s <input_file> <output_file>\n", argv[0]);
    return 1;
  }

  char *input_filename = argv[1];
  char *output_filename = argv[2];  

  FILE *input_file, *output_file;
  float buffer;

  // Open the input file in binary mode
  input_file = fopen(input_filename, "rb");
  if (input_file == NULL) {
    printf("Error opening input file\n");
    return 1;
  }

  // Open the output file in binary mode
  output_file = fopen(output_filename, "wb");
  if (output_file == NULL) {
    printf("Error opening output file\n");
    return 1;
  }

  // Read the input file in 4 byte chunks
  while (fread(&buffer, sizeof(float), 1, input_file) == 1) {
    // Write the data to the output file
    FP32 finput;
    finput.f = buffer;
    FP16 outbuffer = float_to_half_full(finput);
    // try to convert it back
    FP32 backbuffer = half_to_float(outbuffer.u);
    float f_test = backbuffer.f;
    fwrite(&(outbuffer.u), sizeof(char)*2, 1, output_file);
  }

  // Close the input and output files
  fclose(input_file);
  fclose(output_file);

  return 0;
}