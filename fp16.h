#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>

#define LIBPHM_LITTLE_ENDIAN

typedef union FP32 {
  unsigned int u;
  float f;
  struct {
#ifdef LIBPHM_LITTLE_ENDIAN
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
#else
    unsigned int Sign : 1;
    unsigned int Exponent : 8;
    unsigned int Mantissa : 23;
#endif
  } s;
}FP32;

typedef union FP16 {
  unsigned short u;
  struct {
#ifdef LIBPHM_LITTLE_ENDIAN
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
#else
    unsigned int Sign : 1;
    unsigned int Exponent : 5;
    unsigned int Mantissa : 10;
#endif
  } s;
} FP16;


//double FP16_buf_to_double(uint8_t *buf);
union FP16 float_to_half_full(union FP32 f);
//union FP32 half_to_float(union FP16 h);
union FP32 half_to_float(unsigned short h_u);
void swap2(unsigned short *val);

// void BFloat16ToFloat(const uint16_t* src, float* dst, int64_t size);
// void FloatToBFloat16(const float* src, uint16_t* dst, int64_t size);
