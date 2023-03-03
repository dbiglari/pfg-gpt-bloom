#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
//#include "json-c/json.h"
//#include "zip.h"
#include "common.h"
// #include "fp16.h"

// uint16_t fp32tobf16(float x)
// {
//     float y = x;
//     int *p = (int *) &y;
//     unsigned int exp, man;
//     exp = *p & 0x7F800000u;
//     man = *p & 0x007FFFFFu;
//     if (exp == 0 && man == 0) {
//         // zero
//         return x;
//     }
//     if (exp == 0x7F800000u) {
//         // infinity or Nans
//         return x;
//     }
//     // Normalized number
//     // round to nearest
//     float r = x;
//     int *pr = (int *) &r;
//     *pr &= 0xff800000; // r has the same exp as x
//     r = r / 256;
//     y = x + r;

//     *p &= 0xffff0000;

//     uint16_t ret = *p >> 16;
//     return ret;
// }

uint16_t fp32tobf16(float x)
{
  uint32_t f32 = *(uint32_t *)&x;
  int32_t f = (f32 & 0x7fffffff) - 0x38000000;
  if (f < 0)
  {
    return 0;
  }
  else if (f > 0x477fe000 - 0x38000000)
  {
    return 0x7c00;
  }
  else
  {
    uint32_t y = (f32 + 0x10000000) & 0xffffe000;
    return y >> 13;
  }
}

// float bf16tofp32(uint16_t x)
// {
//   int16_t exponent = (x & 0x7c00) >> 10;
//   int16_t fraction = x & 0x03ff;

//   if (exponent == 0) {
//     exponent = -14;
//   } else {
//     exponent -= 15;
//   }

//   float val = (float) (fraction * pow(2, exponent - 10)) / 1024.0f + pow(2, exponent);
//   return val;
// }

// void FloatToBFloat16(const float* src, uint16_t* dst, int64_t size) {
//   const uint16_t* p = (uint16_t*)(src);
//   uint16_t* q = (uint16_t*)(dst);
//   for (; size; p += 2, q++, size--) {
//     *q = p[1];
//   }
// }

// void BFloat16ToFloat(const uint16_t* src, float* dst, int64_t size) {
//   const uint16_t* p = (uint16_t*)(src);
//   uint16_t* q = (uint16_t*)(dst);
//   for (; size; p++, q += 2, size--) {
//     q[0] = 0;
//     q[1] = *p;
//   }
// }

// float bf16tofp32(uint16_t x) {

//   float ret;
//   BFloat16ToFloat(&x, &ret);

//   return ret;
// }

// float bf16tofp32(uint16_t x) {
//     //swap2(&x);
//     x = (x >> 8) | (x << 8);
//     uint32_t sign = x >> 15;
//     uint32_t exponent = (x >> 10) & 0x1F;
//     uint32_t mantissa = x & 0x3FF;
//     int32_t exp = exponent - 15;
//     float result;

//     if (exponent == 0) {
//         // Denormalized number
//         result = ldexp((float)mantissa / 1024.0f, exp);
//     } else if (exponent == 31) {
//         // Inf or NaN
//         result = NAN;
//     } else {
//         // Normalized number
//         result = ldexp(1.0f + (float)mantissa / 1024.0f, exp);
//     }

//     float val = sign ? -result : result;
//     return val;
// }

void swap2(unsigned short *val)
{
  unsigned short tmp = *val;
  unsigned char *dst = (unsigned char *)(val);
  unsigned char *src = (unsigned char *)(&tmp);

  dst[0] = src[1];
  dst[1] = src[0];
}


union FP32 half_to_float(unsigned short h_u)
{
  static const union FP32 magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13; // exponent mask after shift
  union FP32 o;

  o.u = (h_u & 0x7fffU) << 13U;          // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u; // just the exponent
  o.u += (127 - 15) << 23;               // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)   // Inf/NaN?
    o.u += (128 - 16) << 23; // extra exp adjust
  else if (exp_ == 0)        // Zero/Denormal?
  {
    o.u += 1 << 23; // extra exp adjust
    o.f -= magic.f; // renormalize
  }

  o.u |= (h_u & 0x8000U) << 16U; // sign bit
  return o;
}

// union FP32 half_to_float(union FP16 h)
// {
//   static const union FP32 magic = {113 << 23};
//   static const unsigned int shifted_exp = 0x7c00
//                                           << 13; // exponent mask after shift
//   union FP32 o;

//   o.u = (h.u & 0x7fffU) << 13U;          // exponent/mantissa bits
//   unsigned int exp_ = shifted_exp & o.u; // just the exponent
//   o.u += (127 - 15) << 23;               // exponent adjust

//   // handle exponent special cases
//   if (exp_ == shifted_exp)   // Inf/NaN?
//     o.u += (128 - 16) << 23; // extra exp adjust
//   else if (exp_ == 0)        // Zero/Denormal?
//   {
//     o.u += 1 << 23; // extra exp adjust
//     o.f -= magic.f; // renormalize
//   }

//   o.u |= (h.u & 0x8000U) << 16U; // sign bit
//   return o;


//   //  FP32 o = { 0 };

//   //   // From ISPC ref code
//   //   if (h.s.Exponent == 0 && h.s.Mantissa == 0) // (Signed) zero
//   //       o.s.Sign = h.s.Sign;
//   //   else
//   //   {
//   //       if (h.s.Exponent == 0) // Denormal (converts to normalized)
//   //       {
//   //           // Adjust mantissa so it's normalized (and keep
//   //           // track of exponent adjustment)
//   //           int e = -1;
//   //           uint m = h.s.Mantissa;
//   //           do
//   //           {
//   //               e++;
//   //               m <<= 1;
//   //           } while ((m & 0x400) == 0);

//   //           o.s.Mantissa = (m & 0x3ff) << 13;
//   //           o.s.Exponent = 127 - 15 - e;
//   //           o.s.Sign = h.s.Sign;
//   //       }
//   //       else if (h.s.Exponent == 0x1f) // Inf/NaN
//   //       {
//   //           // NOTE: Both can be handled with same code path
//   //           // since we just pass through mantissa bits.
//   //           o.s.Mantissa = h.s.Mantissa << 13;
//   //           o.s.Exponent = 255;
//   //           o.s.Sign = h.s.Sign;
//   //       }
//   //       else // Normalized number
//   //       {
//   //           o.s.Mantissa = h.s.Mantissa << 13;
//   //           o.s.Exponent = 127 - 15 + h.s.Exponent; 
//   //           o.s.Sign = h.s.Sign;
//   //       }
//   //   }

//   //   return o;

// }

union FP16 float_to_half_full(union FP32 f)
{
  union FP16 o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0) // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255) // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
  }
  else // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31) // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0) // Underflow
    {
      if ((14 - newexp) <= 24) // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000; // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1) // Check for rounding
          o.u++;                         // Round, might overflow into exp bit, but this is OK
      }
    }
    else
    {
      o.s.Exponent = (unsigned int)(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000) // Check for rounding
        o.u++;                   // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;
  return o;
}

// pointer to FP16 (2 bytes), return converted double
// double FP16_buf_to_double(uint8_t *buf)
// {
//   FP16 *x = (FP16 *)buf;
//   FP32 y = half_to_float(*x);
//   double z = y.f;
// }
