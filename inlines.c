#include "common.h"


#ifdef DEBUG
pkdflt PKFLT(bloom_precision s)
{
  uint32_t a = *((uint32_t *)&s);
  if (!(a & 0x8000))
    return a >> 16;
  a >>= 16;
  if ((a & 0x7f) == 0x7f)
    return a; // don't overflow mantissa
  return a + 1;
}
#endif

#ifdef DEBUG
bloom_precision UNPKFLT(pkdflt s)
{
  uint32_t a = s;
  // format: bbloom_precision16
  a = (a << 16); //|0x8000;//a;
  return *((bloom_precision *)&a);
}
#endif
// #endif
// #endif

// /* innerloop of matrix multiplication (or "1d convolution").
//  * this is where most of the computation takes place.
//  */

#ifdef DEBUG
bloom_precision conv1dline(bloom_precision a, bloom_precision *v, bloom_precision *m, long long wdt)
{
  long long i;
  bloom_precision tmp = 0;

  for (i = 0; i < wdt; i++)
  {

    tmp += v[i] * m[i];
  }
  return tmp + a;
}

#endif

// #ifdef USE_PKDFLT
#ifdef DEBUG
bloom_precision conv1dline_pkd(bloom_precision a, bloom_precision *v, pkdflt *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * UNPKFLT(m[i]);
  return a;
}
#endif
// #else
// #define conv1dline_pkd conv1dline
// #endif

#ifdef DEBUG
int64_t conv1dline_pkdwte(int64_t a, int32_t *v, wte_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}

int conv1dline_ii(int a, int8_t *v, int8_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}

int conv1dline_fi(bloom_precision a, bloom_precision *v, int8_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}
#endif