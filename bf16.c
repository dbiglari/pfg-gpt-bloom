#include <stdio.h>
#include <stdint.h>
#include "bf16.h"

#ifdef DEBUG
void FloatToBFloat16(const float *src, uint16_t *dst, int64_t size)
{
  const uint16_t *p = (uint16_t *)(src);
  uint16_t *q = (uint16_t *)(dst);
  for (; size; p += 2, q++, size--)
  {
    *q = p[1];
  }
}

void BFloat16ToFloat(const uint16_t *src, float *dst, int64_t size)
{
  const uint16_t *p = (uint16_t *)(src);
  uint16_t *q = (uint16_t *)(dst);
  for (; size; p++, q += 2, size--)
  {
    q[0] = 0;
    q[1] = *p;
  }
}
#endif