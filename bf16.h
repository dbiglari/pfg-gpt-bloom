// inline void FloatToBFloat16(const float* src, uint16_t* dst, int64_t size);
// inline void BFloat16ToFloat(const uint16_t* src, float* dst, int64_t size);

#ifndef DEBUG
inline void FloatToBFloat16(const float* src, uint16_t* dst, int64_t size) {
  const uint16_t* p = (uint16_t*)(src);
  uint16_t* q = (uint16_t*)(dst);
  for (; size; p += 2, q++, size--) {
    *q = p[1];
  }
}
#else
void FloatToBFloat16(const float* src, uint16_t* dst, int64_t size);
#endif

#ifndef DEBUG
inline void BFloat16ToFloat(const uint16_t* src, float* dst, int64_t size) {
  const uint16_t* p = (uint16_t*)(src);
  uint16_t* q = (uint16_t*)(dst);
  for (; size; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
}
#else
void BFloat16ToFloat(const uint16_t* src, float* dst, int64_t size);
#endif
