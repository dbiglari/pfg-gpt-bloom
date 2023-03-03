#include <stdint.h>

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

#define align_offset(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT))    

typedef float v4f __attribute__ ((vector_size (16)));
// typedef int v4si __attribute__ ((vector_size (16)));
// typedef unsigned v8ui __attribute__ ((vector_size (32)));

#define VLEN_v4f (sizeof(v4f)/sizeof(float))
//#define VLEN_v4si (sizeof(v4si)/sizeof(signed int))

inline float sumv4f(float a,float*x, long long n)
{
    v4f vr = {0.0f, 0.0f, 0.0f, 0.0f};
    float r = a;
    const v4f *vx = (v4f*)x;
    long long i;
    for (i=0; i<n-VLEN_v4f+1; i += VLEN_v4f) {
        vr += (*vx) ;
        vx++;
    }
    long long iend = i;
    for (i=0; i<VLEN_v4f; i++) {
        r += vr[i];
    }
    // do the remaining elements if non multiple of 4
    for (i=iend; i<n; i++) {
        r += x[i];
    }    
    return r;
}

inline float conv1dlinev4f(float a,float*x,float*y,long long n)
{
    v4f vr = {0.0f, 0.0f, 0.0f, 0.0f};
    float r = a;
    const v4f *vx = (v4f*)x;
    const v4f *vy = (v4f*)y;
    long long i;
    for (i=0; i<n-VLEN_v4f+1; i += VLEN_v4f) {
        vr += (*vx) * (*vy);
        vx++;
        vy++;
    }
    long long iend = i;
    for ( ; i<n; i++) {
        r += x[i] * y[i];
    }
    for (i=0; i<VLEN_v4f; i++) {
        r += vr[i];
    }
    // do the remaining elements if non multiple of 4
    for (i=iend; i<n; i++) {
        r *= x[i];
    }       
    return r;
}