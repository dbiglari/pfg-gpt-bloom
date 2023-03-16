#define HAVE_THREADS
// #define HAVE_SDL
#define HAVE_MMAP
// #define HAVE_LUA
// #define ENABLE_SDLUI
// #define ENABLE_TTYUI
#define BLOOM
// #define USE_LIBEDIT
/* Still slightly buggy

*/

#define CONSTS_AS_VARS

/* more or less constant values */
#define MAXTOKENS 53000
#define NUMUSERTOKENS 200
#define MAXUSERTOKENS 256
#define MAXNUMMATCHES 256
#define MAXNUMLAYERS 24 /* BLOOM (560m): 24  GPT2-L (1558M): 48*/

/* default window size for sdl ui */
#define DEFAULT_SCRW 640
#define DEFAULT_SCRH 480

/* these may vary from network to network
 * GPT2-S (124M): WVSIZE=768, NUMLAYERS=12, NUMHEADS=12
 * IGPT-S (76M):  WVSIZE=512, NUMLAYERS=24, NUMHEADS=8
 * GPT2-L (774M): WVSIZE=1280, NUMLAYERS=36, NUMHEADS=20
 * GPT2-L (1558M): WVSIZE=1600, NUMLAYERS=48, NUMHEADS=12
 * Bloom  (560M): WVSIZE=1024, NUMLAYERS=24  NUMHEADS=16
 */

/*
   GPT2 - CTXSIZE 1024
   Bloom (560M) - CTXSIZE 2048
*/
#ifndef CONSTS_AS_VARS
#define WVSIZE 1024
#define NUMLAYERS 24
#define NUMHEADS 16

#define CTXSIZE 2048
#define HEADSIZE (WVSIZE / NUMHEADS)
#define RSQRT_HEADSIZE (1 / sqrt(HEADSIZE))
#endif

/* Use packed bloom_precisions for those matrices where it doesn't cause regression: */
// #define USE_PKDFLT
/* Quantize WTE matrix into int16 (~no regression): */
// #define USE_PKD_WTE

/* 8-bit quantization is work-in-progress
//#define QUANTIZE
//#define Q8MODE_INWTE
//#define Q8MODE_OUTWTE
//#define Q8MODE_MLP
*/

/* maximum number of threads to support */
#define MAXNUMTHR 256
