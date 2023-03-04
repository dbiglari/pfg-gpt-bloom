#include "config.h"
#ifdef HAVE_SDL
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#endif
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#ifdef HAVE_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#endif

#include "fp16.h"
#include "fp32to8bit.h"
#include "structs.h"
#include "fastbarrier.h"

int serverPort;
bool startServer;
// Configuration options:  SEE README_appendix.txt

// Eventually most of these options will be turned into command line options.

// debug mode
// #define DEBUG

int global_numthreads;
#define MAXNUMMODELS 64
#define MAXNUMQUERIES 10

// Print performance measurement time for loading a layer from disk
//#define MEASURE_LOAD_TIME
// Print performance measurement time for computing a layer
//#define MEASURE_RUN_TIME
// Print performance measurement time for computing all layers
// #define MEASURE_ALL_LAYERS_TIME
// Print performance measurement time for token selection
// #define MEASURE_TOKEN_TIME

// Produces the same output, every time regardless of random seed chosen
// Uses the greedy token selection algorithm selecting the maximum logit.
#define DETERMINISTIC_OUTPUT

// force a specific number of tokens before allowing any special tokens such as </s>
// -1 forces at least half the number of prompted tokens
// <-1 always prevents special tokens (generate forever, until interrupted)
#define FORCE_GEN_TOKENS -2

// Don't exceed this many tokens
#define HARDMAX_GEN 30

// try to stop at the end of a sentence after this many tokens
#define GRAMMARMAX_GEN 30

// Experimental thread barrier
#define FAST_BARRIER

// Enable 8bit mode (saves memory and computation time at slight expense of accuracy)
#define USE_8BIT

// Enable bfloat16 (used only for 175b model)
// #define USE_BFLOAT16

// produces different output with little speed benefit
// #define EXPERIMENTAL_THREADED_NORMALIZATION

// don't load the weights until they're needed
//#define LOAD_WEIGHTS_ON_DEMAND

// enable usage of SIMD isntructions, not useful, -O3 produces similar optimizations automatically
// #define USE_SIMD

// extract weights on demand, useful to save memory during execution, but increases computation time
//#define EXTRACT_WEIGHTS_ON_DEMAND

// unload weights to conserve ram, if enabled, be sure that LOAD_WEIGHTS_ON_DEMAND is also enabled
//#define UNLOAD_WEIGHTS_NOT_IN_USE

// Unimplemented defines for future networked mode

// accept queries on a REST interface
#define USE_REST_INTERFACE

// receive farmed out jobs
#define ACT_AS_WORKER
#define JOB_SERVER_ADDRESS localhost

// farm out jobs to servers
#define ACT_AS_MANAGER

// if acting as a worker, set the shard number to force using only a specifc shard, supports multiple comma separated shards
// numbers that exceed the maximum shard number for a model are ignored.  -1 indicates any shard can be used by this worker
#define SHARD_NUMBER \
  {                  \
    -1               \
  }

// Maximum number of shards that should be loaded at once (suggested 1 for cluster mode, 2 for single machine mode)
#define MAX_NUM_SHARDS 2

#define MAX_NUM_SHARDS_IN_SET 200

#define FILES_PER_LAYER 12

#ifdef __MAIN__
#define global
#else
#define global extern
#endif

// #define USE_DOUBLE_PRECISION
#ifdef USE_DOUBLE_PRECISION
typedef double bloom_precision;
#else
typedef float bloom_precision;
#endif

#ifdef BLOOM
// bloom uses 32 bit tokens
typedef uint32_t token_t;
#else
typedef uint16_t token_t;
#endif

#ifndef USE_PKDFLT
typedef bloom_precision pkdflt;
#else
#ifdef USE_NATIVE_FP16
typedef __fp16 pkdflt;
#else
typedef uint16_t pkdflt;
#endif
#endif

#ifdef USE_PKD_WTE
typedef int16_t wte_t;
#else
typedef bloom_precision wte_t;
#endif

typedef struct
{
  /* constants (network parameters) */
  bloom_precision *ln1_b, *ln1_g, *ln2_b, *ln2_g;

  // high precision
  bloom_precision *mlp_cfc_b;
  pkdflt *mlp_cfc_w;
  bloom_precision *mlp_cproj_b;
  pkdflt *mlp_cproj_w;
  bloom_precision *attn_cattn_b;
  pkdflt *attn_cattn_w;
  bloom_precision *attn_cproj_b;
  pkdflt *attn_cproj_w;

  // single precision
  float *s_ln1_b, *s_ln1_g, *s_ln2_b, *s_ln2_g;
  float *s_mlp_cfc_b;
  float *s_mlp_cfc_w;
  float *s_mlp_cproj_b;
  float *s_mlp_cproj_w;
  float *s_attn_cattn_b;
  float *s_attn_cattn_w;
  float *s_attn_cproj_b;
  float *s_attn_cproj_w;

  // fp16 precision (pointers to raw data)
  FP16 *fp16_ln1_b, *fp16_ln1_g, *fp16_ln2_b, *fp16_ln2_g;
  FP16 *fp16_mlp_cfc_b;
  FP16 *fp16_mlp_cfc_w;
  FP16 *fp16_mlp_cproj_b;
  FP16 *fp16_mlp_cproj_w;
  FP16 *fp16_attn_cattn_b;
  FP16 *fp16_attn_cattn_w;
  FP16 *fp16_attn_cproj_b;
  FP16 *fp16_attn_cproj_w;

  // 8bit precision (pointers to raw data)
  int8_t *q8_ln1_b, *q8_ln1_g, *q8_ln2_b, *q8_ln2_g;
  int8_t *q8_mlp_cfc_b;
  int8_t *q8_mlp_cfc_w;
  int8_t *q8_mlp_cproj_b;
  int8_t *q8_mlp_cproj_w;
  int8_t *q8_attn_cattn_b;
  int8_t *q8_attn_cattn_w;
  int8_t *q8_attn_cproj_b;
  int8_t *q8_attn_cproj_w;

#ifdef QUANTIZE
  /* 8-bit versions of the network parameters */
  int8_t *mlp_cproj8w, *mlp_cfc8w,
      *mlp_cproj8b, *mlp_cfc8b;
  /* quantizers (something to divide/multiply with) */
  bloom_precision mlp_cfc_w_q;
  bloom_precision mlp_cproj_w_q;
#endif
  /* variables */
  bloom_precision *k, *v;
} hlayer;

typedef struct
{
  bloom_precision prob;
  token_t tok;
} match_t;

/* not used yet! we'll support several simultaneous contexts in the future */
typedef struct
{
  token_t *in;
  bloom_precision *k;
  bloom_precision *v;
  int lgt;
  int validlgt;
} context_t;

/* for packed format */

#define FLAG_HAVE_BASES 1
#define FLAG_HAVE_WTET 2
#define FLAG_HAVE_SOS 4
#define FLAG_HAVE_PALETTE 8
#define FLAG_HAVE_TOKENSTRINGS 16
#define MTYPE_GPT2 0
#define MTYPE_IGPT 1
#define PFMT_bloom_precision32 0
#define PFMT_BF16 1
#define PFMT_IEEE16 2
#define PFMT_INT16 3
#define PFMT_INT8 4

typedef struct
{
  char fileformat[4];
  uint32_t wvsize;
  uint32_t numlayers;
  uint32_t numheads;
  uint32_t numtokens;
  uint32_t ctxsize;
  uint32_t headsize;
  char flags;
  char paramformat;
  char wteformat;
  char reserved0;
  bloom_precision quanter_wte;
} header_t;

#ifdef QUANTIZE
global int8_t *wte8;
global int8_t *wpe8;
#endif
/* igpt extras */
global bloom_precision *palette;

/* quantization extras */
#ifdef QUANTIZE
global bloom_precision quanter_wpe;
#else
#define quanter_wpe 1.0
#endif

/* settings */

// global struct
// {
//   char*name;
//   volatile void*ptr;
//   char type;
// } settingvars[]

// contains data for a shard
// the buf and/or layer in this structure can be loaded/unloaded to conserve memory
typedef struct shard_t
{
  // source
  char filename[2048];

  // data
  void **bufs;
  //int *indices;

  hlayer *layer;

  int shard_number;
  int layer_number;

  // flags to control memory usage
  bool expand_to_layer_struct;

  bool unload_buf_after_use;
  bool unload_layer_after_use;

} shard_t;

typedef struct model_thread_args_t
{
  int thr;
  int querynum;
  int modelnum;
} model_thread_args_t;

#ifdef HAVE_THREADS
typedef struct thrglob_t
{
  volatile pthread_t t[MAXNUMTHR];
  volatile pthread_barrier_t barrier;
  bloom_precision *x;
  int slot;
  int numthr;
  bloom_precision *q;
  bloom_precision *tmp;
  bloom_precision *xn;
  bloom_precision *mlp;
  fast_barrier_t fastbarrier;
  pthread_barrierattr_t fastbarrierattributes;

  bloom_precision mean_temp[MAXNUMTHR];
  bloom_precision mean;
  bloom_precision smean_temp[MAXNUMTHR];
  bloom_precision smean;
} thrglob_t;
#endif

typedef struct query_t
{

  bloom_precision temperature;
  bloom_precision temperature_alt;
  bloom_precision minp;
  bloom_precision minp_for_tagged; /* not used (yet?) */
  int nummatches;

  bloom_precision *attentions;
  bloom_precision *attentions_presoftmax;
  bloom_precision *att;
  bloom_precision *attention_mask;
  bloom_precision *attention_arrange_tensor;

  /* context buffer */
  token_t *context; /* alloc in init() */
  // ^ todo global context_t*context;
  volatile int start;
  volatile int currslot;
  volatile int genstart;
  volatile int genend;
  bloom_precision *lm_logits;
  bloom_precision *currwv; /* alloc in init() */
  int seed;
  int mode;  // 0 = greedy, 1 = sampling
  bool in_use;


  int no_repeat_ngrams;  
  int stop_after_ngram_repeats;
  int start_n_gram_search_on_current_response; // 1 = yes, 0 = no
  
  int force_gen_tokens;
  int hardmax_gen;
  int grammarmax_gen;
  int paragrammarmax_gen;
  char *response;

  thrglob_t thrglob;
  bool isInitialized;
} query_t;

typedef struct gradient_t
{
  float *s_ln1_b_gradients;
  float *s_ln1_g_gradients;
  float *s_ln2_b_gradients;
  float *s_ln2_g_gradients;
  float *s_mlp_cfc_b_gradients;
  float *s_mlp_cfc_w_gradients;
  float *s_mlp_cproj_b_gradients;
  float *s_mlp_cproj_w_gradients;
  float *s_attn_cattn_b_gradients;
  float *s_attn_cattn_w_gradients;
  float *s_attn_cproj_b_gradients;
  float *s_attn_cproj_w_gradients;
  float *s_wte_gradients;
  float *s_welw_gradients;
  float *s_welb_gradients;
  float *s_lnf_g_gradients;
  float *s_lnf_b_gradients;
} gradient_t;

typedef struct model_t
{
  char path[2048];
  char path_to_zip[2048];
  layerfiles_t layerfiles;
  layerfiles_t **shard_layerfiles;
  layerfiles_t *shard_lnfiles;
  layerfiles_t *shard_wtefiles;
  char *sequence;

  bool useshards;
  int num_shards;
  void *weight_map;
  shard_t *shards;
  bloom_precision quanter_wte;

  int numthreads;
  int verbose;
  /* vocabulary & word vector handling */
  int numtokens;
  int numwtetokens;

  int nummodeltokens;
  char **tokenstrings; /* alloc in loadtokens() */
  match_t *matchlist;  /* alloc in init() */
  char *tokenflags;    /* alloc in loadtokens() */
  token_t *tokenrepls; /* alloc in flagTokenForReplace() */
  bloom_precision *targetwv;
  token_t emptytoken; /* set in init() */
  char *tokendata;    /* alloc in loadtokens() */

  bloom_precision *outputcache;
  bloom_precision *alibi;

  bloom_precision base;
  bloom_precision closest_power_of_2;

  bool inUse;
  /* model */
  header_t *h;
  char *modelpath;
  char modelname[256];
  wte_t *wte;
  float *s_wte;
  FP16 *fp16_wte;
  void *q8_wte;
  pkdflt *wpe;
  // welw - bloom model word embeddings layer normalization weights
  float *s_welw;
  FP16 *fp16_welw;
  void *q8_welw;
  pkdflt *welw;
  // welb - bloom model word embeddings layer normalization biases
  float *s_welb;
  pkdflt *welb;
  FP16 *fp16_welb;
  void *q8_welb;
  wte_t **userwte; /* alloc in ui_init() */
  wte_t *wtet;
  wte_t *sos;
  hlayer *layers;
  bloom_precision *lnf_b;
  bloom_precision *lnf_g;
  float *s_lnf_b;
  FP16 *fp16_lnf_b;
  void *q8_lnf_b;
  float *s_lnf_g;
  FP16 *fp16_lnf_g;
  void *q8_lnf_g;

#ifdef CONSTS_AS_VARS
  int WVSIZE;
  int NUMLAYERS;
  int NUMHEADS;

  int CTXSIZE;
  bloom_precision HEADSIZE;
  bloom_precision RSQRT_HEADSIZE;
#endif

  int shardfilenum;

  gradient_t gradients;
  bool isInitialized;
  bool use_bfloat16;
  bool use_8bit;
} model_t;

// queries being run through the model
global query_t *queries;

// models loaded by the system
global model_t *models;

/* ui */
#ifdef HAVE_SDL
#ifdef ENABLE_SDLUI
global SDL_Surface *fb;
global int scrw, scrh;
#endif
#endif

/* functions */
void *readfile(char *fn, int *lgt_ret, char *path);
char *readtextfile(char *fn, char *path);
void runModel(bloom_precision *x, int slot, int modelnum, int querynum);
void renderwordvec(bloom_precision *wv0, int x0, int y0, int dim);
void renderlayernode(bloom_precision *wv, bloom_precision *att, int numheads, int x0, int y0);
void matchToTokens(bloom_precision *wv, match_t *o, int num, bloom_precision temp, int modelindex);
int pickmatch(match_t *list, int sz, bloom_precision minp, bool allowspecial, int modelindex);
wte_t *getwv(long long token, int modelindex);
wte_t *getwv_final(long long token, int modelindex);
bloom_precision tuneTemperatureByContext(int i, int querynum, int modelnum);
void clearcontext(int i);
void purgeoldcontext(int p, int modelnum, int querynum);
int loadtokens_from_tokendata(char *tokendata, int numtokens);
void vzlua(char *scriptfile);
//char *str_replace(char *s, char *old, char *new_str);
char *str_replace(const char *in, const char *pattern, const char *by);
int tokenize_to_context(char *src, int idx, int modelindex, int queryindex);
void linear_transform(bloom_precision *input, bloom_precision *output, bloom_precision *weights, bloom_precision *bias, long long input_size, long long output_size);
int replacetoken(int t, int modelnum);
void ttyui();
void loadmodel(char *modelpath);
int tokenize(char *src, int modelnum);
int savepackedmodel(char *fn, int modelindex);
void http_server();
int loadtokens(char *path);
int loadpalette(char *path);
void syncthreads(int thr, int querynum);
void generate(int start, int genstart_, int genend_, int modelnum, int querynum, bool displayprompt);
int initModel(char *modelpath, int modelnum);
void initQuery(int modelnum, int querynum);
void freeModel(int modelnum);
void freeQuery(int querynum);



#ifdef DEBUG
bloom_precision conv1dline(bloom_precision a, bloom_precision *v, bloom_precision *m, long long wdt);
bloom_precision conv1dline_pkd(bloom_precision a, bloom_precision *v, pkdflt *m, int wdt);
int64_t conv1dline_pkdwte(int64_t a, int32_t *v, wte_t *m, int wdt);
int conv1dline_ii(int a, int8_t *v, int8_t *m, int wdt);
int conv1dline_fi(bloom_precision a, bloom_precision *v, int8_t *m, int wdt);
#endif
/* markov chain */

#if (0)
typedef struct _prob_t
{
  token_t t;
  int freq;
  _prob_t *lt;
  _prob_t *gt;
} prob_t;

typedef struct _markovnode_t
{
  token_t t;
  _markovnode_t *lt;
  _markovnode_t *gt;
  _markovnode_t *next;
  prob_t *probs;
} markovnode_t;

markovnode_t *markovchain;

prob_t *markov_getprobs(markovnode_t *tree, token_t *ctx);
markovnode_t *markov_import(markovnode_t *tree, char *s, int ctxlgt);
int pickMatchWithMarkov(match_t *list, int sz, int slot);
#endif
// void markov_compress(markovnode_t*tree);

#define frand() ((rand() & 65535) / 65536.0)

/*** types and conversion macros for packed bloom_precisions ***/

#ifndef USE_PKDFLT
#ifndef DEBUG
#define UNPKFLT(s) (s)
#define PKFLT(s) (s)
#endif

#else
#ifdef USE_NATIVE_FP16
#define UNPKFLT(s) (s)
#define PKFLT(s) (s)
#define packtensor(s, lgt) (s)

#else
typedef uint16_t pkdflt;

#ifndef DEBUG
inline pkdflt PKFLT(bloom_precision s)
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

#ifndef DEBUG
inline bloom_precision UNPKFLT(pkdflt s)
{
  uint32_t a = s;
  // format: bbloom_precision16
  a = (a << 16); //|0x8000;//a;
  return *((bloom_precision *)&a);
}
#endif

#endif
#endif

#define UNPKWTE(a) (a / quanter_wte)

/* innerloop of matrix multiplication (or "1d convolution").
 * this is where most of the computation takes place.
 */

#ifndef DEBUG
inline bloom_precision conv1dline(bloom_precision a, bloom_precision *v, bloom_precision *m, long long wdt)
{
  long long i;
  for (i = 0; i < wdt; i++)
  {
    a += v[i] * m[i];
  }
  return a;
}
#endif

#ifdef USE_PKDFLT
#ifndef DEBUG
inline bloom_precision conv1dline_pkd(bloom_precision a, bloom_precision *v, pkdflt *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * UNPKFLT(m[i]);
  return a;
}
#endif
#else
#ifndef DEBUG
#define conv1dline_pkd conv1dline
#endif
#endif

#ifndef DEBUG
inline int64_t conv1dline_pkdwte(int64_t a, int32_t *v, wte_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}

inline int conv1dline_ii(int a, int8_t *v, int8_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}

inline int conv1dline_fi(bloom_precision a, bloom_precision *v, int8_t *m, int wdt)
{
  int i;
  for (i = 0; i < wdt; i++)
    a += v[i] * m[i];
  return a;
}
#endif

typedef struct settingvars_t
{
  char *name;
  volatile void *ptr;
  char type;
} settingvars_t;

global settingvars_t *settingvars;
#define NUMVARS 6
