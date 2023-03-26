#include "common.h"
#include "raw_loader.h"
#include "fastbarrier.h"
#include "simd.h"
//#include "bf16.h"
#include "fp32to8bit.h"

/* math helper functions */
#define EPSILON 0.00001


uint32_t fast_reduce(uint32_t x, uint32_t N) {
  return ((uint64_t) x * (uint64_t) N) >> 32 ;
}


uint32_t reduce(uint32_t x, uint32_t N) {
  return x % N;
}


/**
 * @brief  Threaded normalize
 * @note   
 * @param  *o: 
 * @param  *x: 
 * @param  *b: 
 * @param  *g: 
 * @param  thr: 
 * @param  modelnum: 
 * @param  querynum: 
 * @retval None
 */
void normalize_thr(bloom_precision *o, bloom_precision *x, bloom_precision *b, bloom_precision *g, int thr, int modelnum, int querynum)
{
  long long start;
  long long end;
  bloom_precision arrsize;
  int numthr = queries[querynum].thrglob.numthr;

  long long i;
  bloom_precision muller;
  float a = 0;
  // for (i = 0; i < models[modelnum].WVSIZE; i++)
  arrsize = models[modelnum].WVSIZE;
  bloom_precision arrsize_float = arrsize / numthr;
  start = thr * (arrsize_float);
  end = thr * (arrsize_float) + (arrsize_float);
  queries[querynum].thrglob.mean_temp[thr] = 0;
  for (i = start; i < end; i++)
  {
    queries[querynum].thrglob.mean_temp[thr] += x[i];
  }

  syncthreads(thr, querynum);
  if (thr == 0)
  {
    (queries[querynum].thrglob.mean) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (queries[querynum].thrglob.mean) += queries[querynum].thrglob.mean_temp[i];
    }
    (queries[querynum].thrglob.mean) /= models[modelnum].WVSIZE;
  }
  syncthreads(thr, querynum);
  // for (i = 0; i < models[modelnum].WVSIZE; i++)
  arrsize = models[modelnum].WVSIZE;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);
  queries[querynum].thrglob.smean_temp[thr] = 0;
  for (i = start; i < end; i++)
  {
    bloom_precision a = x[i] - (queries[querynum].thrglob.mean);
    queries[querynum].thrglob.smean_temp[thr] += a * a;
  }
  syncthreads(thr, querynum);
  if (thr == 0)
  {

    (queries[querynum].thrglob.smean) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (queries[querynum].thrglob.smean) += queries[querynum].thrglob.smean_temp[i];
    }

    (queries[querynum].thrglob.smean) /= models[modelnum].WVSIZE;
    if ((queries[querynum].thrglob.smean) < EPSILON)
    {
      (queries[querynum].thrglob.smean) = EPSILON;
    }
  }
  syncthreads(thr, querynum);
  muller = sqrt(1.0 / ((queries[querynum].thrglob.smean)));
  if (b)
  {
    // for (i = 0; i < models[modelnum].WVSIZE; i++)
    arrsize = models[modelnum].WVSIZE;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    for (i = start; i < end; i++)
    {
      o[i] = (x[i] - (queries[querynum].thrglob.mean)) * muller * g[i] + b[i];
    }
  }
  else
  {
    // for (i = 0; i < models[modelnum].WVSIZE; i++)
    arrsize = models[modelnum].WVSIZE;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    for (i = start; i < end; i++)
    {
      o[i] = (x[i] - (queries[querynum].thrglob.mean)) * muller * g[i];
    }
  }
  syncthreads(thr, querynum);
}

/**
 * @brief  Is variable a nan?
 * @note   
 * @param  x: 
 * @retval 
 */
int is_nan(bloom_precision x) { return x != x; }

/**
 * @brief  Compute Standard Deviation
 * @note   
 * @param  *data: 
 * @param  n: 
 * @param  *mean: 
 * @param  *variance: 
 * @param  eps: 
 * @retval None
 */
void stddev(const bloom_precision *data, int n, bloom_precision *mean, bloom_precision *variance, bloom_precision eps)
{
  bloom_precision sum = 0;
  bloom_precision mean_l = 0;
  for (int i = 0; i < n; i++)
  {
    sum += data[i];
  }
  mean_l = sum / n;
  bloom_precision variance_l = 0;
  for (int i = 0; i < n; i++)
  {
    variance_l += pow(data[i] - mean_l, 2);
  }
  variance_l = variance_l / n;
  *variance = sqrt(variance_l);
  *mean = mean_l;
  return;
}


/*
Fast integer square root function

The basic idea behind the method is to calculate the exponential of the logarithm of the integer divided by two. 

Time Complexity: O(1), The time complexity of the given approach is O(1) since it uses only one mathematical formula exp(log(x) / 2)
 which is constant time, and a few arithmetic operations, comparisons, and function calls that take constant time as well. Therefore, 
 the time complexity of this algorithm is constant or O(1).

 Space Complexity: O(1), The space complexity of the given approach is O(1) as it only uses a constant amount of extra space.
 It declares two integer variables, result and floorResult, which each take constant space, and there is no dynamic memory 
 allocation or recursive calls. Therefore, the space complexity of this algorithm is constant or O(1).*/
int fast_sqrt_q8(int x)
{
    // using exponential and logarithmic function to
    // calculate square root of x
    double result = exp(log(x) / 2);
    // floor function to get integer part of the result
    int floorResult = floor(result);
 
    // If the integer square of the floor result is equal to
    // the input x,
    // then x is a perfect square, and floor result is the
    // square root.
    if (floorResult * floorResult == x) {
        return floorResult;
    }
    else { // If not, then x is not a perfect square, and
           // floor result is the floor of the square root.
        return floorResult;
    }
}


void stddev_q8(const int8_t *data, int n, int8_t *mean, int8_t *variance, int8_t eps)
{
  int8_t sum = 0;
  int8_t mean_l = 0;
  for (int i = 0; i < n; i++)
  {
    sum += data[i];
  }
  mean_l = sum / n;
  int8_t variance_l = 0;
  for (int i = 0; i < n; i++)
  {
    variance_l += pow(data[i] - mean_l, 2);
  }
  variance_l = variance_l / n;
  *variance = fast_sqrt_q8(variance_l);
  *mean = mean_l;
  return;
}

/**
 * @brief  Layer normalize
 * @note   
 * @param  *o: 
 * @param  *x: 
 * @param  *b: 
 * @param  *g: 
 * @param  eps: 
 * @param  size: 
 * @retval None
 */
void normalize_torch(bloom_precision *o, bloom_precision *x, bloom_precision *b, bloom_precision *g, bloom_precision eps, int size)
{
  int i;
  bloom_precision mean = 0, smean = 0;
  stddev(o, size, &mean, &smean, eps);
  if (b)
    for (i = 0; i < size; i++)
    {
      o[i] = (x[i] - mean) / (smean + eps) * g[i] + b[i];
    }
  else
    for (i = 0; i < size; i++)
      o[i] = (x[i] - mean) / (smean + eps) * g[i];
}

#ifndef DEBUG
inline
#endif
    void
    LayerNormKernelImplInternal(
        const bloom_precision *X,
        const bloom_precision *gamma,
        const bloom_precision *beta,
        int64_t M,
        int64_t N,
        bloom_precision eps,
        bloom_precision *Y)
{

  const bloom_precision *X_data = X;
  const bloom_precision *gamma_data = gamma;
  const bloom_precision *beta_data = beta;
  bloom_precision *Y_data = Y;
  bloom_precision mean_data = 0.0;
  bloom_precision rstd_data = 0.0;
  const bloom_precision c = 1.0 / N;
  const bool gamma_null = gamma_data == NULL;
  const bool beta_null = beta_data == NULL;
  // for (int64_t i = 0; i < M; ++i) {
  const bloom_precision *X_ptr = X_data;
  bloom_precision *Y_ptr = Y_data;
  bloom_precision mean_val = 0.0;
  bloom_precision rstd_val = 0.0;
  for (int64_t j = 0; j < N; ++j)
  {
    mean_val += X_ptr[j];
    rstd_val += X_ptr[j] * X_ptr[j];
  }
  mean_val *= c;
  rstd_val = 1.0 / sqrt(rstd_val * c - mean_val * mean_val + eps);
  const bloom_precision scale = rstd_val;
  const bloom_precision bias = -rstd_val * mean_val;
  for (int64_t j = 0; j < N; ++j)
  {
    const bloom_precision gamma_v = gamma_null ? 1.0 : gamma_data[j];
    const bloom_precision beta_v = beta_null ? 0.0 : beta_data[j];
    Y_ptr[j] = (X_ptr[j] * scale + bias) * gamma_v + beta_v;
  }
  mean_data = mean_val;
  rstd_data = rstd_val;
  //}
}

void normalize_ex_q8(int8_t *o, int8_t *x, int8_t *b, int8_t *g, int8_t eps, long long size, int modelnum, int querynum)
{


  int i;
  int8_t mean = 0, smean = 0;
  stddev_q8(o, size, &mean, &smean, eps);
  if (b)
    for (i = 0; i < size; i++)
    {
      o[i] = (x[i] - mean) / (smean + eps) * g[i] + b[i];
    }
  else
    for (i = 0; i < size; i++)
      o[i] = (x[i] - mean) / (smean + eps) * g[i];

}

#ifndef DEBUG
inline
#endif
    void
    normalize_ex(bloom_precision *o, bloom_precision *x, bloom_precision *b, bloom_precision *g, bloom_precision eps, long long size, int modelnum, int querynum)
{
  LayerNormKernelImplInternal(x, g, b, 1, size, eps, o);
}

#ifndef DEBUG
inline
#endif
    void
    normalize(bloom_precision *o, bloom_precision *x, bloom_precision *b, bloom_precision *g, int modelnum, int querynum)
{
  normalize_ex(o, x, b, g, EPSILON, models[modelnum].WVSIZE, modelnum, querynum);
}

/* globals for multithreading */

#ifdef HAVE_THREADS

void syncthreads2(int thr, int querynum)
{
  if (queries[querynum].thrglob.numthr <= 1)
    return;
  pthread_barrier_wait((pthread_barrier_t *)&queries[querynum].thrglob.barrier);
}

void syncthreads(int thr, int querynum)
{
  if (queries[querynum].thrglob.numthr <= 1)
    return;

#ifdef FAST_BARRIER
  fast_barrier_wait(&queries[querynum].thrglob.fastbarrier);
#else
  pthread_barrier_wait((pthread_barrier_t *)&queries[querynum].thrglob.barrier);
#endif
}
#else
#define syncthreads(int thr)
#endif

/**
 * @brief  matrix multiply column major
 * @note   
 * @param  *A: 
 * @param  *B: 
 * @param  *C: 
 * @param  batch_size: 
 * @param  n: 
 * @param  m: 
 * @param  p: 
 * @retval None
 */
void bmm_3D_col_major(const float *A, const float *B, float *C, int batch_size, int n, int m, int p)
{
  for (int b = 0; b < batch_size; b++)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < p; j++)
      {
        float sum = 0;
        for (int k = 0; k < m; k++)
        {
          sum += A[b * n * m + i * m + k] * B[b * m * p + k * p + j];
        }
        C[b * n * p + i * p + j] = sum;
      }
    }
  }
}

/**
 * @brief  Matrix multiply row major
 * @note   
 * @param  *A: 
 * @param  *B: 
 * @param  *C: 
 * @param  batch_size: 
 * @param  n: 
 * @param  m: 
 * @param  p: 
 * @retval None
 */
void bmm_3D_row_major(const float *A, const float *B, float *C, int batch_size, int n, int m, int p)
{
  for (int i = 0; i < batch_size; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < p; k++)
      {
        C[i * n * p + j * p + k] = 0;
        for (int l = 0; l < m; l++)
        {
          C[i * n * p + j * p + k] += A[i * n * m + j * m + l] * B[i * m * p + l * p + k];
        }
      }
    }
  }
}


/**
 * @brief  Run a layer of the model
 * @note   
 * @param  *x: 
 * @param  layeridx: 
 * @param  here: 
 * @param  thr: 
 * @param  numthr: 
 * @param  modelnum: 
 * @param  querynum: 
 * @retval None
 */
void runLayer(bloom_precision *x, int layeridx, int here, int thr, int numthr, int modelnum, int querynum)
{
  long long start;
  long long end;
  bloom_precision arrsize;
  long long dsz;
  long long dcnt;
  long long sz;
  int modelindex = modelnum;
  int layernum = layeridx;
  int WVSIZE = models[modelnum].WVSIZE;
  int CTXSIZE = models[modelnum].CTXSIZE;
  int HEADSIZE = models[modelnum].HEADSIZE;
  int NUMHEADS = models[modelnum].NUMHEADS;
  int NUMLAYERS = models[modelnum].NUMLAYERS;
  int closest_power_of_2 = models[modelnum].closest_power_of_2;
  bloom_precision RSQRT_HEADSIZE = models[modelnum].RSQRT_HEADSIZE;
  bloom_precision FP16_size = 2.0;
  bloom_precision *att = queries[querynum].att;
  bloom_precision *attentions = queries[querynum].attentions;
  bloom_precision *attentions_presoftmax = queries[querynum].attentions_presoftmax;

#ifdef HAVE_THREADS
  bloom_precision *q = queries[querynum].thrglob.q;
  bloom_precision *tmp = queries[querynum].thrglob.tmp;
  bloom_precision *xn = queries[querynum].thrglob.xn;
  bloom_precision *mlp = queries[querynum].thrglob.mlp;
#else
  bloom_precision q[WVSIZE];             /* q vectors are only needed locally */
  bloom_precision tmp[WVSIZE * CTXSIZE]; /* tmp space for operations */
  bloom_precision xn[WVSIZE];
#endif
  long long i, j, h, k;
  hlayer *l = &models[modelnum].layers[layeridx];

  if (models[modelnum].verbose >= 2)
    fprintf(stderr, "layer %d...\n", layeridx);

#ifdef EXTRACT_WEIGHTS_ON_DEMAND

  sz = models[modelnum].WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].ln1_g == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].ln1_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    // copy values
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln1_g;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].ln1_g[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln1_g;
          models[modelindex].layers[layernum].ln1_g[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }

    if (models[modelnum].use_8bit)
    {
      if (models[modelindex].layers[layernum].q8_ln1_g== NULL)
        models[modelindex].layers[layernum].q8_ln1_g = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln1_g, dcnt, 1, NULL);
    }
  }

  sz = models[modelnum].WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].ln1_b == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].ln1_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln1_b;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].ln1_b[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln1_b;
          models[modelindex].layers[layernum].ln1_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  syncthreads(thr, querynum);
#endif

#ifdef EXPERIMENTAL_THREADED_NORMALIZATION
  normalize_thr(xn, x, l->ln1_b, l->ln1_g, thr);
#else
  if (!thr)
  {
    //printf ("-----start layer----\n");
    stopwatch_start(&begin_glob);    
    normalize(xn, x, l->ln1_b, l->ln1_g, modelnum, querynum);
    stopwatch_end("ln1 normalize", begin_glob);
  }
#endif

#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  syncthreads(thr, querynum);
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].ln1_g != NULL)
    {
      free(models[modelindex].layers[layernum].ln1_g);
      models[modelindex].layers[layernum].ln1_g = NULL;
    }
    if (models[modelindex].layers[layernum].fp16_ln1_b != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_ln1_b);
      models[modelindex].layers[layernum].fp16_ln1_b = NULL;
    }
  }

  sz = ((long long)models[modelnum].WVSIZE) * 3 * WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].attn_cattn_w == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].attn_cattn_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cattn_w;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].attn_cattn_w[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cattn_w;
          models[modelindex].layers[layernum].attn_cattn_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_attn_cattn_w== NULL)
      models[modelindex].layers[layernum].q8_attn_cattn_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cattn_w, dcnt, 1, NULL);
  }  

  sz = ((long long)models[modelnum].WVSIZE) * 3 * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].attn_cattn_b == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].attn_cattn_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cattn_b;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].attn_cattn_b[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cattn_b;
          models[modelindex].layers[layernum].attn_cattn_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_attn_cattn_b== NULL)
      models[modelindex].layers[layernum].q8_attn_cattn_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cattn_b, dcnt, 1, NULL);
  }   

#endif

  syncthreads(thr, querynum);


  /* produce query/key/value vectors for this slot */
  if (thr==0)
  {
    stopwatch_start(&begin_glob);
  }

  if (thr ==0)        
  {
    int numthr_temp = 1;
    bloom_precision *b = l->attn_cattn_b;
    pkdflt *w = (pkdflt *)l->attn_cattn_w;
    bloom_precision *k = l->k;
    bloom_precision *v = l->v;


    long long qi = 0;
    long long kvi = 0;
    long long index = 0;

    arrsize = WVSIZE * 3;
    bloom_precision arrsize_float = arrsize / numthr_temp;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);

    int j = 0;
    int firsttime = 0;
    int mod = 0;
    int WVSIZE_times_i = WVSIZE *start;
    long long WVSIZE_times_here = here * WVSIZE;
    for (i = start; i < end; i++)
    {
      long long i_over_HEADSIZE = (i/HEADSIZE);
      mod = ((i_over_HEADSIZE) % 3);
      qi = ((i_over_HEADSIZE)/3)*HEADSIZE + i % HEADSIZE;
      kvi = WVSIZE_times_here + qi;        

      bloom_precision a = 0;
      if (models[modelnum].use_opencl == 0 || models[modelnum].use_opencl ==3)
      {
        a = conv1dline(b ? b[i] : 0, xn, w + WVSIZE_times_i, WVSIZE);
        //a = conv1dline_thr(b ? b[i] : 0, xn, w + WVSIZE_times_i, WVSIZE, global_numthreads > 2 ? 2 : global_numthreads);
      }
      else if (models[modelnum].use_opencl == 1)
      {

        // do conv1dline on the gpu
        int arraychoice = 0;
        a = conv1dline_cl(b ? b[i] : 0, xn, WVSIZE_times_i, WVSIZE, arraychoice, modelnum, layeridx, thr);     
        
      }
      else if (models[modelnum].use_opencl == 3)
      {

        // use thread pooling to solve dot product
        int arraychoice = 0;

        a = conv1dline_pool(b ? b[i] : 0, xn, w + WVSIZE_times_i, WVSIZE, modelnum, querynum, thr);     
        
      } 

      if (thr ==0)   
      {
        int q=0;
        q++;
      }     

      if (mod == 0)
      {
        // index based off of i to support multithreading
        q[qi] = a;
      }
      else if (mod == 1)
      {
        // index based off of i to support multithreading
        k[kvi] = a;
      }
      else if (mod == 2)
      {
        // index based off of i to support multithreading
        v[kvi] = a;
      }

      WVSIZE_times_i += WVSIZE;
    }
  }
  if (thr==0)
  {
    stopwatch_end("qkv vectors", begin_glob);
  }  


#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  syncthreads(thr, querynum);
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].attn_cattn_w != NULL)
    {
      free(models[modelindex].layers[layernum].attn_cattn_w);
      models[modelindex].layers[layernum].attn_cattn_w = NULL;
    }
    if (models[modelindex].layers[layernum].attn_cattn_b != NULL)
    {
      free(models[modelindex].layers[layernum].attn_cattn_b);
      models[modelindex].layers[layernum].attn_cattn_b = NULL;
    }

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
    if (models[modelindex].layers[layernum].fp16_attn_cattn_w != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_attn_cattn_w);
      models[modelindex].layers[layernum].fp16_attn_cattn_w = NULL;
    }
    if (models[modelindex].layers[layernum].fp16_attn_cattn_b != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_attn_cattn_b);
      models[modelindex].layers[layernum].fp16_attn_cattn_b = NULL;
    }
#endif
  }
#endif

  syncthreads(thr, querynum);

  if (models[modelnum].verbose >= 3)
    fprintf(stderr, "heads...\n");

  if (thr==0)
  {
    stopwatch_start(&begin_glob);
  } 
  {
    bloom_precision *k = l->k;
    long long layeridx_NUMHEADS = layeridx * NUMHEADS;

    arrsize = NUMHEADS;
    bloom_precision arrsize_float = arrsize / numthr;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    long long h_CTXSIZE = start * CTXSIZE;
    long long h_HEADSIZE = start * HEADSIZE;  
    
    for (h = start; h < end; h++)
    {
      long long_closest_power_of_2 = ((long)closest_power_of_2);
      long long_closest_power_of_2_i = 0;      
      int WVSIZE_times_i = 0;  
      /* query * keys = attentions */
      for (i = 0; i <= here; i++)
      {
        bloom_precision a = 0;

        if (models[modelnum].use_opencl == 0 || models[modelnum].use_opencl ==3)
        {
          a = conv1dline(0, &(q[h_HEADSIZE]), &(k[((WVSIZE_times_i) + (h_HEADSIZE))]), HEADSIZE);
        }
        else if (models[modelnum].use_opencl == 1)
        {
          // do conv1dline on the gpu
          int arraychoice = 0;
          a = 0;// conv1dline_cl(b ? b[i] : 0, xn, WVSIZE_times_i, WVSIZE, arraychoice, modelnum, layeridx, thr);     
        }        
        att[h_CTXSIZE + i] = a * RSQRT_HEADSIZE + models[modelnum].alibi[long_closest_power_of_2_i + h];
        attentions_presoftmax[layeridx_NUMHEADS + h] = att[i];
        long_closest_power_of_2_i += long_closest_power_of_2;
        WVSIZE_times_i+= WVSIZE;
      }

      /* softmax attentions to make them sum up to 1.0 */
      bloom_precision max = att[h_CTXSIZE];
      for (i = 1; i <= here; i++)
        if (att[h_CTXSIZE + i] > max)
          max = att[h_CTXSIZE + i];
      bloom_precision sum = 0;
      for (i = 0; i <= here; i++)
      {
        bloom_precision a = exp(att[h_CTXSIZE + i] - max);
        att[h_CTXSIZE + i] = a;
        sum += a;
      }
      bloom_precision sumr = 1.0 / sum;
      for (i = 0; i <= here; i++)
        att[h_CTXSIZE + i] *= sumr;

      h_CTXSIZE += CTXSIZE;
      h_HEADSIZE += HEADSIZE;        
    }
  }
  if (thr==0)
  {
    stopwatch_end("attentions", begin_glob);
  }   

  if (thr==0)
  {
    stopwatch_start(&begin_glob);
  } 
  /* apply attentions to values */
  {
    bloom_precision *l_v = l->v;

    arrsize = models[modelnum].NUMHEADS;
    bloom_precision arrsize_float = arrsize / numthr;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    long long h_CTXSIZE = start * CTXSIZE;
    long long h_HEADSIZE = start * HEADSIZE;     
    for (h = start; h < end; h++)
    {
      for (j = 0; j < HEADSIZE; j++)
      {
        tmp[h_HEADSIZE + j] = 0;
        int WVSIZE_times_i = 0;  
        for (i = 0; i < here + 1; i++)
        {
          tmp[h_HEADSIZE + j] += (*(att + h_CTXSIZE + i)) * (*(l_v + (WVSIZE_times_i + h_HEADSIZE + j)));
          WVSIZE_times_i+=WVSIZE;
        }
      }
      h_CTXSIZE += CTXSIZE;
      h_HEADSIZE += HEADSIZE;       
    }
  }
  if (thr==0)
  {
    stopwatch_end("apply attentions to values", begin_glob);
  } 

  syncthreads(thr, querynum);

  if (models[modelnum].verbose >= 3)
    fprintf(stderr, "project...\n");

#ifdef EXTRACT_WEIGHTS_ON_DEMAND

  sz = ((long long)WVSIZE) * WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].attn_cproj_w == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].attn_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cproj_w;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].attn_cproj_w[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cproj_w;
          models[modelindex].layers[layernum].attn_cproj_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_attn_cproj_w== NULL)
      models[modelindex].layers[layernum].q8_attn_cproj_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_w, dcnt, 1, NULL);
  }    

  sz = WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].attn_cproj_b == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].attn_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cproj_b;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].attn_cproj_b[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_attn_cproj_b;
          models[modelindex].layers[layernum].attn_cproj_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_attn_cproj_b== NULL)
      models[modelindex].layers[layernum].q8_attn_cproj_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_b, dcnt, 1, NULL);
  }      

  syncthreads(thr, querynum);
#endif

  /* projection (WVSIZExWVSIZE) */
  if (thr==0)
  {
    stopwatch_start(&begin_glob);
  }   
  {
    pkdflt *w = (pkdflt *)l->attn_cproj_w;
    bloom_precision *b = l->attn_cproj_b;

    arrsize = WVSIZE;
    bloom_precision arrsize_float = arrsize / numthr;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    long long WVSIZE_i = start * WVSIZE;
    for (i = start; i < end; i++)
    {
      int WVSIZE_times_i = WVSIZE_i;

      if (models[modelnum].use_opencl == 0 || models[modelnum].use_opencl ==3)
      {
        
        x[i] += conv1dline(b ? b[i] : 0, tmp, w + WVSIZE_times_i, WVSIZE);
      }
      else if (models[modelnum].use_opencl == 1)
      {
        // do conv1dline on the gpu
        int arraychoice = 0;
        x[i] = 0;// conv1dline_cl(b ? b[i] : 0, xn, WVSIZE_times_i, WVSIZE, arraychoice, modelnum, layeridx, thr);     
      } 
      WVSIZE_i += WVSIZE;  
    }
  }
  if (thr==0)
  {
    stopwatch_end("projection", begin_glob);
  }   

#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  syncthreads(thr, querynum);
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].attn_cproj_w != NULL)
    {
      free(models[modelindex].layers[layernum].attn_cproj_w);
      models[modelindex].layers[layernum].attn_cproj_w = NULL;
    }
    if (models[modelindex].layers[layernum].attn_cproj_b != NULL)
    {
      free(models[modelindex].layers[layernum].attn_cproj_b);
      models[modelindex].layers[layernum].attn_cproj_b = NULL;
    }
  }

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].fp16_attn_cproj_w != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_attn_cproj_w);
      models[modelindex].layers[layernum].fp16_attn_cproj_w = NULL;
    }
    if (models[modelindex].layers[layernum].fp16_attn_cproj_b != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_attn_cproj_b);
      models[modelindex].layers[layernum].fp16_attn_cproj_b = NULL;
    }
  }
#endif

  sz = models[modelnum].WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].ln2_g == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].ln2_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln2_g;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].ln2_g[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln2_g;
          models[modelindex].layers[layernum].ln2_g[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_ln2_g== NULL)
      models[modelindex].layers[layernum].q8_ln2_g = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_g, dcnt, 1, NULL);
  }

  sz = models[modelnum].WVSIZE * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].ln2_b == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].ln2_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln2_b;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].ln2_b[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_ln2_b;
          models[modelindex].layers[layernum].ln2_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_ln2_b== NULL)
      models[modelindex].layers[layernum].q8_ln2_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_b, dcnt, 1, NULL);
  }     

#endif

  syncthreads(thr, querynum);

/* normalize again */
#ifdef EXPERIMENTAL_THREADED_NORMALIZATION
  normalize_thr(xn, x, l->ln2_b, l->ln2_g, thr);
#else
  if (!thr)
  {
  
    stopwatch_start(&begin_glob);
  
    normalize(xn, x, l->ln2_b, l->ln2_g, modelnum, querynum);
  
    stopwatch_end("normalization ln2", begin_glob);
  
  }
#endif

#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  syncthreads(thr, querynum);
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].ln2_g != NULL)
    {
      free(models[modelindex].layers[layernum].ln2_g);
      models[modelindex].layers[layernum].ln2_g = NULL;
    }

    if (models[modelindex].layers[layernum].ln2_b != NULL)
    {
      free(models[modelindex].layers[layernum].ln2_b);
      models[modelindex].layers[layernum].ln2_b = NULL;
    }
  }

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].fp16_ln2_g != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_ln2_g);
      models[modelindex].layers[layernum].fp16_ln2_g = NULL;
    }
    if (models[modelindex].layers[layernum].fp16_ln2_b != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_ln2_b);
      models[modelindex].layers[layernum].fp16_ln2_b = NULL;
    }
  }
#endif

  sz = ((long long)WVSIZE) * WVSIZE * 4 * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].mlp_cfc_w == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].mlp_cfc_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cfc_w;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].mlp_cfc_w[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cfc_w;
          models[modelindex].layers[layernum].mlp_cfc_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_mlp_cfc_w== NULL)
      models[modelindex].layers[layernum].q8_mlp_cfc_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_w, dcnt, 1, NULL);
  }  

  sz = WVSIZE * 4 * FP16_size;
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  if (models[modelindex].layers[layernum].mlp_cfc_b == NULL)
  {
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      models[modelindex].layers[layernum].mlp_cfc_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
    }
    syncthreads(thr, querynum);
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cfc_b;
      arrsize = dcnt;
      start = thr * (arrsize / numthr);
      start = (start / 2) * 2;
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      end = (end / 2) * 2;
      BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].mlp_cfc_b[start]), end - start);
    }
    else
    {
      if (thr == 0)
      {
        for (long long j = 0; j < dcnt; j++)
        {
          uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cfc_b;
          models[modelindex].layers[layernum].mlp_cfc_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
        }
      }
    }
  }

  if (models[modelnum].use_8bit)
  {
    if (models[modelindex].layers[layernum].q8_mlp_cfc_b== NULL)
      models[modelindex].layers[layernum].q8_mlp_cfc_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, 1, NULL);
  }  

#endif

  syncthreads(thr, querynum);

  if (models[modelnum].verbose >= 3)
    fprintf(stderr, "mlp...\n");

  /* multilayer perceptron (WVSIZE -> WVSIZE*4 -> WVSIZE) */
  if (thr==0)
  {
    stopwatch_start(&begin_glob);
  }   
  {
    pkdflt *w = (pkdflt *)l->mlp_cfc_w;
    bloom_precision *b = l->mlp_cfc_b;
#ifndef HAVE_THREADS
    bloom_precision mlp[WVSIZE * 4];
#endif
    arrsize = WVSIZE * 4;
    bloom_precision arrsize_float = arrsize / numthr;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    long WVSIZE_i = start * WVSIZE;
    for (i = start; i < end; i++)
    {
      int WVSIZE_times_i = WVSIZE_i;
      bloom_precision a = 0;
      if (models[modelnum].use_opencl == 0 || models[modelnum].use_opencl ==3)
      {
        
        a += conv1dline(b ? b[i] : 0, xn, w + WVSIZE_times_i, WVSIZE);
      }
      else if (models[modelnum].use_opencl == 1)
      {
        // do conv1dline on the gpu
        int arraychoice = 0;
        a = 0;// conv1dline_cl(b ? b[i] : 0, xn, WVSIZE_times_i, WVSIZE, arraychoice, modelnum, layeridx, thr);     
      }    
          
      a = a * 0.5 * (1.0 + tanh(0.7978845676080871 * a * (1.0 + 0.044715 * a * a)));
      mlp[i] = a;
      WVSIZE_i += WVSIZE;
    }

  if (thr==0)
  {
    stopwatch_end("multilayer perceptron stage 1", begin_glob);
  } 
#ifdef EXTRACT_WEIGHTS_ON_DEMAND
    syncthreads(thr, querynum);
    if (thr == 0)
    {
      if (models[modelindex].layers[layernum].mlp_cfc_w != NULL)
      {
        free(models[modelindex].layers[layernum].mlp_cfc_w);
        models[modelindex].layers[layernum].mlp_cfc_w = NULL;
      }

      if (models[modelindex].layers[layernum].mlp_cfc_b != NULL)
      {
        free(models[modelindex].layers[layernum].mlp_cfc_b);
        models[modelindex].layers[layernum].mlp_cfc_b = NULL;
      }
    }

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
    if (thr == 0)
    {
      if (models[modelindex].layers[layernum].fp16_mlp_cfc_w != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cfc_w);
        models[modelindex].layers[layernum].fp16_mlp_cfc_w = NULL;
      }
      if (models[modelindex].layers[layernum].fp16_mlp_cfc_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cfc_b);
        models[modelindex].layers[layernum].fp16_mlp_cfc_b = NULL;
      }
    }
#endif

    sz = ((long long)WVSIZE) * WVSIZE * 4 * FP16_size;
    dsz = sz * FP16_size;
    dcnt = sz / FP16_size;
    if (models[modelindex].layers[layernum].mlp_cproj_w == NULL)
    {
      syncthreads(thr, querynum);
      if (thr == 0)
      {
        models[modelindex].layers[layernum].mlp_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
      }
      syncthreads(thr, querynum);
      if (models[modelnum].use_bfloat16)
      {
        uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cproj_w;
        arrsize = dcnt;
        start = thr * (arrsize / numthr);
        start = (start / 2) * 2;
        end = thr * (arrsize / numthr) + (arrsize / numthr);
        end = (end / 2) * 2;
        BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].mlp_cproj_w[start]), end - start);
      }
      else
      {
        if (thr == 0)
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cproj_w;
            models[modelindex].layers[layernum].mlp_cproj_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }
    }

    if (models[modelnum].use_8bit)
    {
      if (models[modelindex].layers[layernum].q8_mlp_cfc_b== NULL)
        models[modelindex].layers[layernum].q8_mlp_cfc_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, 1, NULL);
    }    

    sz = ((long long)WVSIZE); //* 4 * FP16_size;
    dsz = sz * FP16_size;
    dcnt = sz / FP16_size;
    if (models[modelindex].layers[layernum].mlp_cproj_b == NULL)
    {
      syncthreads(thr, querynum);
      if (thr == 0)
      {
        models[modelindex].layers[layernum].mlp_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
      }
      syncthreads(thr, querynum);
      if (models[modelnum].use_bfloat16)
      {
        uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cproj_b;
        arrsize = dcnt;
        start = thr * (arrsize / numthr);
        start = (start / 2) * 2;
        end = thr * (arrsize / numthr) + (arrsize / numthr);
        end = (end / 2) * 2;
        BFloat16ToFloat((uint16_t *)&(ptr[start]), &(models[modelindex].layers[layernum].mlp_cproj_b[start]), end - start);
      }
      else
      {
        if (thr == 0)
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *) models[modelindex].layers[layernum].fp16_mlp_cproj_b;
            models[modelindex].layers[layernum].mlp_cproj_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }
    }

    if (models[modelnum].use_8bit)
    {
      if (models[modelindex].layers[layernum].q8_mlp_cfc_b== NULL)
        models[modelindex].layers[layernum].q8_mlp_cfc_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, 1, NULL);
    }    

#endif
    syncthreads(thr, querynum);

    if (thr==0)
    {
      stopwatch_start(&begin_glob);
    }     

    long long WVSIZE_4 = WVSIZE * 4;
    w = (pkdflt *)l->mlp_cproj_w;
    b = l->mlp_cproj_b;
    arrsize = WVSIZE;
    arrsize_float = arrsize / numthr;
    start = thr * (arrsize_float);
    end = thr * (arrsize_float) + (arrsize_float);
    int WVSIZE_4_times_i = start * WVSIZE_4;
    for (i = start; i < end; i++)
    {
      if (models[modelnum].use_opencl == 0 || models[modelnum].use_opencl ==3)
      {
        
        x[i] += conv1dline(b ? b[i] : 0, mlp, w + WVSIZE_4_times_i, WVSIZE_4);      
      }
      else if (models[modelnum].use_opencl == 1)
      {
        // do conv1dline on the gpu
        int arraychoice = 0;
        x[i] =0;
      }    
      WVSIZE_4_times_i += WVSIZE_4;                
      
    }
  }

  if (thr==0)
  {
    stopwatch_end("multilayer perceptron stage 2", begin_glob);
    //printf ("-----end layer----\n\n\n");
  }     


#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  syncthreads(thr, querynum);
  if (thr == 0)
  {
    if (models[modelindex].layers[layernum].mlp_cproj_w != NULL)
    {
      free(models[modelindex].layers[layernum].mlp_cproj_w);
      models[modelindex].layers[layernum].mlp_cproj_w = NULL;
    }
    if (models[modelindex].layers[layernum].mlp_cproj_b != NULL)
    {
      free(models[modelindex].layers[layernum].mlp_cproj_b);
      models[modelindex].layers[layernum].mlp_cproj_b = NULL;
    }

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
    if (models[modelindex].layers[layernum].fp16_mlp_cproj_w != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_mlp_cproj_w);
      models[modelindex].layers[layernum].fp16_mlp_cproj_w = NULL;
    }
    if (models[modelindex].layers[layernum].fp16_mlp_cproj_b != NULL)
    {
      free(models[modelindex].layers[layernum].fp16_mlp_cproj_b);
      models[modelindex].layers[layernum].fp16_mlp_cproj_b = NULL;
    }
#endif
  }
  syncthreads(thr, querynum);
#endif
}

void *perthread(void *args);

int unloadtimes = 0;



int GetWTE_Plus_Positional_Salt_q8(int slot, int modelnum, int querynum, int WVSIZE,
                            int8_t *x)
{
  /* get the token's wordvector (wte) + positional salt (wpe) */
  long long tok =
      slot < 0 ? models[modelnum].emptytoken : queries[querynum].context[slot];
  if (tok > models[modelnum].numtokens)
  {
    tok = 1;
  }
  int8_t *wv = getwv_q8(tok, modelnum);
  if (slot < 0)
    slot = 0;

  memcpy(x, wv, sizeof(int8_t) * WVSIZE);    

  return slot;
}

int GetWTE_Plus_Positional_Salt(int slot, int modelnum, int querynum, int WVSIZE,
                            bloom_precision *x)
{
  /* get the token's wordvector (wte) + positional salt (wpe) */
  long long tok =
      slot < 0 ? models[modelnum].emptytoken : queries[querynum].context[slot];
  if (tok > models[modelnum].numtokens)
  {
    tok = 1;
  }
  wte_t *wv = getwv(tok, modelnum);
  if (slot < 0)
    slot = 0;

  memcpy(x, wv, sizeof(bloom_precision) * WVSIZE);


  return slot;
}


void Setup_AliBi_Matrix(int querynum, int CTXSIZE, int slot,
		int closest_power_of_2, int modelnum) {

    // setup alibi matrix based on attention
    memset( models[modelnum].attention_arrange_tensor, 0,
        sizeof(bloom_precision) * CTXSIZE);
    memset( models[modelnum].attention_mask, 0,
        sizeof(bloom_precision) * CTXSIZE);
    for (long long k = 0; k < CTXSIZE; k++) {
       models[modelnum].attention_mask[k] = 1;
    }
    bloom_precision cumulative_attention_mask_sum = 0;
    for (long long k = 0; k < CTXSIZE; k++) {
      cumulative_attention_mask_sum +=  models[modelnum].attention_mask[k];
       models[modelnum].attention_arrange_tensor[k] =
          (cumulative_attention_mask_sum - 1);
    }
    for (long long k = 0; k < CTXSIZE; k++) {
      for (long long j = 0; j < closest_power_of_2; j++) {
        models[modelnum].alibi[k * (int) closest_power_of_2 + j] = pow(
            models[modelnum].base, (bloom_precision) (j + 1))
            *  models[modelnum].attention_arrange_tensor[k];
      }
    }
  
}

void runModel(bloom_precision *x, int slot, int modelnum, int querynum)
{
  struct timespec token_time;
  stopwatch_start(&token_time);

  long long dsz;
  long long dcnt;
  long long sz;
  bloom_precision FP16_size = 2.0;
  int modelindex = modelnum;
  int WVSIZE = models[modelnum].WVSIZE;
  int CTXSIZE = models[modelnum].CTXSIZE;
  int closest_power_of_2 = models[modelnum].closest_power_of_2;
  int HEADSIZE = models[modelnum].HEADSIZE;
  int NUMHEADS = models[modelnum].NUMHEADS;
  int NUMLAYERS = models[modelnum].NUMLAYERS;
  long long i, j;
  int8_t *q8_x = NULL;
  int8_t *q8_xn = NULL;
  int16_t *q16_x = NULL;
  int16_t *q16_xn = NULL;  

#ifdef HAVE_THREADS
  queries[querynum].thrglob.numthr = models[modelnum].numthreads;
#endif

#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  if (models[modelindex].wte == NULL)
  {
    sz = WVSIZE * (((long long)models[modelnum].numwtetokens) + NUMUSERTOKENS) * FP16_size;
    fflush(stdout);
    dsz = sz * FP16_size;
    dcnt = sz / FP16_size;
    models[modelindex].wte = (wte_t *)malloc(sizeof(bloom_precision) * (dcnt + MAXUSERTOKENS * WVSIZE));
    memset(models[modelindex].wte, 0, sizeof(bloom_precision) * (dcnt + MAXUSERTOKENS * WVSIZE));
    if (models[modelnum].use_bfloat16)
    {
      uint16_t *ptr = (uint16_t *) models[modelindex].fp16_wte;
      BFloat16ToFloat((uint16_t *)ptr, models[modelindex].wte, dcnt);
    }
    else
    {
      for (long long i = 0; i < dcnt; i++)
      {
        uint16_t *ptr = (uint16_t *) models[modelindex].fp16_wte;
        models[modelindex].wte[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
      }
    }

    if (models[modelindex].use_8bit==true)
    {
      if (models[modelindex].q8_wte==NULL)
        models[modelindex].q8_wte = convert1dfloatarrayto8bit(models[modelindex].wte, dcnt, 1, NULL);
    }        
  }

#endif
  //printf ("\n\n==========start token===========");
  //stopwatch_start(&begin_glob);    
  /* get the token's wordvector (wte) + positional salt (wpe) */
  slot = GetWTE_Plus_Positional_Salt(slot, modelnum, querynum, WVSIZE, x);  
	//stopwatch_end("word tokenization embedding", begin_glob);  

#ifdef EXTRACT_WEIGHTS_ON_DEMAND
  if (models[modelindex].wte != NULL)
  {
    free(models[modelindex].wte);
    models[modelindex].wte = NULL;
  }
#endif

  //stopwatch_start(&begin_glob);    

  normalize_ex(x, x, models[modelnum].welb, models[modelnum].welw, 1e-5, WVSIZE, modelnum, querynum);

  //stopwatch_end("word embedding normalization", begin_glob);  

#ifdef HAVE_THREADS
  /* alloc memory for some variables we can't keep local when threaded */
  if (!queries[querynum].thrglob.q)
    queries[querynum].thrglob.q = malloc(WVSIZE * sizeof(bloom_precision));
  if (!queries[querynum].thrglob.tmp)
    queries[querynum].thrglob.tmp = malloc(WVSIZE * sizeof(bloom_precision));
  if (!queries[querynum].thrglob.xn)
    queries[querynum].thrglob.xn = malloc(WVSIZE * sizeof(bloom_precision));
  if (!queries[querynum].thrglob.mlp)
    queries[querynum].thrglob.mlp = malloc(WVSIZE * 4 * sizeof(bloom_precision));
#endif


  if (models[modelnum].use_opencl == 2)
  {
    // run through all layers
    runAllLayers_cl(x, slot, modelnum, querynum);
  }
  else
  {

    #ifdef MEASURE_ALL_LAYERS_TIME
      clock_t run_all_t;
      run_all_t = clock();
    #endif

    #ifdef HAVE_THREADS
      if (models[modelnum].numthreads <= 1)
      {
    #endif
        for (i = 0; i < NUMLAYERS; i++)
        {
    #ifdef LOAD_WEIGHTS_ON_DEMAND
    #ifdef MEASURE_LOAD_TIME
          struct timespec load_t_end;
          struct timespec load_t_start;
          clock_gettime(CLOCK_REALTIME, &load_t_start);
    #endif

          load_layer_container(modelnum, i);
    #ifdef MEASURE_LOAD_TIME
          clock_gettime(CLOCK_REALTIME, &load_t_end);
          double t_ns = (double)(load_t_end.tv_sec - load_t_start.tv_sec) * 1.0e9 +
                        (double)(load_t_end.tv_nsec - load_t_start.tv_nsec);

          fprintf(stderr, "load_layer_container(%d,%lld): elapsed time %fs\n", modelnum, i, t_ns / 1.0e9);
          fflush(stderr);

    #endif

    #endif
          if (models[modelnum].verbose >= 2)
            fprintf(stderr, "layer %lld\n", i);

    #ifdef MEASURE_RUN_TIME

          struct timespec run_t_end;
          struct timespec run_t_start;
          clock_gettime(CLOCK_REALTIME, &run_t_start);

    #endif
          
          runLayer(x, i, slot, 0, 1, modelnum, querynum);

    #ifdef MEASURE_RUN_TIME

          clock_gettime(CLOCK_REALTIME, &run_t_end);
          double t2_ns = (double)(run_t_end.tv_sec - run_t_start.tv_sec) * 1.0e9 +
                        (double)(run_t_end.tv_nsec - run_t_start.tv_nsec);
          fprintf(stderr, "runLayer(x,%lld,%d,0,1,%d,%d):  elapsed time %fs\n", i, slot, modelnum, querynum, t2_ns / 1.0e9);
          fflush(stderr);
    #endif

    #ifdef UNLOAD_WEIGHTS_NOT_IN_USE
          unload_layer_container(modelnum, i);
    #endif
        }

    #ifdef MEASURE_ALL_LAYERS_TIME
        run_all_t = clock() - run_all_t;
        double run_all_time_taken = ((double)run_all_t) / CLOCKS_PER_SEC; // in seconds

        fprintf(stderr, "runLayer all layers: %fs \n", i, run_all_time_taken);
        fflush(stderr);
    #endif
    #ifdef HAVE_THREADS
      }
      else
      {
        if (models[modelnum].numthreads > MAXNUMTHR)
        {
          models[modelnum].numthreads = MAXNUMTHR;
        }
        queries[querynum].thrglob.x = x;
        queries[querynum].thrglob.slot = slot;
        queries[querynum].thrglob.numthr = models[modelnum].numthreads;
        model_thread_args_t thread_args[models[modelnum].numthreads];
        pthread_barrier_init((pthread_barrier_t *)&queries[querynum].thrglob.barrier, NULL, models[modelnum].numthreads);
        fast_barrier_init(&queries[querynum].thrglob.fastbarrier, &queries[querynum].thrglob.fastbarrierattributes, queries[querynum].thrglob.numthr);

        for (i = 0; i < models[modelnum].numthreads; i++)
        {
          thread_args[i].thr = i;
          thread_args[i].modelnum = modelnum;
          thread_args[i].querynum = querynum;
          pthread_create((pthread_t *)&queries[querynum].thrglob.t[i], NULL, perthread, &thread_args[i]);
        }
        for (i = 0; i < models[modelnum].numthreads; i++)
          pthread_join(queries[querynum].thrglob.t[i], NULL);
        pthread_barrier_destroy((pthread_barrier_t *)&queries[querynum].thrglob.barrier);
        fast_barrier_destroy(&queries[querynum].thrglob.fastbarrier);
      }
    #endif
  }
  /* normalize the final result */

  //stopwatch_start(&begin_glob);      
  normalize_ex(x, x, models[modelnum].lnf_b, models[modelnum].lnf_g, 1e-5, WVSIZE, modelnum, querynum);
  //stopwatch_end("normalization final", begin_glob);  

  /* cache it if cache is present */
  if (models[modelnum].outputcache && slot)
    memcpy(models[modelnum].outputcache + WVSIZE * slot, x, WVSIZE * sizeof(bloom_precision));

  //printf ("==========end token===========\n\n");

  //printf ("Measured time: %lf\n", total_elapsed_measured);
  stopwatch_end("Token Generation Time", token_time);  

}

#ifdef HAVE_THREADS
/**
 * @brief  Run layers per thread
 * @note   
 * @param  *args: 
 * @retval None
 */
void *perthread(void *args)
{

  model_thread_args_t *thr_args = ((model_thread_args_t *)args);
  int thr = thr_args->thr;
  int querynum = thr_args->querynum;
  int modelnum = thr_args->modelnum;

  int WVSIZE = models[modelnum].WVSIZE;
  int CTXSIZE = models[modelnum].CTXSIZE;
  int closest_power_of_2 = models[modelnum].closest_power_of_2;
  int HEADSIZE = models[modelnum].HEADSIZE;
  int NUMHEADS = models[modelnum].NUMHEADS;
  int NUMLAYERS = models[modelnum].NUMLAYERS;

  int i;
  int s;

  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  /* Set affinity mask to include CPUs 0 to 7 */

  CPU_ZERO(&cpuset);
  CPU_SET(thr, &cpuset);

  s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

#ifdef MEASURE_ALL_LAYERS_TIME
  clock_t run_all_t;
  if (thr == 0)
  {
    run_all_t = clock();
  }
#endif

  if (models[modelnum].verbose >= 2)
    fprintf(stderr, "thread %d started\n", thr);
  for (i = 0; i < NUMLAYERS; i++)
  {
    // // experimental, skip layers
    // int skip = NUMLAYERS+1;
    // skip = NUMLAYERS;
    // if (((i+queries[querynum].thrglob.slot) % skip) == 0)
    //   continue;

    syncthreads(thr, querynum);
#ifdef LOAD_WEIGHTS_ON_DEMAND
    struct timespec load_t_end;
    struct timespec load_t_start;
    if (thr == 0)
    {
#ifdef MEASURE_LOAD_TIME
      clock_gettime(CLOCK_REALTIME, &load_t_start);
#endif
    }
    load_layer_container_thr(modelnum, i, thr);
    syncthreads(thr, querynum);
#ifdef MEASURE_LOAD_TIME
    if (thr == 0)
    {
      clock_gettime(CLOCK_REALTIME, &load_t_end);
      double t_ns = (double)(load_t_end.tv_sec - load_t_start.tv_sec) * 1.0e9 +
                    (double)(load_t_end.tv_nsec - load_t_start.tv_nsec);

      fprintf(stderr, "load_layer_container(%d,%d): elapsed time %fs \n", modelnum, i, t_ns / 1.0e9);
      fflush(stderr);
    }
#endif
    
#endif

#ifdef MEASURE_RUN_TIME
    clock_t run_t;
    struct timespec run_t_end;
    struct timespec run_t_start;
    if (thr == 0)
    {
      run_t = clock();
      clock_gettime(CLOCK_REALTIME, &run_t_start);
    }
#endif

    runLayer(queries[querynum].thrglob.x, i, queries[querynum].thrglob.slot, thr, queries[querynum].thrglob.numthr, modelnum, querynum);

#ifdef MEASURE_RUN_TIME
    if (thr == 0)
    {
      clock_gettime(CLOCK_REALTIME, &run_t_end);
      double t2_ns = (double)(run_t_end.tv_sec - run_t_start.tv_sec) * 1.0e9 +
                     (double)(run_t_end.tv_nsec - run_t_start.tv_nsec);
      fprintf(stderr, "runLayer(x,%d,%d,0,1):  elapsed time %fs \n", i, queries[querynum].thrglob.slot, t2_ns / 1.0e9);
      fflush(stderr);
    }
#endif

    syncthreads(thr, querynum);

#ifdef UNLOAD_WEIGHTS_NOT_IN_USE
    if (thr == 0)
    {
      unload_layer_container(modelnum, i);
    }
#endif
    syncthreads(thr, querynum);
  }

#ifdef MEASURE_ALL_LAYERS_TIME
  if (thr == 0)
  {
    run_all_t = clock() - run_all_t;
    double run_all_time_taken = ((double)run_all_t) / CLOCKS_PER_SEC; // in seconds

    fprintf(stderr, "runLayer all layers: %fs \n", i, run_all_time_taken);
    fflush(stderr);
  }
#endif

  if (models[modelnum].verbose >= 2)
    fprintf(stderr, "thread %d finished\n", thr);
}
#endif


/**
 * @brief  Perform a linear transform
 * @note   
 * @param  *input: 
 * @param  *output: 
 * @param  *weights: 
 * @param  *bias: 
 * @param  input_size: 
 * @param  output_size: 
 * @retval None
 */
void linear_transform(bloom_precision *input, bloom_precision *output, bloom_precision *weights, bloom_precision *bias, long long input_size, long long output_size)
{
  long long i, j;
  for (i = 0; i < output_size; i++)
  {
    output[i] = 0;
    if (bias != NULL)
      output[i] = bias[i];
    for (j = 0; j < input_size; j++)
    {
      output[i] += input[j] * weights[i * input_size + j];
    }
  }
}

typedef struct lm_logit_t
{

  bloom_precision *input;
  bloom_precision *output;
  bloom_precision *weights;
  bloom_precision *bias;
  long long input_size;
  long long output_size;
  int thr;
  int numthr;
  pthread_t pid;

} lm_logit_t;


void *lt_thr(void *thread_args)
{
  lm_logit_t *logits = (lm_logit_t *)thread_args;

  long long arrsize = logits->output_size;
  bloom_precision arrsize_float = arrsize / logits->numthr;
  long long start = logits->thr * (arrsize_float);
  long long end = logits->thr * (arrsize_float) + (arrsize_float);
    
  long long i, j;
  for (i = start; i < end; i++)
  {
    long i_times_input_size = i * logits->input_size;
    // testing speed
    // if (i%30!=0)
    //   continue;
    logits->output[i] = 0;
    if (logits->bias != NULL)
      logits->output[i] = logits->bias[i];
    for (j = 0; j < logits->input_size; j++)
    {
      logits->output[i] += logits->input[j] * logits->weights[i_times_input_size + j];
    }
  }
}

void linear_transform_thr(bloom_precision *input, bloom_precision *output, bloom_precision *weights, bloom_precision *bias, long long input_size, long long output_size, int numthr)
{
  int i;
  lm_logit_t *thread_args = malloc(sizeof(lm_logit_t)*numthr);
  for (i = 0; i < numthr; i++)
  {
    thread_args[i].thr = i;
    thread_args[i].input = input;
    thread_args[i].output = output;
    thread_args[i].weights = weights;
    thread_args[i].bias = bias;
    thread_args[i].input_size = input_size;
    thread_args[i].output_size = output_size;
    thread_args[i].numthr = numthr;

    pthread_create((pthread_t *)&thread_args[i].pid, NULL, lt_thr, &thread_args[i]);
  }
  for (i = 0; i < numthr; i++)
    pthread_join(thread_args[i].pid, NULL);  

  free(thread_args);
}




typedef struct conv1dline_t
{

  bloom_precision a;
  bloom_precision *v;
  bloom_precision *m;
  long long wdt;
  int thr;
  int numthr;
  pthread_t pid;

} conv1dline_t;

void *conv1dline_thr_proc(void *thread_args)
{
  conv1dline_t *conv1dline = (conv1dline_t *)thread_args;
  
  long long i;

  long long arrsize = conv1dline->wdt;
  long long start = conv1dline->thr * (arrsize / conv1dline->numthr);
  long long end = conv1dline->thr * (arrsize / conv1dline->numthr) + (arrsize / conv1dline->numthr);
  long long j;

  conv1dline->a = 0;
  for (i = start; i < end; i++)
  {
    conv1dline->a += conv1dline->v[i] * conv1dline->m[i];
  }
}





void conv1dline_pool_work(void *worker_args)
{

  threadpool_worker_data_t *conv1dline = (threadpool_worker_data_t *)worker_args;
  
  long long i;
  
  float a = 0;
  for (i = 0; i < conv1dline->size; i++)
  {
    a += conv1dline->v[conv1dline->v_offset+i] * conv1dline->m[conv1dline->m_offset+i];
  }  
  conv1dline->a = a;  
}

float conv1dline_pool(float a, float *v, float *m, int size,  int modelnum, int querynum, int thr)
{
    int numthreads = global_numthreadpool;
    int worksize = size/numthreads;
    int offset = 0;
    threadpool_worker_data_t *worker_args = malloc(sizeof(threadpool_worker_data_t)*global_numthreadpool);
    for (int i=0;i<numthreads;i++)
    {
      worker_args[i].a = 0;
      worker_args[i].v = v;
      worker_args[i].m = m;
      worker_args[i].v_offset = offset;
      worker_args[i].m_offset = offset;
      worker_args[i].size = worksize;
      if (i==numthreads-1)
      {
        worker_args[i].size = size - offset;
      }

      thpool_add_work(queries[modelnum].thpool, conv1dline_pool_work, (void*)&(worker_args[i]));
      offset+= worksize;
    }
    thpool_wait(queries[modelnum].thpool);

    for (int i=0;i<numthreads;i++)
    {
      a+=(worker_args[i].a);
    }    

    free(worker_args);
    return a;
}



bloom_precision conv1dline_thr(bloom_precision a, bloom_precision *v, bloom_precision *m, long long wdt, int numthr)
{

  float a_ret = a;
  int i;
  conv1dline_t *thread_args = malloc(sizeof(conv1dline_t)*numthr);

  for (i = 0; i < numthr; i++)
  {
    thread_args[i].thr = i;
    thread_args[i].a = 0;
    thread_args[i].v= v;
    thread_args[i].m = m;
    thread_args[i].wdt = wdt;
    thread_args[i].numthr = numthr;
    pthread_create((pthread_t *)&thread_args[i].pid, NULL, conv1dline_thr_proc, &thread_args[i]);
  }
  for (i = 0; i < numthr; i++)
  {
    pthread_join(thread_args[i].pid, NULL);  
    a_ret += thread_args[i].a;
  }

  free(thread_args);
  return a_ret;
}