float conv1dline(float a, __global float *v, __global float *m, long wdt)
{
  long i;
  for (i = 0; i < wdt; i++)
  {
    a += v[i] * m[i];
  }
  return a;
}

void normalize_cl_thr(__global float *xn, __global float *x,  __global float *b, __global float *g, float eps,  long size, int thr, int numthr, __global float *scratch)
{

  long start;
  long end;
  float arrsize;

  long i;
  float muller;
  float a = 0;

  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);

  scratch[thr+2] = 0;
  for (i = start; i < end; i++)
  {
    scratch[thr+2] += x[i];
  }

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  if (thr == 0)
  {
    (scratch[0]) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (scratch[0]) += scratch[i+2];
    }
    (scratch[0]) /= size;
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);

  scratch[thr+3] = 0;
  for (i = start; i < end; i++)
  {
    scratch[thr+3] += x[i] * x[i];
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  if (thr == 0)
  {

    (scratch[1]) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (scratch[1]) += scratch[i+3];
    }
  }  

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

  float mean_val = scratch[0];
  float rstd_val = 1.0 / sqrt(scratch[1] /((float)size) - (mean_val) * (mean_val) + (eps));
  float scale = (rstd_val);
  float bias = -(rstd_val) * (mean_val);

  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);

  for (i = start; i < end; i++)
  {
    float gamma_v = g[i];
    float beta_v = b[i];      
    xn[i] = (x[i] * scale + bias) * gamma_v + beta_v;  
  }

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

}


__kernel void layer_cl(
    __global float *x,
    __global float *xn,
    __global float *y,
    unsigned int WVSIZE,
    __global float *s_ln1_b,
    __global float *s_ln1_g,
    __global float *s_ln2_b,
    __global float *s_ln2_g,
    __global float *s_mlp_cfc_b,
    __global float *s_mlp_cfc_w,
    __global float *s_mlp_cproj_b,
    __global float *s_mlp_cproj_w,
    __global float *s_attn_cattn_b,
    __global float *s_attn_cattn_w,
    __global float *s_attn_cproj_b,
    __global float *s_attn_cproj_w,
    __global float *att,
    __global float *attentions,
    __global float *attentions_presoftmax,
    __global float *alibi,
    __global float *tmp,
    __global float *q,    
    __global float *k,
    __global float *v,
    float closest_power_of_2,
    unsigned int CTXSIZE,
    unsigned int HEADSIZE,
    unsigned int NUMHEADS,
    unsigned int NUMLAYERS,
    unsigned int layeridx,
    unsigned int here,
__global float *scratch
    ) 
{
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);
  int grp_id = get_group_id(0);

  int numthr = l_size;
  int thr = l_id;
  long i;
  long h;


  
  if (y[0] == 0)
  {
    normalize_cl_thr(xn, x, s_ln1_b, s_ln1_g, 0.00001, WVSIZE, thr, numthr, scratch);
    return;
  }

  if (y[0] == 1)
  {
    /* produce query/key/value vectors for this slot */

    {
      float *b = s_attn_cattn_b;
      float *w = (float *)s_attn_cattn_w;

      long vi = 0;
      long ki = 0;
      long qi = 0;
      long kvi = 0;
      
      float arrsize = WVSIZE * 3;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);

      int j = 0;
      int firsttime = 0;
      int mod = 0;
      int WVSIZE_times_i;
      for (i = start; i < end; i++)
      {

        if (firsttime == 0)
        {
          WVSIZE_times_i = WVSIZE * i;
          long i_over_HEADSIZE = (i/HEADSIZE);
          mod = ((i_over_HEADSIZE) % 3);
          qi = ((i_over_HEADSIZE)/3)*HEADSIZE + i % HEADSIZE;
          kvi = here * WVSIZE + ((i_over_HEADSIZE)/3)*HEADSIZE+ i % HEADSIZE;
        }
                
        float a = s_attn_cattn_b[i];
        
        a = conv1dline(a, xn, &(s_attn_cattn_w[WVSIZE_times_i]), WVSIZE);
        
        if (mod == 0)
        {
          // index based off of i to support multithreading
          q[qi] = a;
          qi++;
        }
        else if (mod == 1)
        {
          // index based off of i to support multithreading
          k[kvi] = a;
          //ki++;
        }
        else if (mod == 2)
        {
          // index based off of i to support multithreading
          v[kvi] = a;
          //vi++;
          kvi++;
        }

        mod++;
        if (mod>3)
          mod = 0;

      }
    }
    return;
  }


  if (y[0] == 2)
  {
    float RSQRT_HEADSIZE = (1.0 / sqrt((float)HEADSIZE));
    long layeridx_NUMHEADS = layeridx * NUMHEADS;
    float arrsize = NUMHEADS;
    float arrsize_over_numthr = arrsize /  numthr;
    long start = thr * (arrsize_over_numthr);
    long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);
    for (h = start; h < end; h++)
    {

      long h_CTXSIZE = h * CTXSIZE;
      long h_HEADSIZE = h * HEADSIZE;
      /* query * keys = attentions */
      for (i = 0; i <= here; i++)
      {
        float a = conv1dline(0, &(q[h_HEADSIZE]), &(k[((i * WVSIZE) + (h_HEADSIZE))]), HEADSIZE);
        att[h_CTXSIZE + i] = a * RSQRT_HEADSIZE + alibi[((long)closest_power_of_2) * i + h];
        attentions_presoftmax[layeridx_NUMHEADS + h] = att[i];
      }

      /* softmax attentions to make them sum up to 1.0 */
      float max = att[h_CTXSIZE];
      for (i = 1; i <= here; i++)
        if (att[h_CTXSIZE + i] > max)
          max = att[h_CTXSIZE + i];
      float sum = 0;
      for (i = 0; i <= here; i++)
      {
        float a = exp(att[h_CTXSIZE + i] - max);
        att[h_CTXSIZE + i] = a;
        sum += a;
      }
      float sumr = 1.0 / sum;
      for (i = 0; i <= here; i++)
        att[h_CTXSIZE + i] *= sumr;


    }
    return;
    }

  
  if (y[0] == 3)
  {
    /* apply attentions to values */
    {
      long j;
      float *l_v = v;
      float arrsize = NUMHEADS;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);
      for (h = start; h < end; h++)
      {
        long h_HEADSIZE = h * HEADSIZE;
        long h_CTXSIZE = h * CTXSIZE;
        for (j = 0; j < HEADSIZE; j++)
        {
          tmp[h_HEADSIZE + j] = 0;
          for (i = 0; i < here + 1; i++)
          {
            tmp[h_HEADSIZE + j] += att[h_CTXSIZE + i] * v[i * WVSIZE + h_HEADSIZE + j];
          }
        }
      }
    }
    return;
  }

  if (y[0] == 4)
  {
  /* projection (WVSIZExWVSIZE) */
    {
      float *w = (float *)s_attn_cproj_w;
      float *b = s_attn_cproj_b;
      float arrsize = WVSIZE;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);
      for (i = start; i < end; i++)
      {
        float a = b[i];
        x[i] += conv1dline(a, tmp, &(s_attn_cproj_w[ WVSIZE * i]), WVSIZE);
      }
    }  
    return;
  }

  if (y[0] == 5)
  {
  

    normalize_cl_thr(xn, x, s_ln2_b, s_ln2_g, 0.00001, WVSIZE, thr, numthr, scratch);
  
    return;
  }

  if (y[0] == 6)
  {
    /* multilayer perceptron (WVSIZE -> WVSIZE*4 -> WVSIZE) */
    {
      float *w = (float *)s_mlp_cfc_w;
      float *b = s_mlp_cfc_b;

      float *mlp = tmp;

      float arrsize = WVSIZE * 4;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);

      for (i = start; i < end; i++)
      {
        float a = b[i];
        a = conv1dline(a, xn, &(s_mlp_cfc_w[WVSIZE * i]), WVSIZE);

        a = a * 0.5 * (1.0 + tanh(0.7978845676080871 * a * (1.0 + 0.044715 * a * a)));
        mlp[i] = a;
      }
    }
    return;
  }

  if (y[0] == 7)
  {
    {
     
      long WVSIZE_4 = WVSIZE * 4;
      float *w = (float *)s_mlp_cproj_w;
      float *b = s_mlp_cproj_b;

      float arrsize = WVSIZE;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);
      for (i = start; i < end; i++)
      {
        float a = b[i];
        x[i] += conv1dline(a, tmp, &(s_mlp_cproj_w[WVSIZE_4 * i]), WVSIZE_4);
      }
      
      return;
    }
  }
}
