float conv1dline(float a, __global float *v, __global float *m, long wdt)
{
  long i;
  for (i = 0; i < wdt; i++)
  {
    a += v[i] * m[i];
  }
  return a;
}


// void normalize_cl_thr(__global float *y, __global float *x,  __global float *b, __global float *g, float eps,  long size, int thr, int numthr)
// {

//   long start;
//   long end;
//   float arrsize;

//   long i;
//   float muller;
//   float a = 0;


//   arrsize = size;
//   start = thr * (arrsize / numthr);
//   end = thr * (arrsize / numthr) + (arrsize / numthr);
//   float mean_temp = 0;
//   for (i = start; i < end; i++)
//   {
//     mean_temp += x[i];
//   }

//   barrier(CLK_GLOBAL_MEM_FENCE );
//   if (thr == 0)
//   {
//     (thrglob.mean) = 0;
//     for (int i = 0; i < numthr; i++)
//     {
//       (thrglob.mean) += thrglob.mean_temp[i];
//     }
//     (thrglob.mean) /= size;
//   }
//   barrier(CLK_GLOBAL_MEM_FENCE );

//   arrsize = size;
//   start = thr * (arrsize / numthr);
//   end = thr * (arrsize / numthr) + (arrsize / numthr);
//   float smean_temp = 0;
//   for (i = start; i < end; i++)
//   {
//     float a = x[i] - (thrglob.mean);
//     smean_temp += a * a;
//   }
//   barrier(CLK_GLOBAL_MEM_FENCE );
//   if (thr == 0)
//   {

//     (thrglob.smean) = 0;
//     for (int i = 0; i < numthr; i++)
//     {
//       (thrglob.smean) += thrglob.smean_temp[i];
//     }

//     (thrglob.smean) /= size;
//     if ((thrglob.smean) < eps)
//     {
//       (thrglob.smean) = eps;
//     }
//   }

//   barrier(CLK_GLOBAL_MEM_FENCE );

//   muller = sqrt(1.0 / ((thrglob.smean)));
//   if (b)
//   {
//     // for (i = 0; i < models[modelnum].WVSIZE; i++)
//     arrsize = models[modelnum].WVSIZE;
//     start = thr * (arrsize / numthr);
//     end = thr * (arrsize / numthr) + (arrsize / numthr);
//     for (i = start; i < end; i++)
//     {
//       o[i] = (x[i] - (thrglob.mean)) * muller * g[i] + b[i];
//     }
//   }
//   else
//   {
//     // for (i = 0; i < models[modelnum].WVSIZE; i++)
//     arrsize = models[modelnum].WVSIZE;
//     start = thr * (arrsize / numthr);
//     end = thr * (arrsize / numthr) + (arrsize / numthr);
//     for (i = start; i < end; i++)
//     {
//       o[i] = (x[i] - (thrglob.mean)) * muller * g[i];
//     }
//   }
//   barrier(CLK_GLOBAL_MEM_FENCE );

// }

 void normalize_cl(__global float *y, __global float *x,  __global float *b, __global float *g, float eps,  long size)
 {
  float mean_val;
  float rstd_val;
  for (int i=0;i<size;i++)
  {
    mean_val += x[i];
    rstd_val += x[i] * x[i];
  }
  mean_val /= (float)size;
  rstd_val = 1.0 / sqrt(rstd_val /((float)size) - (mean_val) * (mean_val) + (eps));

  float scale = (rstd_val);
  float bias = -(rstd_val) * (mean_val);

  for (int i=0;i<size;i++)
  {
    float gamma_v = g[i];
    float beta_v = b[i];
    y[i] = (x[i] * scale + bias) * gamma_v + beta_v;  
  }
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
    unsigned int here
    ) {
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);

  // do single threaded first
  int numthr = g_size;
  int thr = g_id;
  long i;
  long h;
  float RSQRT_HEADSIZE = (1.0 / sqrt((float)HEADSIZE));



  // // fp16 save/load example
  // float test = 0.125;
  // half *test2 = s_attn_cattn_w;
  // vstore_half(test, 0, test2);  // save float as fp16
  // test = vload_half(0, test2); // load fp16 into float

  // if (numthr > (WVSIZE/2))
  // {
  //   numthr = (WVSIZE/2);
  // }

  if (g_id < numthr)
  {    
    if (g_id==0)
    {
      normalize_cl(xn, x, s_ln1_b, s_ln1_g, 0.00001, WVSIZE);
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE );

    /* produce query/key/value vectors for this slot */
    {
      float *b = s_attn_cattn_b;
      float *w = (float *)s_attn_cattn_w;

      long vi = 0;
      long ki = 0;
      long qi = 0;


      // for(i=thr;i<models[modelnum].WVSIZE*3;i+=numthr)
      // {
      float arrsize = WVSIZE * 3;
      long start = thr * (arrsize / numthr);
      long end = thr * (arrsize / numthr) + (arrsize / numthr);

      int j = 0;
      long row = (start / HEADSIZE);
      long row_mod_3 = ((start / HEADSIZE) % 3);
      long row_over_3 = row / 3;
      long row_over_3_times_HEADSIZE = row_over_3 * HEADSIZE;
      for (i = start; i < end; i++)
      {
        float a = s_attn_cattn_b[i];
        
        a = conv1dline(s_attn_cattn_b[i], xn, &(s_attn_cattn_w[i]), WVSIZE);

        if (j >= HEADSIZE)
        {
          row++;
          row_mod_3++;
          if (row_mod_3 >= 3)
          {
            row_mod_3 = 0;
            row_over_3++;
            row_over_3_times_HEADSIZE = row_over_3 * HEADSIZE;
          }

          j = 0;
        }

        if ((row_mod_3) == 0)
        {
          // index based off of i to support multithreading
          q[row_over_3_times_HEADSIZE + j] = a;
          qi++;
        }
        else if ((row_mod_3) == 1)
        {
          // index based off of i to support multithreading
          k[here * WVSIZE + row_over_3_times_HEADSIZE + j] = a;
          ki++;
        }
        else if ((row_mod_3) == 2)
        {
          // index based off of i to support multithreading
          v[here * WVSIZE + row_over_3_times_HEADSIZE + j] = a;
          vi++;
        }
        j++;
      }
    }

  barrier(CLK_GLOBAL_MEM_FENCE );




  long layeridx_NUMHEADS = layeridx * NUMHEADS;
  float arrsize = NUMHEADS;
  long start = thr * (arrsize / numthr);
  long end = thr * (arrsize / numthr) + (arrsize / numthr);
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

    /* store attention data for visualization */
    if (attentions)
      for (i = 0; i <= here; i++)
      {
        attentions[layeridx_NUMHEADS + h] = att[h_CTXSIZE + i];
      }
  }

  /* apply attentions to values */
  {
    long j;
    float *l_v = v;
    arrsize = NUMHEADS;
    start = thr * (arrsize / numthr);
    end = thr * (arrsize / numthr) + (arrsize / numthr);
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

  /* projection (WVSIZExWVSIZE) */
  {
    float *w = (float *)s_attn_cproj_w;
    float *b = s_attn_cproj_b;
    arrsize = WVSIZE;
    start = thr * (arrsize / numthr);
    end = thr * (arrsize / numthr) + (arrsize / numthr);
    for (i = start; i < end; i++)
    {
      x[i] += conv1dline(b[i], tmp, &(s_attn_cproj_w[ i]), WVSIZE);
    }
  }  

  barrier(CLK_GLOBAL_MEM_FENCE );

  normalize_cl(xn, x, s_ln2_b, s_ln2_g, 0.00001, WVSIZE);

  barrier(CLK_GLOBAL_MEM_FENCE );

  /* multilayer perceptron (WVSIZE -> WVSIZE*4 -> WVSIZE) */
  {
    float *w = (float *)s_mlp_cfc_w;
    float *b = s_mlp_cfc_b;

    float *mlp = tmp;

    // for(i=thr;i<WVSIZE*4;i+=numthr)
    // {
    arrsize = WVSIZE * 4;
    start = thr * (arrsize / numthr);
    end = thr * (arrsize / numthr) + (arrsize / numthr);
    // int j=0;
    // long long row=start;
    for (i = start; i < end; i++)
    {

      float a = conv1dline(b[i], xn, &(s_mlp_cfc_w[WVSIZE * i]), WVSIZE);

      a = a * 0.5 * (1.0 + tanh(0.7978845676080871 * a * (1.0 + 0.044715 * a * a)));
      // a = 0.5 * a * (1 + tanh(0.7978845676080871 * (a + 0.044715 * a * a * a)));
      mlp[i] = a;
    }

    barrier(CLK_GLOBAL_MEM_FENCE );


    long WVSIZE_4 = WVSIZE * 4;
    w = (float *)s_mlp_cproj_w;
    b = s_mlp_cproj_b;
    // for(i=thr;i<WVSIZE;i+=numthr)
    // {
    arrsize = WVSIZE;
    start = thr * (arrsize / numthr);
    end = thr * (arrsize / numthr) + (arrsize / numthr);
    for (i = start; i < end; i++)
    {
      x[i] += conv1dline(b[i], tmp, &(s_mlp_cproj_w[WVSIZE_4 * i]), WVSIZE_4);
    }
  }

  }
  else
  {

    barrier(CLK_GLOBAL_MEM_FENCE );
    barrier(CLK_GLOBAL_MEM_FENCE );
    barrier(CLK_GLOBAL_MEM_FENCE );
  }
}