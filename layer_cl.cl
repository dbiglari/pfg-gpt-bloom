float conv1dline(float a, __global float *v, __global float *m, long wdt)
{
  long i;
  for (i = 0; i < wdt; i++)
  {
    a += v[i] * m[i];
  }
  return a;
}

void normalize_cl_thr(__global float *y, __global float *x,  __global float *b, __global float *g, float eps,  long size, int thr, int numthr, __global float *tmp)
{

  long start;
  long end;
  float arrsize;

  long i;
  float muller;
  float a = 0;
  float tmp0=tmp[0];
  float tmp1=tmp[1];

  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);
  float mean_temp = tmp[thr+2];
  for (i = start; i < end; i++)
  {
    tmp[thr+2] += x[i];
  }

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  if (thr == 0)
  {
    (tmp[0]) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (tmp[0]) += tmp[i+2];
    }
    (tmp[0]) /= size;
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);
  float smean_temp = tmp[thr+3];
  for (i = start; i < end; i++)
  {
    tmp[thr+3] += x[i] * x[i];
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  if (thr == 0)
  {

    (tmp[1]) = 0;
    for (int i = 0; i < numthr; i++)
    {
      (tmp[1]) += tmp[i+3];
    }
  }  

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

  float mean_val = tmp[0];
  float rstd_val = 1.0 / sqrt(tmp[1] /((float)size) - (mean_val) * (mean_val) + (eps));
  float scale = (rstd_val);
  float bias = -(rstd_val) * (mean_val);

  // for (i = 0; i < models[modelnum].WVSIZE; i++)
  arrsize = size;
  start = thr * (arrsize / numthr);
  end = thr * (arrsize / numthr) + (arrsize / numthr);
  for (i = start; i < end; i++)
  {
    //y[i] = (x[i] - (tmp[0])) * muller * g[i] + b[i];
    float gamma_v = g[i];
    float beta_v = b[i];      
    y[i] = (x[i] * scale + bias) * gamma_v + beta_v;  
  }

  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

  tmp[thr+2] = mean_temp;
  tmp[thr+3] = smean_temp;
  if (thr == 0)
  {
    tmp[0] = tmp0;
    tmp[1] = tmp1;
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
}

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

  // printf ("mean: %lf\n", mean_val);
  // printf ("smean: %lf\n", rstd_val);

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
    ) 
{
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);
  int grp_id = get_group_id(0);

  // do single threaded first
  int numthr = l_size;
  int thr = l_id;
  long i;
  long h;
  float RSQRT_HEADSIZE = (1.0 / sqrt((float)HEADSIZE));

  //numthr = 32;
  //thr = 0;
  
  //printf ("barrier 1 - thr %d numthr  %d\n", thr, numthr);
  if (grp_id > 0)
  {
    printf ("lost thread %d %d\n", thr, grp_id);
    return;
  }
 // if(g_id != 0)
 //  return;
  // if (l_id != 0)
  // {
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  //   return;
  // }

//  if (thr>numthr-1)
//  {
//     //printf ("lost thread %d %d\n", thr, grp_id);
//     return;
//  }

  //printf ("thread: %d\n", thr);
  // // fp16 save/load example
  // float test = 0.125;
  // half *test2 = s_attn_cattn_w;
  // vstore_half(test, 0, test2);  // save float as fp16
  // test = vload_half(0, test2); // load fp16 into float

  
  if (y[0] == 0)
  {
    normalize_cl_thr(xn, x, s_ln1_b, s_ln1_g, 0.00001, WVSIZE, thr, numthr, tmp);
    //if (g_id == 0)
//      normalize_cl(xn, x, s_ln1_b, s_ln1_g, 0.00001, WVSIZE);
    return;
  }

  if (y[0] == 1)
  {
        //printf ("run1\n");
    /* produce query/key/value vectors for this slot */
    //numthr = 16;
    //if (l_id < 16)
    {
      float *b = s_attn_cattn_b;
      float *w = (float *)s_attn_cattn_w;

      long vi = 0;
      long ki = 0;
      long qi = 0;

      float arrsize = WVSIZE * 3;
      long start = thr * (arrsize /  numthr);
      long end = thr * (arrsize / numthr) + (arrsize / numthr);

      int j = 0;
      long row = (start / HEADSIZE);
      long row_mod_3 = ((start / HEADSIZE) % 3);
      long row_over_3 = row / 3;
      long row_over_3_times_HEADSIZE = row_over_3 * HEADSIZE;
      for (i = start; i < end; i++)
      {
        float a = s_attn_cattn_b[i];
        
        a = conv1dline(a, xn, &(s_attn_cattn_w[WVSIZE * i]), WVSIZE);

        if (((i/HEADSIZE) % 3) == 0)
        {
          // index based off of i to support multithreading
          q[((i/HEADSIZE)/3)*HEADSIZE + i % HEADSIZE] = a;
          qi++;
        }
        else if (((i/HEADSIZE) % 3) == 1)
        {
          // index based off of i to support multithreading
          k[here * WVSIZE + ((i/HEADSIZE)/3)*HEADSIZE+ i % HEADSIZE] = a;
          ki++;
        }
        else if (((i/HEADSIZE) % 3) == 2)
        {
          // index based off of i to support multithreading
          v[here * WVSIZE + ((i/HEADSIZE)/3)*HEADSIZE + i % HEADSIZE] = a;
          vi++;
        }

      }
    }
    return;
  }


  if (y[0] == 2)
  {
        //printf ("run2\n");
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
      // if (attentions)
      //   for (i = 0; i <= here; i++)
      //   {
      //     attentions[layeridx_NUMHEADS + h] = att[h_CTXSIZE + i];
      //   }
    }
    return;
  }

  
  if (y[0] == 3)
  {
        //printf ("run3\n");
    long layeridx_NUMHEADS = layeridx * NUMHEADS;
    float arrsize = NUMHEADS;
    long start = thr * (arrsize / numthr);
    long end = thr * (arrsize / numthr) + (arrsize / numthr);    
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
    return;
  }

  if (y[0] == 4)
  {
        //printf ("run4\n");
    long layeridx_NUMHEADS = layeridx * NUMHEADS;
    float arrsize = NUMHEADS;
    long start = thr * (arrsize / numthr);
    long end = thr * (arrsize / numthr) + (arrsize / numthr);        
  /* projection (WVSIZExWVSIZE) */
    {
      float *w = (float *)s_attn_cproj_w;
      float *b = s_attn_cproj_b;
      arrsize = WVSIZE;
      start = thr * (arrsize / numthr);
      end = thr * (arrsize / numthr) + (arrsize / numthr);
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

    normalize_cl_thr(xn, x, s_ln2_b, s_ln2_g, 0.00001, WVSIZE, thr, numthr, tmp);
    //printf ("run5\n");
    //if (g_id == 0)
    //  normalize_cl(xn, x, s_ln2_b, s_ln2_g, 0.00001, WVSIZE);


    return;
  }

  if (y[0] == 6)
  {
    //printf ("run6\n");
    long layeridx_NUMHEADS = layeridx * NUMHEADS;
    float arrsize = NUMHEADS;
    long start = thr * (arrsize / numthr);
    long end = thr * (arrsize / numthr) + (arrsize / numthr);    
    /* multilayer perceptron (WVSIZE -> WVSIZE*4 -> WVSIZE) */
    {
      float *w = (float *)s_mlp_cfc_w;
      float *b = s_mlp_cfc_b;

      float *mlp = tmp;

      arrsize = WVSIZE * 4;
      start = thr * (arrsize / numthr);
      end = thr * (arrsize / numthr) + (arrsize / numthr);

      for (i = start; i < end; i++)
      {
        float a = b[i];
        a = conv1dline(a, xn, &(s_mlp_cfc_w[WVSIZE * i]), WVSIZE);

        a = a * 0.5 * (1.0 + tanh(0.7978845676080871 * a * (1.0 + 0.044715 * a * a)));
        mlp[i] = a;
      }

      work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);


      long WVSIZE_4 = WVSIZE * 4;
      w = (float *)s_mlp_cproj_w;
      b = s_mlp_cproj_b;

      arrsize = WVSIZE;
      start = thr * (arrsize / numthr);
      end = thr * (arrsize / numthr) + (arrsize / numthr);
      for (i = start; i < end; i++)
      {
        float a = b[i];
        x[i] += conv1dline(a, tmp, &(s_mlp_cproj_w[WVSIZE_4 * i]), WVSIZE_4);
      }
      
      work_group_barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

    }
  }
}
