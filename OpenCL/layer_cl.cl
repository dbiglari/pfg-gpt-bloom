 

 void normalize()
 {
  float mean_val;
  float rstd_val;
  for (int i=0;i<size;i++)
  {
    mean_val += x[i];
    rstd_val += x[i] * x[i];
  }
  mean_val /= (float)size;
  rstd_val = 1.0 / sqrt(*rstd_val /((float)size) - (*mean_val) * (*mean_val) + (*eps));
 }

__kernel void layer_cl(
    __global float *x,
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
    __global float *k,
    __global float *v,
    unsigned int CTXSIZE,
    unsigned int HEADSIZE,
    unsigned int NUMHEADS,
    unsigned int NUMLAYERS   
    ) {
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);

  // do single threaded first
  int numthr = 1;
  int thr = 1;

  if (g_id == 0)
  {    
    //normalize(x, x, s_ln1_b, s_ln1_g, modelnum, querynum);

    //barrier(CLK_GLOBAL_MEM_FENCE );

  //   /* produce query/key/value vectors for this slot */
  //   {
  //     float *b = s_attn_cattn_b;
  //     float *w = (float *)s_attn_cattn_w;

  //     long vi = 0;
  //     long ki = 0;
  //     long qi = 0;

  //     // for(i=thr;i<models[modelnum].WVSIZE*3;i+=numthr)
  //     // {
  //     arrsize = WVSIZE * 3;
  //     start = thr * (arrsize / numthr);
  //     end = thr * (arrsize / numthr) + (arrsize / numthr);

  //     int j = 0;
  //     long row = (start / HEADSIZE);
  //     long row_mod_3 = ((start / HEADSIZE) % 3);
  //     long row_over_3 = row / 3;
  //     long row_over_3_times_HEADSIZE = row_over_3 * HEADSIZE;
  //     for (i = start; i < end; i++)
  //     {
  //       float a = conv1dline(b ? b[i] : 0, xn, w + WVSIZE * i, WVSIZE);

  //       // long row = (i / HEADSIZE);
  //       if (j >= HEADSIZE)
  //       {
  //         row++;
  //         row_mod_3++;
  //         if (row_mod_3 >= 3)
  //         {
  //           row_mod_3 = 0;
  //           row_over_3++;
  //           row_over_3_times_HEADSIZE = row_over_3 * HEADSIZE;
  //         }

  //         j = 0;
  //       }

  //       if ((row_mod_3) == 0)
  //       {
  //         // index based off of i to support multithreading
  //         // long tqi = (j);
  //         // q[0 * WVSIZE + (row / 3) * HEADSIZE + tqi] = a;
  //         q[row_over_3_times_HEADSIZE + j] = a;
  //         qi++;
  //       }
  //       else if ((row_mod_3) == 1)
  //       {
  //         // index based off of i to support multithreading
  //         // long tki = (j);
  //         s_k[here * WVSIZE + row_over_3_times_HEADSIZE + j] = a;
  //         ki++;
  //       }
  //       else if ((row_mod_3) == 2)
  //       {
  //         // index based off of i to support multithreading
  //         // long tvi = (j);
  //         s_v[here * WVSIZE + row_over_3_times_HEADSIZE + j] = a;
  //         vi++;
  //       }
  //       j++;
  //     }
  //   }

  //   barrier(CLK_GLOBAL_MEM_FENCE );




  // long layeridx_NUMHEADS = layeridx * NUMHEADS;
  // // for(h=thr;h<models[modelnum].NUMHEADS;h+=numthr)
  // // {
  // arrsize = NUMHEADS;
  // start = thr * (arrsize / numthr);
  // end = thr * (arrsize / numthr) + (arrsize / numthr);
  // for (h = start; h < end; h++)
  // {

  //   long h_CTXSIZE = h * CTXSIZE;
  //   long h_HEADSIZE = h * HEADSIZE;
  //   /* query * keys = attentions */
  //   for (i = 0; i <= here; i++)
  //   {
  //     float a = conv1dline(0, &(q[h_HEADSIZE]), &(l->k[((i * WVSIZE) + (h_HEADSIZE))]), HEADSIZE);
  //     att[h_CTXSIZE + i] = a * RSQRT_HEADSIZE + models[modelnum].alibi[((long)closest_power_of_2) * i + h];
  //     // queries[querynum].attentions_presoftmax[0*models[modelnum].WVSIZE*models[modelnum].NUMLAYERS+layeridx*models[modelnum].NUMHEADS+h]=queries[querynum].att[i];
  //     attentions_presoftmax[layeridx_NUMHEADS + h] = att[i];
  //   }

  //   /* softmax attentions to make them sum up to 1.0 */
  //   float max = att[h_CTXSIZE];
  //   for (i = 1; i <= here; i++)
  //     if (att[h_CTXSIZE + i] > max)
  //       max = att[h_CTXSIZE + i];
  //   float sum = 0;
  //   for (i = 0; i <= here; i++)
  //   {
  //     float a = exp(att[h_CTXSIZE + i] - max);
  //     att[h_CTXSIZE + i] = a;
  //     sum += a;
  //   }
  //   float sumr = 1.0 / sum;
  //   for (i = 0; i <= here; i++)
  //     att[h_CTXSIZE + i] *= sumr;

  //   /* store attention data for visualization */
  //   if (attentions)
  //     for (i = 0; i <= here; i++)
  //     {
  //       attentions[layeridx_NUMHEADS + h] = att[h_CTXSIZE + i];
  //     }
  // }

  // /* apply attentions to values */
  // {
  //   float *l_v = l->v;
  //   // for(h=thr;h<NUMHEADS;h+=numthr)
  //   // {
  //   arrsize = models[modelnum].NUMHEADS;
  //   start = thr * (arrsize / numthr);
  //   end = thr * (arrsize / numthr) + (arrsize / numthr);
  //   for (h = start; h < end; h++)
  //   {
  //     long h_HEADSIZE = h * HEADSIZE;
  //     long h_CTXSIZE = h * CTXSIZE;
  //     for (j = 0; j < HEADSIZE; j++)
  //     {
  //       tmp[h_HEADSIZE + j] = 0;
  //       for (i = 0; i < here + 1; i++)
  //       {
  //         // tmp[h*HEADSIZE+j]+=conv1dline(0,queries[querynum].att+h*CTXSIZE+i,l->v+(i*WVSIZE+h*HEADSIZE+j),1);
  //         tmp[h_HEADSIZE + j] += (*(att + h_CTXSIZE + i)) * (*(l_v + (i * WVSIZE + h_HEADSIZE + j)));
  //       }
  //     }
  //   }
  // }


  }
}