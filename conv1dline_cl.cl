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


__kernel void conv1dline_cl(
    float b,
    __global float *v,
    unsigned int m_offset,
    unsigned int size,
    unsigned int arraychoice,
    __global float *s_attn_cattn_b,
    __global float *s_attn_cattn_w,    
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


 
}
