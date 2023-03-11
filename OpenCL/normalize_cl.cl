__kernel void normalize_cl(
__global float *x,  
__global float *o,
__global float *b,
__global float *g,
__global float *eps,
__global float *mean_val,
__global float *rstd_val,
const unsigned long size) {
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);
  

  if (g_id < size)
  {

    // do average and standard deviation on one cores, slower
    // if (g_id == 0)
    // {    
    //   for (int i=0;i<size;i++)
    //   {
    //     (*mean_val) += x[i];
    //     (*rstd_val) += x[i] * x[i];
    //   }
    //   (*mean_val) /= (float)size;
    //   (*rstd_val) = 1.0 / sqrt(*rstd_val /((float)size) - (*mean_val) * (*mean_val) + (*eps));
    // }

    // do average and standard deviation across multiple cores
    if (g_id < num_groups)
    {
      mean_val[g_id] = 0;
      rstd_val[g_id] = 0;
      for (int i=g_id*l_size;i<g_id*l_size+l_size;i++)
      {
        (mean_val[g_id]) += x[i];
        (rstd_val[g_id]) += x[i] * x[i];
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE );
    if (g_id == 0)
    {
      for (int i=1;i<num_groups;i++)
      {
        (mean_val[0]) += mean_val[i];
        (rstd_val[0]) += rstd_val[i];
      }         
      float c = 1.0/((float)size);
      (mean_val[0]) *= c;
      (rstd_val[0]) = 1.0 / sqrt(rstd_val[0] * c - (mean_val[0]) * (mean_val[0]) + (*eps));
    }      
    barrier(CLK_GLOBAL_MEM_FENCE );    
    
    
  
    float scale = (rstd_val[0]);
    float bias = -(rstd_val[0]) * (mean_val[0]);

    float gamma_v = g[g_id];
    float beta_v = b[g_id];
    o[g_id] = (x[g_id] * scale + bias) * gamma_v + beta_v;
  }


}