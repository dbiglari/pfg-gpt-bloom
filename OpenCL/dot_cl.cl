__kernel void dot_cl(
__global float *v,
__global float *a,  
__global float *m,
const unsigned long wdt) {
  int g_id = get_global_id(0);
  int num_groups = get_num_groups(0);
  int l_id = get_local_id(0);
  int g_size = get_global_size(0);
  int l_size = get_local_size(0);

  if (g_id < wdt)
  {
    if (g_id < num_groups)
    {
      a[g_id] = 0;
      
      for (int i=g_id*l_size;i<g_id*l_size+l_size;i++)
      {
        if (i < wdt)
        {
          a[g_id] += v[i] * m[i];
        }
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE );
    if (g_id == 0)
    {
      for (int i=1;i<num_groups;i++)
      {
        a[0] += a[i];
      }         
    }      
  }
}