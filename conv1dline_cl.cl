
__kernel void conv1dline_cl(
    float b,
    __global float *v,
    unsigned int m_offset,
    unsigned int size,
    unsigned int arraychoice,
    __global float *s_attn_cattn_b,
    __global float *s_attn_cattn_w,    
    __global float *output,
    __local float *scratch
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



  //  int gid = get_global_id(0);
  //  int lid = get_local_id(0);
  //  int group_size = get_local_size(0);

  //  scratch[lid] = v[gid] * s_attn_cattn_w[gid];
  //  barrier(CLK_LOCAL_MEM_FENCE);

  //  for(int i = group_size/2; i>0; i >>= 1) {
  //     if(lid < i) {
  //        scratch[lid] += scratch[lid + i];
  //     }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //  }

  //  if(lid == 0) {
  //     output[get_group_id(0)] = dot(scratch[0], (float4)(1.0f));
  //  }
 
}
