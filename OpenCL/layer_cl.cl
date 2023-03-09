 

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

}