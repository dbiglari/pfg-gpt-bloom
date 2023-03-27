// void *lt_thr(void *thread_args)
// {
//   lm_logit_t *logits = (lm_logit_t *)thread_args;

//   long long arrsize = logits->output_size;
//   bloom_precision arrsize_float = arrsize / logits->numthr;
//   long long start = logits->thr * (arrsize_float);
//   long long end = logits->thr * (arrsize_float) + (arrsize_float);
    
//   long long i, j;
//   long i_times_input_size = start * logits->input_size;
//   for (i = start; i < end; i++)
//   {
    
//     // testing speed
//     // if (i%30!=0)
//     //   continue;
//     logits->output[i] = 0;
//     if (logits->bias != NULL)
//       logits->output[i] = logits->bias[i];
//     for (j = 0; j < logits->input_size; j++)
//     {
//       logits->output[i] += logits->input[j] * logits->weights[i_times_input_size + j];
//     }
//     i_times_input_size += logits->input_size;
//   }
// }


__kernel void linear_transform_cl(
  __global float *input,
   __global float *output,
   __global float *weights,
   __global float *bias,
   unsigned int input_size,
   unsigned int output_size,
   __global float *y)
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

  if (y[0] == 0  || y[0]<0)
  {
    int numthr = num_groups;
    int thr = grp_id;
    int numsubthr = l_size;
    int subthr = l_id;

    // if (thr == 0 && subthr == 0)
    // {
    //   printf ("%d %d\n", numthr, numsubthr);
    // }


    {

      
      float arrsize = output_size;
      float arrsize_over_numthr = arrsize /  numthr;
      long start = thr * (arrsize_over_numthr);
      long end = thr * (arrsize_over_numthr) + (arrsize_over_numthr);

      int j = 0;
      long i_times_input_size = start * input_size;
      for (i = start; i < end; i++)
      {
        __local float temparray[256];

        float subarrsize = input_size;
        float subarrsize_over_numsubthr = subarrsize /  numsubthr;
        long substart = subthr * (subarrsize_over_numsubthr);
        long subend = subthr * (subarrsize_over_numsubthr) + (subarrsize_over_numsubthr);

        temparray[subthr]=0;
        for (j = substart; j < subend; j++)
        {
          temparray[subthr] += input[j] * weights[i_times_input_size + j];
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if (subthr == 0)
        {
          output[i] = bias[i];
          for (j = 0; j < numsubthr; j++)
          {
            output[i]+=temparray[j];
          }
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        i_times_input_size += input_size;       
        
      }
    }
 
  }
  
  // if (y[0] == 1  || y[0]<0)
  // {
  //   int numthr = num_groups;
  //   int thr = grp_id;
  //   int numsubthr = l_size;
  //   int subthr = l_id;

  //   if (thr == 0 && subthr == 0)
  //   {
  //     printf ("%d %d\n", input_size, output_size);
      
  //     int j = 0;
  //     long i_times_input_size = 0 * input_size;
  //     for (i = 0; i < output_size; i++)
  //     {
  //       output[i] = bias[i]; 
  //       for (j = 0; j < input_size; j++)
  //       {
  //         output[i] = input[j];
  //       //   output[i] += input[j] * weights[i_times_input_size + j];
  //       }

  //       i_times_input_size += input_size;       
        
  //     }
  //   }
 
  // }  


}
