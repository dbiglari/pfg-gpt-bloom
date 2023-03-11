//
// File:       normalize_C.c
//
////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <time.h>
//#include <CL/c.h>
#include "layer_cl.h"


// // 560m parameters
// #define WV_SIZE 1024
// #define CTX_SIZE 2048 
// #define NUM_HEADS 16
// #define NUM_LAYERS 24

// // 1b7 parameters
// #define WV_SIZE 2048
// #define CTX_SIZE 2048 
// #define NUM_HEADS 16
// #define NUM_LAYERS 24


// // 3b parameters
// #define WV_SIZE 2560
// #define CTX_SIZE 2048 
// #define NUM_HEADS 32
// #define NUM_LAYERS 30

// // 7b1 parameters
// #define WV_SIZE 4096
// #define CTX_SIZE 2048 
// #define NUM_HEADS 32
// #define NUM_LAYERS 30

// 175b parameters
#define WV_SIZE 14336
#define CTX_SIZE 2048 
#define NUM_HEADS 112
#define NUM_LAYERS 70


////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//


////////////////////////////////////////////////////////////////////////////////

char *KernelSource;


/* file management functions */
void *readfile(char *fn, int *lgt_ret, char *path)
{
  char filename[2048];
  if (path != NULL)
    sprintf(filename, "%s/%s", path, fn);
  else
    strcpy(filename, fn);
  FILE *file = fopen(filename, "r");
  if (file == NULL)
  {
    fprintf(stderr, "Expected file \"%s\" not found", path);
    return NULL;
  }
  if (lgt_ret)
    *lgt_ret = 0;

  fseek(file, 0, SEEK_END);
  long len = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *buffer = malloc(len + 1);

  if (buffer == NULL)
  {
    fprintf(stderr, "Unable to allocate memory for file");
    fclose(file);
    return NULL;
  }

  size_t readbytes = fread(buffer, 1, len, file);
  if (readbytes != len)
  {
    // something weird happened
    fprintf(stderr, "readbytes != len in %s\n", filename);
  }
  buffer[len] = '\0';
  fclose(file);

  if (lgt_ret)
    *lgt_ret = len;
  return (void *)buffer;
}

// malloc wrapper, used to count bytes allocated during testing
void *malloc_wrapper(opencl_kernel_layer_cl_t *state, long size)
{
    state->total_malloc += size;
    void * ret_ptr = malloc(size);
    return ret_ptr;
}

////////////////////////////////////////////////////////////////////////////////

int layer_cl_test()
{
    opencl_kernel_layer_cl_t *state = NULL;
    state = layer_cl_wrapper(state);

    state->initialize = 1;
    state->populate_data_for_test = 0;
    state->WVSIZE = WV_SIZE;
    state->CTXSIZE = CTX_SIZE;
    state->NUMHEADS = NUM_HEADS;
    state->NUMLAYERS = NUM_LAYERS;    
    state->HEADSIZE = state->WVSIZE / state->NUMHEADS;
    state->closest_power_of_2 = pow(2, floor(log2(state->NUMHEADS)));
    state->x  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->xn  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->y  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_ln1_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_ln1_g  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_ln2_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_ln2_g  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_mlp_cfc_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * 4);
    state->s_mlp_cfc_w  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * state->WVSIZE *4);
    state->s_mlp_cproj_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_mlp_cproj_w  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * state->WVSIZE * 4);
    state->s_attn_cattn_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * 3);
    state->s_attn_cattn_w  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * state->WVSIZE *3);
    state->s_attn_cproj_b  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE);
    state->s_attn_cproj_w  = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * state->WVSIZE);
    state->att  = (float *) malloc_wrapper(state, sizeof(float) * state->closest_power_of_2 * state->CTXSIZE * state->NUMHEADS + state->CTXSIZE);
    state->attentions  = (float *) malloc_wrapper(state, sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS);
    state->attentions_presoftmax  = (float *) malloc_wrapper(state, sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS);
    state->alibi = (float *) malloc_wrapper(state, sizeof(float) * state->closest_power_of_2 * state->CTXSIZE);
    state->tmp = (float *) malloc_wrapper(state, sizeof(float) * state->WVSIZE * state->CTXSIZE);
    state->q  = (float *) malloc_wrapper(state, sizeof(float) * state->CTXSIZE * state->WVSIZE );    
    state->k  = (float *) malloc_wrapper(state, sizeof(float) * state->CTXSIZE * state->WVSIZE );
    state->v  = (float *) malloc_wrapper(state, sizeof(float) * state->CTXSIZE * state->WVSIZE );
    state->here = 0;
    
    for(int i = 0; i < state->WVSIZE; i++)
    {
        state->x[i] = rand() / (float)RAND_MAX;
        state->y[i] = rand() / (float)RAND_MAX;
    }  
    layer_cl_wrapper(state);

    state->setparams = 1;
    state->set_x = 1;
    state->set_xn = 1;
    state->set_s_ln1_b = 1;
    state->set_s_ln1_g = 1;
    state->set_s_ln2_b = 1;
    state->set_s_ln2_g = 1;
    state->set_s_mlp_cfc_b = 1;
    state->set_s_mlp_cfc_w = 1;
    state->set_s_mlp_cproj_b = 1;
    state->set_s_mlp_cproj_w = 1;
    state->set_s_attn_cattn_b = 1;
    state->set_s_attn_cattn_w = 1;
    state->set_s_attn_cproj_b = 1;
    state->set_s_attn_cproj_w = 1;
    state->set_att = 1;
    state->set_attentions = 1;
    state->set_attentions_presoftmax = 1;
    state->set_alibi = 1;    
    state->set_tmp = 1;    
    state->set_q = 1;
    state->set_k = 1;
    state->set_v = 1;
    state->set_WVSIZE = 1;
    state->set_CTXSIZE = 1;
    state->set_HEADSIZE = 1;
    state->set_NUMHEADS = 1;
    state->set_NUMLAYERS = 1;
    state->set_layeridx = 1;
    state->set_closest_power_of_2 = 1;
    state->set_y = 1;     
    state->set_here = 1;
    state->get_max_workgroup = 1;
    layer_cl_wrapper(state);


    double sum = 0;
    double add = 1;

    // Start measuring time
    struct timespec begin, end; 
    clock_gettime(CLOCK_REALTIME, &begin);

    for (int i=0;i<state->NUMLAYERS;i++)
    {

        state->setparams = 1;
        state->set_x = 1;

        // state->set_att = 1;
        // state->set_attentions = 1;
        // state->set_attentions_presoftmax = 1;
        // state->set_alibi = 1;
        // state->set_here = 1;

        layer_cl_wrapper(state);
        
        state->execute = 1; 
        layer_cl_wrapper(state);
    }
    
    // Stop measuring time and calculate the elapsed time
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;
    
    printf("Time measured: %.9f seconds.\n", elapsed);
    
    return 0;

    state->cleanup = 1;
    layer_cl_wrapper(state);



    // opencl_kernel_layer_cl_t state={0};
    // state.useDeviceNum=0;
    // state.populate_data_for_test = 1;
    // initialize_layer_cl(&state);
    // set_parameters_layer_cl(&state);
    // printf ("initialization complete\n");
    // fflush(stdout);
    // execute_layer_cl(&state);
    // printf ("execution complete\n");
    // fflush(stdout);    
    // release_layer_cl(&state);    
}

#ifdef BUILD_TEST
int main(int argc, char** argv)
{
    layer_cl_test();
}
#endif

opencl_kernel_layer_cl_t *layer_cl_wrapper(opencl_kernel_layer_cl_t *state)
{


    if (state == NULL)
    {
        state = (opencl_kernel_layer_cl_t *) malloc(sizeof(opencl_kernel_layer_cl_t));
        memset(state, 0, sizeof(opencl_kernel_layer_cl_t));
        return state;
    }
    
    if(state->initialize == 1)
    {
        initialize_layer_cl(state);
        state->initialize = 0;
        return state;
    }

    if (state->setparams == 1)
    {
        set_parameters_layer_cl(state);
        state->setparams = 0;
        return state;
    }    
   
    if (state->execute == 1)
    {
        execute_layer_cl(state);
        state->execute = 0;
        return state;
    }

    if (state->cleanup == 1)
    {
        release_layer_cl(state); 
        state->cleanup = 0;
        return state;
    }    

    return state;

}


void initialize_layer_cl(opencl_kernel_layer_cl_t *state)
{
    state->numPlatforms; //the NO. of platforms
    state->platform = NULL; //the chosen platform    

    state->numDevices = 0;
    state->gpu = 1;
   
    // Fill our data set with random float values
    //
    int i = 0;

    if (state->populate_data_for_test == 1)
    {
        state->WVSIZE = WV_SIZE;
        state->CTXSIZE = CTX_SIZE;
        state->NUMHEADS = NUM_HEADS;
        state->NUMLAYERS = NUM_LAYERS;    
        state->HEADSIZE = state->WVSIZE / state->NUMHEADS;
        state->closest_power_of_2 = pow(2, floor(log2(state->NUMHEADS)));

        state->x  = (float *) malloc(sizeof(short) * state->WVSIZE);
        state->xn  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->y  = (float *) malloc(sizeof(short) * state->WVSIZE);

        state->s_ln1_b  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_ln1_g  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_ln2_b  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_ln2_g  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_mlp_cfc_b  = (float *) malloc(sizeof(float) * state->WVSIZE * 4);
        state->s_mlp_cfc_w  = (float *) malloc(sizeof(float) * state->WVSIZE * state->WVSIZE *4);
        state->s_mlp_cproj_b  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_mlp_cproj_w  = (float *) malloc(sizeof(float) * state->WVSIZE * state->WVSIZE * 4);
        state->s_attn_cattn_b  = (float *) malloc(sizeof(float) * state->WVSIZE * 3);
        state->s_attn_cattn_w  = (float *) malloc(sizeof(float) * state->WVSIZE * state->WVSIZE *3);
        state->s_attn_cproj_b  = (float *) malloc(sizeof(float) * state->WVSIZE);
        state->s_attn_cproj_w  = (float *) malloc(sizeof(float) * state->WVSIZE * state->WVSIZE);
        state->att  = (float *) malloc(sizeof(float) * state->closest_power_of_2 * state->CTXSIZE * state->NUMHEADS + state->CTXSIZE);
        state->attentions  = (float *) malloc(sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS);
        state->attentions_presoftmax  = (float *) malloc(sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS);
        state->alibi = (float *) malloc(sizeof(float) * state->closest_power_of_2 * state->CTXSIZE);
        state->tmp = (float *) malloc(sizeof(float) * state->WVSIZE * state->CTXSIZE);
        state->q  = (float *) malloc(sizeof(float) * state->CTXSIZE * state->WVSIZE);
        state->k  = (float *) malloc(sizeof(float) * state->CTXSIZE * state->WVSIZE);
        state->v  = (float *) malloc(sizeof(float) * state->CTXSIZE * state->WVSIZE);
        state->here = 0;

        for(i = 0; i < state->WVSIZE; i++)
        {
            state->x[i] = rand() / (float)RAND_MAX;
            state->xn[i] = rand() / (float)RAND_MAX;
            state->y[i] = rand() / (float)RAND_MAX;
        }

        state->set_x = 1;
        state->set_xn = 1;
        state->set_s_ln1_b = 1;
        state->set_s_ln1_g = 1;
        state->set_s_ln2_b = 1;
        state->set_s_ln2_g = 1;
        state->set_s_mlp_cfc_b = 1;
        state->set_s_mlp_cfc_w = 1;
        state->set_s_mlp_cproj_b = 1;
        state->set_s_mlp_cproj_w = 1;
        state->set_s_attn_cattn_b = 1;
        state->set_s_attn_cattn_w = 1;
        state->set_s_attn_cproj_b = 1;
        state->set_s_attn_cproj_w = 1;
        state->set_att = 1;
        state->set_attentions = 1;
        state->set_attentions_presoftmax = 1;
        state->set_alibi = 1;
        state->set_tmp = 1;
        state->set_q = 1;
        state->set_k = 1;
        state->set_v = 1;
        state->set_WVSIZE = 1;
        state->set_CTXSIZE = 1;
        state->set_HEADSIZE = 1;
        state->set_NUMHEADS = 1;
        state->set_NUMLAYERS = 1;
        state->set_layeridx = 1;
        state->set_closest_power_of_2 = 1;
        state->set_y = 1;
        state->set_here = 1;

    }

    KernelSource = readfile("layer_cl.cl", &state->length, ".");

	state->status = clGetPlatformIDs(0, NULL, &state->numPlatforms);
	if (state->status != CL_SUCCESS)
	{
		
		return EXIT_FAILURE;
	}        

	/*For clarity, choose the first available platform. */
	if (state->numPlatforms > 0)
	{
		cl_platform_id* platforms = 
                     (cl_platform_id*)malloc(state->numPlatforms * sizeof(cl_platform_id));
		state->status = clGetPlatformIDs(state->numPlatforms, platforms, NULL);
		state->platform = platforms[0];
		free(platforms);
	}
    
    // Connect to a compute device
    //
    state->status = clGetDeviceIDs(state->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &state->numDevices);
	if (state->numDevices == 0) //no GPU available.
	{
		state->status = clGetDeviceIDs(state->platform, CL_DEVICE_TYPE_CPU, 0, NULL, &state->numDevices);
		state->devices = (cl_device_id*)malloc(state->numDevices * sizeof(cl_device_id));
		state->status = clGetDeviceIDs(state->platform, CL_DEVICE_TYPE_CPU, state->numDevices, state->devices, NULL);
	}
	else
	{
		state->devices = (cl_device_id*)malloc(state->numDevices * sizeof(cl_device_id));
		state->status = clGetDeviceIDs(state->platform, CL_DEVICE_TYPE_GPU, state->numDevices, state->devices, NULL);
	}

    // Create a compute context 
    //
    state->context = clCreateContext(0, 1, &state->devices[state->useDeviceNum], NULL, NULL, &state->err);
    if (!state->context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    state->commands = clCreateCommandQueue(state->context, state->devices[state->useDeviceNum], 0, &state->err);
    if (!state->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    state->program = clCreateProgramWithSource(state->context, 1, (const char **) & KernelSource, NULL, &state->err);
    if (!state->program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    state->err = clBuildProgram(state->program, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(state->program, state->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    state->kernel = clCreateKernel(state->program, "layer_cl", &state->err);
    if (!state->kernel || state->err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    state->x_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(short) * state->WVSIZE , NULL, NULL);
    state->xn_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE , NULL, NULL);

    state->s_ln1_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE, NULL, NULL);
    state->s_ln1_g_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE, NULL, NULL);
    state->s_ln2_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE, NULL, NULL);
    state->s_ln2_g_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE , NULL, NULL);
    state->s_mlp_cfc_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE *4 , NULL, NULL);
    state->s_mlp_cfc_w_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * state->WVSIZE *4, NULL, NULL);
    state->s_mlp_cproj_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE , NULL, NULL);
    state->s_mlp_cproj_w_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * state->WVSIZE * 4 , NULL, NULL);
    state->s_attn_cattn_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * 3 , NULL, NULL);
    state->s_attn_cattn_w_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * state->WVSIZE *3, NULL, NULL);
    state->s_attn_cproj_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE , NULL, NULL);
    state->s_attn_cproj_w_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * state->WVSIZE, NULL, NULL);
    state->att_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->closest_power_of_2 * state->CTXSIZE * state->NUMHEADS + state->CTXSIZE , NULL, NULL);
    state->attentions_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS , NULL, NULL);
    state->attentions_presoftmax_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS , NULL, NULL);
    state->alibi_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->closest_power_of_2 , NULL, NULL);
    state->tmp_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->WVSIZE , NULL, NULL);
    state->q_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->WVSIZE, NULL, NULL);
    state->k_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->WVSIZE, NULL, NULL);
    state->v_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->CTXSIZE * state->WVSIZE, NULL, NULL);

    state->y_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(short) * state->WVSIZE , NULL, NULL);

    if (!state->x_data || !state->y_data )
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
}


void set_parameters_layer_cl(opencl_kernel_layer_cl_t *state)
{

 

    // Set the arguments to our compute kernel
    //
    state->err = 0;

    // Write our data set into the input array in device memory 
    //
    if (state->set_x == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->x_data, CL_TRUE, 0, sizeof(short) * state->WVSIZE, state->x, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_y == 1)
    {    
        state->err |= clEnqueueWriteBuffer(state->commands, state->y_data, CL_TRUE, 0, sizeof(short) * state->WVSIZE, state->y, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }    
    }
    if (state->set_xn == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->xn_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->xn, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_ln1_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_ln1_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_ln1_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }

    if (state->set_s_ln1_g == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_ln1_g_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_ln1_g, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_ln2_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_ln2_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_ln2_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_ln2_g == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_ln2_g_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_ln2_g, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_mlp_cfc_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_mlp_cfc_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE * 4, state->s_mlp_cfc_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_mlp_cfc_w == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_mlp_cfc_w_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE * state->WVSIZE * 4, state->s_mlp_cfc_w, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_mlp_cproj_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_mlp_cproj_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_mlp_cproj_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_mlp_cproj_w == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_mlp_cproj_w_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE * state->WVSIZE * 4, state->s_mlp_cproj_w, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_attn_cattn_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_attn_cattn_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE * 3, state->s_attn_cattn_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_attn_cattn_w == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_attn_cattn_w_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE * state->WVSIZE * 3, state->s_attn_cattn_w, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_attn_cproj_b == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_attn_cproj_b_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->s_attn_cproj_b, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_s_attn_cproj_w == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->s_attn_cproj_w_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE  * state->WVSIZE, state->s_attn_cproj_w, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_att == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->att_data, CL_TRUE, 0, sizeof(float) * state->closest_power_of_2 * state->CTXSIZE * state->NUMHEADS + state->CTXSIZE, state->att, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_attentions == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->attentions_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS, state->attentions, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }
    if (state->set_attentions_presoftmax == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->attentions_presoftmax_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->NUMLAYERS * state->NUMHEADS, state->attentions_presoftmax, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }
    if (state->set_alibi == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->alibi_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->closest_power_of_2, state->alibi, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }    
    if (state->set_tmp == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->tmp_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->WVSIZE, state->tmp, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }    
    if (state->set_q == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->q_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->WVSIZE, state->q, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }    
    if (state->set_k == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->k_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->WVSIZE, state->k, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }    
    if (state->set_v == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->v_data, CL_TRUE, 0, sizeof(float) * state->CTXSIZE * state->WVSIZE, state->v, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }    


    if (state->set_x == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->x_data);
        state->set_x = 0;
    }
    if (state->set_xn == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->xn_data);
        state->set_xn = 0;
    }    
    if (state->set_y == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &state->y_data);
        state->set_y = 0;
    }
    if (state->set_WVSIZE == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 3, sizeof(unsigned int), &state->WVSIZE);
        state->set_WVSIZE = 0;
    }
    if (state->set_s_ln1_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 4, sizeof(cl_mem), &state->s_ln1_b_data);
        state->set_s_ln1_b = 0;
    }
    if (state->set_s_ln1_g == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 5, sizeof(cl_mem), &state->s_ln1_g_data);    
        state->set_s_ln1_g = 0;
    }
    if (state->set_s_ln2_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 6, sizeof(cl_mem), &state->s_ln2_b_data);
        state->set_s_ln2_b = 0;
    }
    if (state->set_s_ln2_g == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 7, sizeof(cl_mem), &state->s_ln2_g_data);
        state->set_s_ln2_g = 0;
    }
    if (state->set_s_mlp_cfc_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 8, sizeof(cl_mem), &state->s_mlp_cfc_b_data);
        state->set_s_mlp_cfc_b = 0;
    }
    if (state->set_s_mlp_cfc_w == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 9, sizeof(cl_mem), &state->s_mlp_cfc_w_data);
        state->set_s_mlp_cfc_w = 0;
    }
    if (state->set_s_mlp_cproj_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 10, sizeof(cl_mem), &state->s_mlp_cproj_b_data);
        state->set_s_mlp_cproj_b = 0;
    }
    if (state->set_s_mlp_cproj_w == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 11, sizeof(cl_mem), &state->s_mlp_cproj_w_data);
        state->set_s_mlp_cproj_w = 0;
    }
    if (state->set_s_attn_cattn_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 12, sizeof(cl_mem), &state->s_attn_cattn_b_data);
        state->set_s_attn_cattn_b = 0;
    }
    if (state->set_s_attn_cattn_w == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 13, sizeof(cl_mem), &state->s_attn_cattn_w_data);
        state->set_s_attn_cattn_w = 0;
    }
    if (state->set_s_attn_cproj_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 14, sizeof(cl_mem), &state->s_attn_cproj_b_data);
        state->set_s_attn_cproj_b = 0;
    }
    if (state->set_s_attn_cproj_w == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 15, sizeof(cl_mem), &state->s_attn_cproj_w_data);
        state->set_s_attn_cproj_w = 0;
    }
    if (state->set_att == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 16, sizeof(cl_mem), &state->att_data);
        state->set_att = 0;
    }
    if (state->set_attentions == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 17, sizeof(cl_mem), &state->attentions_data);
        state->set_attentions = 0;
    }
    if (state->set_attentions_presoftmax == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 18, sizeof(cl_mem), &state->attentions_presoftmax_data);
        state->set_attentions_presoftmax = 0;
    }
    if (state->set_alibi == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 19, sizeof(cl_mem), &state->alibi_data);
        state->set_alibi = 0;
    }    
    if (state->set_tmp == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 20, sizeof(cl_mem), &state->tmp_data);
        state->set_tmp = 0;
    }    
    if (state->set_q == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 21, sizeof(cl_mem), &state->q_data);
        state->set_q = 0;
    }    
    if (state->set_k == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 22, sizeof(cl_mem), &state->k_data);
        state->set_k = 0;
    }
    if (state->set_v == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 23, sizeof(cl_mem), &state->v_data);
        state->set_v = 0;
    }        
    if (state->set_closest_power_of_2 == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 24, sizeof(float), &state->closest_power_of_2);
        state->set_closest_power_of_2 = 0;
    }       
    if (state->set_CTXSIZE == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 25, sizeof(unsigned int), &state->CTXSIZE);
        state->set_CTXSIZE = 0;
    }
    if (state->set_HEADSIZE == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 26, sizeof(unsigned int), &state->HEADSIZE);
        state->set_HEADSIZE = 0;
    }
    if (state->set_NUMHEADS == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 27, sizeof(unsigned int), &state->NUMHEADS);
        state->set_NUMHEADS = 0;
    }
    if (state->set_NUMLAYERS == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 28, sizeof(unsigned int), &state->NUMLAYERS);
        state->set_NUMLAYERS = 0;
    }
    if (state->set_layeridx == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 29, sizeof(unsigned int), &state->layeridx);
        state->set_layeridx = 0;
    }    
    if (state->set_here == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 30, sizeof(unsigned int), &state->here);
        state->set_here = 0;
    }    


    if (state->get_max_workgroup == 1)
    {
        // Get the maximum work group size for executing the kernel on the device
        //
        state->err = clGetKernelWorkGroupInfo(state->kernel, state->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(state->local), &state->local, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to retrieve kernel work group info! %d\n", state->err);
            exit(1);
        }
        state->get_max_workgroup = 0;
    }

}

void execute_layer_cl(opencl_kernel_layer_cl_t *state)
{
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    state->global = state->WVSIZE;
    //state->local = state->WVSIZE;
    state->err = clEnqueueNDRangeKernel(state->commands, state->kernel, 1, NULL, &state->global, &state->local, 0, NULL, NULL);
    if (state->err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(state->commands);

    // Read back the results from the device to verify the output
    //
    state->err = clEnqueueReadBuffer( state->commands, state->y_data, CL_TRUE, 0, sizeof(short) * state->WVSIZE, state->y, 0, NULL, NULL );  
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", state->err);
        exit(1);
    }

    
}


    
void release_layer_cl(opencl_kernel_layer_cl_t *state)
{
    // Shutdown and cleanup
    //
    clReleaseMemObject(state->x_data);
    clReleaseMemObject(state->xn_data);
    clReleaseMemObject(state->y_data);

    clReleaseMemObject(state->s_ln1_b_data);
    clReleaseMemObject(state->s_ln1_g_data);    
    clReleaseMemObject(state->s_ln2_b_data);
    clReleaseMemObject(state->s_ln2_g_data);
    clReleaseMemObject(state->s_mlp_cfc_b_data);
    clReleaseMemObject(state->s_mlp_cfc_w_data);
    clReleaseMemObject(state->s_mlp_cproj_b_data);
    clReleaseMemObject(state->s_mlp_cproj_w_data);
    clReleaseMemObject(state->s_attn_cattn_b_data);
    clReleaseMemObject(state->s_attn_cattn_w_data);
    clReleaseMemObject(state->s_attn_cproj_b_data);
    clReleaseMemObject(state->s_attn_cproj_w_data);
    clReleaseMemObject(state->att_data);
    clReleaseMemObject(state->attentions_data);
    clReleaseMemObject(state->attentions_presoftmax_data);
    clReleaseMemObject(state->alibi_data);
    clReleaseMemObject(state->tmp_data);
    clReleaseMemObject(state->k_data);
    clReleaseMemObject(state->v_data);


        
    clReleaseProgram(state->program);
    clReleaseKernel(state->kernel);
    clReleaseCommandQueue(state->commands);
    clReleaseContext(state->context);

    free(state->x);
    free(state->y);

    return 0;
}


