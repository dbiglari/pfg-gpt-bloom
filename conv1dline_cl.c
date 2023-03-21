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
#include <time.h>
#include <CL/cl.h>
#include "common.h"


// 560m parameters
#define WV_SIZE 1024
#define CTX_SIZE 2048 
#define NUM_HEADS 16
#define NUM_LAYERS 24

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

// // 175b parameters
// #define WV_SIZE 14336
// #define CTX_SIZE 2048 
// #define NUM_HEADS 112
// #define NUM_LAYERS 70

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//


////////////////////////////////////////////////////////////////////////////////

char *KernelSource_conv1dline_cl;

void *readfile(char *fn, int *lgt_ret, char *path);

#ifdef BUILD_TEST

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
#endif


float conv1dline_cl(float a, float *v, int m_offset, int size, int arraychoice, int modelnum, int layeridx, int thr)
{

    // arraychoice 0 = attn_cattn

    if (models[modelnum].layers[layeridx].conv1dline_cl[thr] == NULL)
    {
        // allocate the structures
        models[modelnum].layers[layeridx].conv1dline_cl[thr] = conv1dline_cl_wrapper(NULL);
        
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->initialize = 1;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->populate_data_for_test = 0;   
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->WVSIZE = models[modelnum].WVSIZE;    
        
        
        // build the opencl context
        conv1dline_cl_wrapper(models[modelnum].layers[layeridx].conv1dline_cl[thr]);

        models[modelnum].layers[layeridx].conv1dline_cl[thr]->setparams = 1;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->v  = (float *) malloc_wrapper(&(models[modelnum].layers[layeridx].conv1dline_cl[thr]->total_malloc), sizeof(float) * models[modelnum].layers[layeridx].conv1dline_cl[thr]->WVSIZE);

        // pass in all the arrays

        models[modelnum].layers[layeridx].conv1dline_cl[thr]->s_attn_cattn_b  = models[modelnum].layers[layeridx].attn_cattn_b;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->s_attn_cattn_w  = models[modelnum].layers[layeridx].attn_cattn_w;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->scratch = (float *) malloc_wrapper(&(models[modelnum].layers[layeridx].conv1dline_cl[thr]->total_malloc), sizeof(float) * (models[modelnum].layers[layeridx].conv1dline_cl[thr]->WVSIZE +2));
        
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_v = 1;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_s_attn_cattn_b = 1;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_s_attn_cattn_w = 1;
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_scratch = 1;  
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->get_max_workgroup = 1;
        conv1dline_cl_wrapper(models[modelnum].layers[layeridx].conv1dline_cl[thr]);         
        models[modelnum].layers[layeridx].conv1dline_cl[thr]->numCores = models[modelnum].layers[layeridx].conv1dline_cl[thr]->local;
                
    }

    opencl_kernel_model_conv1dline_cl_t *state = models[modelnum].layers[layeridx].conv1dline_cl[thr];
    // set parameters
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->setparams = 1;  
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_v = 1;
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_b = 1;
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_size = 1;
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_m_offset = 1;
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->set_arraychoice = 1;
    conv1dline_cl_wrapper(models[modelnum].layers[layeridx].conv1dline_cl[thr]); 


    models[modelnum].layers[layeridx].conv1dline_cl[thr]->execute = 1;
    models[modelnum].layers[layeridx].conv1dline_cl[thr]->get_a = 1; 
    //stopwatch_start();
    conv1dline_cl_wrapper(models[modelnum].layers[layeridx].conv1dline_cl[thr]);   
    //stopwatch_end();

    return models[modelnum].layers[layeridx].conv1dline_cl[thr]->a;
}



////////////////////////////////////////////////////////////////////////////////

#ifdef BUILD_TEST
int main(int argc, char** argv)
{
    conv1dline_cl_test();
}
#endif

opencl_kernel_model_conv1dline_cl_t *conv1dline_cl_wrapper(opencl_kernel_model_conv1dline_cl_t *state)
{


    if (state == NULL)
    {
        state = (opencl_kernel_model_conv1dline_cl_t *) malloc(sizeof(opencl_kernel_model_conv1dline_cl_t));
        memset(state, 0, sizeof(opencl_kernel_model_conv1dline_cl_t));
        return state;
    }
    
    if(state->initialize == 1)
    {
        initialize_conv1dline_cl(state);
        state->initialize = 0;
        return state;
    }

    if (state->setparams == 1)
    {
        set_parameters_conv1dline_cl(state);
        state->setparams = 0;
        return state;
    }    
   
    if (state->execute == 1)
    {
        execute_conv1dline_cl(state);
        state->execute = 0;
        return state;
    }

    if (state->cleanup == 1)
    {
        release_conv1dline_cl(state); 
        state->cleanup = 0;
        return state;
    }    

    return state;

}


int initialize_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state)
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
        state->size = WV_SIZE;
        state->v  = (float *) malloc(sizeof(float) * state->size);
        state->s_attn_cattn_b  = (float *) malloc(sizeof(float) * state->size * 3);
        state->s_attn_cattn_w  = (float *) malloc(sizeof(float) * state->size * state->size *3);
        state->scratch = (float *) malloc(sizeof(float) * (state->size +2));
        state->v  = (float *) malloc(sizeof(float) * state->size);

        for(i = 0; i < state->size; i++)
        {
            state->v[i] = 0;
        }

        state->set_v = 1;
        state->set_s_attn_cattn_b = 1;
        state->set_s_attn_cattn_w = 1;
        state->set_scratch = 1;
        state->set_v = 1;
        state->set_size = 1;
    }

    KernelSource_conv1dline_cl = readfile("conv1dline_cl.cl", &state->length, ".");

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
    if (context == NULL)
    {
        context = clCreateContext(0, 1, &state->devices[state->useDeviceNum], NULL, NULL, &state->err);
        state->context = context;
    }
    else
    {
        state->context = context;
    }
    if (!state->context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    state->commands = clCreateCommandQueueWithProperties(state->context, state->devices[state->useDeviceNum], 0, &state->err);
    if (!state->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    state->program = clCreateProgramWithSource(state->context, 1, (const char **) & KernelSource_conv1dline_cl, NULL, &state->err);
    if (!state->program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    state->err = clBuildProgram(state->program, 0, NULL, "", NULL, NULL);
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
    state->kernel = clCreateKernel(state->program, "conv1dline_cl", &state->err);
    if (!state->kernel || state->err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    state->v_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * 4 , NULL, NULL);
    state->s_attn_cattn_b_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * 3 , NULL, NULL);
    state->s_attn_cattn_w_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE * state->WVSIZE *3, NULL, NULL);
    state->scratch_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * ((state->WVSIZE*4) +2), NULL, NULL);

    if (!state->v_data )
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
}


void set_parameters_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state)
{

 

    // Set the arguments to our compute kernel
    //
    state->err = 0;

    // Write our data set into the input array in device memory 
    //
    if (state->set_v == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->v_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->v, 0, NULL, NULL);
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
    if (state->set_scratch == 1)
    {
        state->err |= clEnqueueWriteBuffer(state->commands, state->scratch_data, CL_TRUE, 0, sizeof(float) * ((state->WVSIZE*4)+2), state->scratch, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }                                                        
    }


    if (state->set_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 0, sizeof(float), &state->b);
        state->set_b = 0;
    }    
 

    if (state->set_v == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->v_data);
        state->set_v = 0;
    }    

    if (state->set_m_offset == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 2, sizeof(unsigned int), &state->m_offset);
        state->set_m_offset = 0;
    }    

    if (state->set_size == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 3, sizeof(unsigned int), &state->size);
        state->set_size = 0;
    }    

    if (state->set_arraychoice == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 4, sizeof(unsigned int), &state->arraychoice);
        state->set_arraychoice = 0;
    }        
 
    if (state->set_s_attn_cattn_b == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 5, sizeof(cl_mem), &state->s_attn_cattn_b_data);
        state->set_s_attn_cattn_b = 0;
    }
    if (state->set_s_attn_cattn_w == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 6, sizeof(cl_mem), &state->s_attn_cattn_w_data);
        state->set_s_attn_cattn_w = 0;
    }   
    if (state->set_scratch == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 7, sizeof(cl_mem), &state->scratch_data);
        state->set_scratch = 0;
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

int execute_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state)
{
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    state->global = state->numCores;
    state->local = state->numCores;
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
    if (state->get_scratch == 1)
    {
        state->err = clEnqueueReadBuffer( state->commands, state->scratch_data, CL_TRUE, 0, sizeof(float) * state->size, state->v, 0, NULL, NULL );  
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", state->err);
            exit(1);
        }
        state->get_scratch = 0;

    }     

    
}


    
int release_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state)
{
    // Shutdown and cleanup
    //

    clReleaseMemObject(state->v_data);
    clReleaseMemObject(state->s_attn_cattn_b_data);
    clReleaseMemObject(state->s_attn_cattn_w_data);
    clReleaseMemObject(state->scratch_data);
        
    clReleaseProgram(state->program);
    clReleaseKernel(state->kernel);
    clReleaseCommandQueue(state->commands);
    clReleaseContext(state->context);

    free(state->v);
    free(state->scratch);

    return 0;
}

