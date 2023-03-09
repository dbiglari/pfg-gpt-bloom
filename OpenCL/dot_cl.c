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
//#include <CL/c.h>
#include "dot_cl.h"

#define WV_SIZE (4096)



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

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    opencl_kernel_dot_cl_t *state = NULL;

    // allocate the state for the cl kernel
    dot_cl_wrapper(state, 0, NULL, NULL, 0);


    state->initialize = 1;
    state->populate_data_for_test = 0;
    // initialize state for cl kernel
    state->count = 4096;
    state->m_size = 4096*1024;
    dot_cl_wrapper(state, 0, NULL, NULL, 0);


    //opencl_kernel_dot_cl_t state={0};
    //state.useDeviceNum=0;
    //state.populate_data_for_test = 1;
    // initialize_dot_cl(&state);
    // set_parameters_dot_cl(&state);
    // printf ("initialization complete\n");
    // fflush(stdout);
    // execute_dot_cl(&state);
    // printf ("execution complete\n");
    // fflush(stdout);    
    // release_dot_cl(&state);    
}


opencl_kernel_dot_cl_t *dot_cl_wrapper(opencl_kernel_dot_cl_t *state, float a, float *v, float *m, long long wdt)
{
    if (state == NULL)
    {
        state = (opencl_kernel_dot_cl_t *) malloc(sizeof(opencl_kernel_dot_cl_t));
        memset(state, 0, sizeof(opencl_kernel_dot_cl_t));
        return;
    }
    
    if(state->initialize == 1)
    {
        initialize_dot_cl(state);
        return;
    }

    if (state->setparams == 1)
    {
        set_parameters_dot_cl(state);
        return;
    }    
   
    if (state->execute == 1)
    {
        execute_dot_cl(state);
        return;
    }

    if (state->cleanup == 1)
    {
        release_dot_cl(state); 
        return;
    }    

    return state;

}


void initialize_dot_cl(opencl_kernel_dot_cl_t *state)
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
        state->count = WV_SIZE;
        state->wdt = state->count;
        state->m_size = state->count;
        state->v_data = (float *) malloc(sizeof(float) * state->count);
        state->m_data = (float *) malloc(sizeof(float) * state->m_size);
        state->a_data = (float *) malloc(sizeof(float) * state->count);

        for(i = 0; i < state->count; i++)
        {
            state->v_data[i] = 1.0/((float)(i+1));//rand() / (float)RAND_MAX;
        }
        for(i = 0; i< state->m_size; i++)
        {
            state->m_data[i] = 1.0/((float)(i+1));//rand() / (float)RAND_MAX;
        }
    }

    KernelSource = readfile("dot_cl.cl", &state->length, ".");

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
    state->err = clBuildProgram(state->program, 0, NULL, NULL, NULL, NULL);
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
    state->kernel = clCreateKernel(state->program, "dot_cl", &state->err);
    if (!state->kernel || state->err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    state->v = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->count, NULL, NULL);
    state->a = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float) * state->count, NULL, NULL);
    state->m = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float) * state->m_size, NULL, NULL);
    if (!state->v || !state->a || !state->m )
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
}


void set_parameters_dot_cl(opencl_kernel_dot_cl_t *state)
{

 

    // Set the arguments to our compute kernel
    //
    state->err = 0;
    if (state->set_v==1)
    {

        // Write our data set into the input array in device memory 
        //
        state->err = clEnqueueWriteBuffer(state->commands, state->v, CL_TRUE, 0, sizeof(float) * state->count, state->v_data, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
                
        state->err  = clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->v);
    }
    if (state->set_a==1)
    {


        state->err = clEnqueueWriteBuffer(state->commands, state->a, CL_TRUE, 0, sizeof(float) * state->count, state->a_data, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }   

        state->err |= clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->a);
    }
    if (state->set_m==1)
    {


        state->err = clEnqueueWriteBuffer(state->commands, state->m, CL_TRUE, 0, sizeof(float) * state->m_size, state->m_data, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }   
                
        state->err |= clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &state->m);
    }
    if (state->set_m_start==1)        
    {
        state->err |= clSetKernelArg(state->kernel, 3, sizeof(unsigned long), &state->m_start);
    }
    if (state->set_wdt==1)        
    {
        state->err |= clSetKernelArg(state->kernel, 4, sizeof(unsigned long), &state->wdt);
    }
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", state->err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    state->err = clGetKernelWorkGroupInfo(state->kernel, state->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(state->local), &state->local, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", state->err);
        exit(1);
    }

}

void execute_dot_cl(opencl_kernel_dot_cl_t *state)
{
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    state->global = state->count;
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
    state->err = clEnqueueReadBuffer( state->commands, state->a, CL_TRUE, 0, sizeof(float) * state->count, state->a_data, 0, NULL, NULL );  
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", state->err);
        exit(1);
    }

    
}


    
void release_dot_cl(opencl_kernel_dot_cl_t *state)
{
    // Shutdown and cleanup
    //
    clReleaseMemObject(state->v);
    clReleaseMemObject(state->a);
    clReleaseMemObject(state->m);
        
    clReleaseProgram(state->program);
    clReleaseKernel(state->kernel);
    clReleaseCommandQueue(state->commands);
    clReleaseContext(state->context);

    free(state->v_data);
    free(state->m_data);
    free(state->a_data);

    return 0;
}

