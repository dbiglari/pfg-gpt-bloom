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


////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//


////////////////////////////////////////////////////////////////////////////////

char *KernelSource_linear_transform_cl;

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


////////////////////////////////////////////////////////////////////////////////

int linear_transform_cl_test()
{
    opencl_kernel_model_linear_transform_cl_t *state = NULL;
    state = linear_transform_cl_wrapper(state);

    state->initialize = 1;
    state->populate_data_for_test = 0;
    state->WVSIZE = WV_SIZE;
    state->CTXSIZE = CTX_SIZE;
    state->NUMHEADS = NUM_HEADS;
    state->NUMLAYERS = NUM_LAYERS;    
    state->HEADSIZE = state->WVSIZE / state->NUMHEADS;
    state->closest_power_of_2 = pow(2, floor(log2(state->NUMHEADS)));
    state->y  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->WVSIZE);
    state->input  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size);
    state->output  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->output_size);
    state->weights  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size * state->output_size);
    state->bias  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size);

    state->here = 0;
    
    for(int i = 0; i < state->WVSIZE; i++)
    {
        state->x[i] = rand() / (float)RAND_MAX;
        state->y[i] = rand() / (float)RAND_MAX;
    }  
    linear_transform_cl_wrapper(state);

    state->setparams = 1;

    state->set_input = 1;
    state->set_output = 1;
    state->set_weights = 1;
    state->set_bias = 1;
    state->set_input_size = 1;
    state->set_output_size = 1;


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
    linear_transform_cl_wrapper(state);


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
      //  printf ("uploading layer %d ", i);
        
    struct timespec begin1, end1; 
    clock_gettime(CLOCK_REALTIME, &begin1);


        linear_transform_cl_wrapper(state);

    // Stop measuring time and calculate the elapsed time
    clock_gettime(CLOCK_REALTIME, &end1);
    long seconds1 = end1.tv_sec - begin1.tv_sec;
    long nanoseconds1 = end1.tv_nsec - begin1.tv_nsec;
    double elapsed1 = seconds1 + nanoseconds1*1e-9;
    
   // printf("Time measured: %.9f seconds.\n", elapsed1);     
   // fflush(stdout);
   
        
        state->execute = 1; 
        state->get_y = 1;
  //      printf ("executing layer %d ", i);
    

    struct timespec begin2, end2; 
    clock_gettime(CLOCK_REALTIME, &begin2);

    linear_transform_cl_wrapper(state);


    // Stop measuring time and calculate the elapsed time
    clock_gettime(CLOCK_REALTIME, &end2);
    long seconds2 = end2.tv_sec - begin2.tv_sec;
    long nanoseconds2 = end2.tv_nsec - begin2.tv_nsec;
    double elapsed2 = seconds2 + nanoseconds2*1e-9;
    
   // printf("Time measured: %.9f seconds.\n", elapsed2);     
   // fflush(stdout);        
    }
    
    // Stop measuring time and calculate the elapsed time
    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;
    
    printf("Time measured: %.9f seconds.\n", elapsed);
    
    return 0;

    state->cleanup = 1;
    linear_transform_cl_wrapper(state);


}

#ifdef BUILD_TEST
int main(int argc, char** argv)
{
    linear_transform_cl_test();
}
#endif

opencl_kernel_model_linear_transform_cl_t *linear_transform_cl_wrapper(opencl_kernel_model_linear_transform_cl_t *state)
{


    if (state == NULL)
    {
        state = (opencl_kernel_model_linear_transform_cl_t *) malloc(sizeof(opencl_kernel_model_linear_transform_cl_t));
        memset(state, 0, sizeof(opencl_kernel_model_linear_transform_cl_t));
        return state;
    }
    
    if(state->initialize == 1)
    {
        initialize_linear_transform_cl(state);
        state->initialize = 0;
        return state;
    }

    if (state->setparams == 1)
    {
        set_parameters_linear_transform_cl(state);
        state->setparams = 0;
        return state;
    }    
   
    if (state->execute == 1)
    {
        execute_linear_transform_cl(state);
        state->execute = 0;
        return state;
    }

    if (state->cleanup == 1)
    {
        release_linear_transform_cl(state); 
        state->cleanup = 0;
        return state;
    }    

    return state;

}


int initialize_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state)
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

        state->y  = (float *) malloc(sizeof(float) * state->WVSIZE);


        state->closest_power_of_2 = pow(2, floor(log2(state->NUMHEADS)));
        state->y  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->WVSIZE);
        state->input  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size);
        state->output  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->output_size);
        state->weights  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size * state->output_size);
        state->bias  = (float *) malloc_wrapper(&(state->total_malloc), sizeof(float) * state->input_size);

        state->here = 0;

        for(i = 0; i < state->WVSIZE; i++)
        {
            state->y[i] = rand() / (float)RAND_MAX;
        }

        state->set_input = 1;
        state->set_output = 1;
        state->set_weights = 1;
        state->set_bias = 1;
        state->set_input_size = 1;
        state->set_output_size = 1;
        
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

    KernelSource_linear_transform_cl = readfile("linear_transform_cl.cl", &state->length, ".");

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
    state->commands = clCreateCommandQueueWithProperties(state->context, state->devices[state->useDeviceNum], 0, &state->err);
    if (!state->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    state->program = clCreateProgramWithSource(state->context, 1, (const char **) & KernelSource_linear_transform_cl, NULL, &state->err);
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
    state->kernel = clCreateKernel(state->program, "linear_transform_cl", &state->err);
    if (!state->kernel || state->err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //


    state->input_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->input_size, NULL, NULL);
    state->output_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->output_size, NULL, NULL);
    state->weights_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->input_size * state->output_size, NULL, NULL);
    state->bias_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->input_size, NULL, NULL);

    state->y_data = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->WVSIZE , NULL, NULL);

    if (!state->y_data )
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
}


void set_parameters_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state)
{

 

    // Set the arguments to our compute kernel
    //
    state->err = 0;

    // Write our data set into the input array in device memory 
    //

    if (state->set_y == 1)
    {    
        state->err |= clEnqueueWriteBuffer(state->commands, state->y_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->y, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }    
    }

    if (state->set_input == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->input_data, CL_TRUE, 0, sizeof(float) * state->input_size, state->input, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }

    if (state->set_output == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->output_data, CL_TRUE, 0, sizeof(float) * state->output_size, state->output, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }

    if (state->set_weights == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->weights_data, CL_TRUE, 0, sizeof(float) * state->input_size * state->output_size, state->weights, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }

    if (state->set_bias == 1)
    {
        state->err = clEnqueueWriteBuffer(state->commands, state->bias_data, CL_TRUE, 0, sizeof(float) * state->input_size, state->bias, 0, NULL, NULL);
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
    }                
   

 
    if (state->set_input == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->input_data);
        state->set_input = 0;
    }

    if (state->set_output == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->output_data);
        state->set_output = 0;
    }

    if (state->set_weights == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &state->weights_data);
        state->set_weights = 0;
    }    

    if (state->set_bias == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 3, sizeof(cl_mem), &state->bias_data);
        state->set_bias = 0;
    }    

    if (state->set_input_size == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 4, sizeof(unsigned int), &state->input_size);
        state->set_input_size = 0;
    }

    if (state->set_output_size == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 5, sizeof(unsigned int), &state->output_size);
        state->set_output_size = 0;
    }        
    if (state->set_y == 1)
    {
        state->err |= clSetKernelArg(state->kernel, 6, sizeof(cl_mem), &state->y_data);
        state->set_y = 0;
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

int execute_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state)
{
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    state->global = state->numCores_global;
    state->local = state->numCores_local;
    
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
    if (state->get_y == 1)
    {
        state->err = clEnqueueReadBuffer( state->commands, state->y_data, CL_TRUE, 0, sizeof(float) * state->WVSIZE, state->y, 0, NULL, NULL );  
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", state->err);
            exit(1);
        }
        state->get_y = 0;

    }

    if (state->get_output == 1)
    {
        state->err = clEnqueueReadBuffer( state->commands, state->output_data, CL_TRUE, 0, sizeof(float) * state->output_size, state->output, 0, NULL, NULL );  
        if (state->err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", state->err);
            exit(1);
        }
        state->get_output = 0;

    }    

    
}


    
int release_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state)
{
    // Shutdown and cleanup
    //
    clReleaseMemObject(state->y_data);

    clReleaseMemObject(state->input_data);
    clReleaseMemObject(state->output_data);
    clReleaseMemObject(state->weights_data);
    clReleaseMemObject(state->bias_data);

    clReleaseProgram(state->program);
    clReleaseKernel(state->kernel);
    clReleaseCommandQueue(state->commands);
    clReleaseContext(state->context);

    free(state->x);
    free(state->y);

    return 0;
}

void Initialize_OpenCL_For_Model_linear_transform_cl(int modelnum)
{

    for (int i=0;i<models[modelnum].NUMLAYERS;i++)
    {

        // initialize layer based on model (first time)
        if (models[modelnum].state_linear_transform_cl == NULL)
        {
            // initialize opencl context for this layer
            models[modelnum].state_linear_transform_cl = linear_transform_cl_wrapper(NULL);
            models[modelnum].state_linear_transform_cl->initialize = 1;
            models[modelnum].state_linear_transform_cl->numCores_local = 16;
            models[modelnum].state_linear_transform_cl->numCores_global = 4096;
            models[modelnum].state_linear_transform_cl->populate_data_for_test = 0;
            models[modelnum].state_linear_transform_cl->WVSIZE = models[modelnum].WVSIZE;
            models[modelnum].state_linear_transform_cl->CTXSIZE = models[modelnum].CTXSIZE;
            models[modelnum].state_linear_transform_cl->NUMHEADS = models[modelnum].NUMHEADS;
            models[modelnum].state_linear_transform_cl->NUMLAYERS = models[modelnum].NUMLAYERS;    
            models[modelnum].state_linear_transform_cl->HEADSIZE = models[modelnum].state_linear_transform_cl->WVSIZE / models[modelnum].state_linear_transform_cl->NUMHEADS;
            models[modelnum].state_linear_transform_cl->closest_power_of_2 = pow(2, floor(log2(models[modelnum].state_linear_transform_cl->NUMHEADS)));
            linear_transform_cl_wrapper(models[modelnum].state_linear_transform_cl);
            models[modelnum].state_linear_transform_cl->setparams = 1;

            models[modelnum].state_linear_transform_cl->y  = (float *) malloc_wrapper(&models[modelnum].state_linear_transform_cl->total_malloc, sizeof(float) * models[modelnum].state_linear_transform_cl->WVSIZE);

            models[modelnum].state_linear_transform_cl->weights  = models[modelnum].wte;
            models[modelnum].state_linear_transform_cl->bias = (float *) malloc_wrapper(&models[modelnum].state_linear_transform_cl->total_malloc, sizeof(float) * models[modelnum].state_linear_transform_cl->input_size);
            
            models[modelnum].state_linear_transform_cl->layeridx = i;
            


            models[modelnum].state_linear_transform_cl->set_input = 1;
            models[modelnum].state_linear_transform_cl->set_output = 1;
            models[modelnum].state_linear_transform_cl->set_weights = 1;
            models[modelnum].state_linear_transform_cl->set_bias = 1;
            models[modelnum].state_linear_transform_cl->set_input_size = 1;
            models[modelnum].state_linear_transform_cl->set_output_size = 1;            
           
            models[modelnum].state_linear_transform_cl->set_WVSIZE = 1;
            models[modelnum].state_linear_transform_cl->set_CTXSIZE = 1;
            models[modelnum].state_linear_transform_cl->set_HEADSIZE = 1;
            models[modelnum].state_linear_transform_cl->set_NUMHEADS = 1;
            models[modelnum].state_linear_transform_cl->set_NUMLAYERS = 1;
            models[modelnum].state_linear_transform_cl->set_layeridx = 1;
            models[modelnum].state_linear_transform_cl->set_closest_power_of_2 = 1;
            models[modelnum].state_linear_transform_cl->set_y = 1;
            models[modelnum].state_linear_transform_cl->set_here = 1;    
            models[modelnum].state_linear_transform_cl->get_max_workgroup = 1;    
            linear_transform_cl_wrapper(models[modelnum].state_linear_transform_cl);       
        }
    }

}


void linear_transform_cl(float *input, float *output, float *weights, float *bias, long long input_size, long long output_size, int modelnum, int querynum)
{
    runlinear_transform_cl(input, output,weights, bias, input_size, output_size, modelnum, querynum);
}

void runlinear_transform_cl(float *input, float *output, float *weights, float *bias, long long input_size, long long output_size, int modelnum, int querynum)
{
    // initialize layer based on model (first time)
    if (models[modelnum].state_linear_transform_cl == NULL)
    {
        models[modelnum].state_linear_transform_cl = linear_transform_cl_wrapper(NULL);
        
        models[modelnum].state_linear_transform_cl->initialize = 1;
        models[modelnum].state_linear_transform_cl->populate_data_for_test = 0;       
        
        models[modelnum].state_linear_transform_cl->WVSIZE = models[modelnum].WVSIZE;
        models[modelnum].state_linear_transform_cl->CTXSIZE = models[modelnum].CTXSIZE;
        models[modelnum].state_linear_transform_cl->NUMHEADS = models[modelnum].NUMHEADS;
        models[modelnum].state_linear_transform_cl->NUMLAYERS = models[modelnum].NUMLAYERS;    
        models[modelnum].state_linear_transform_cl->HEADSIZE = models[modelnum].state_linear_transform_cl->WVSIZE / models[modelnum].state_linear_transform_cl->NUMHEADS;
        models[modelnum].state_linear_transform_cl->closest_power_of_2 = pow(2, floor(log2(models[modelnum].state_linear_transform_cl->NUMHEADS)));

        models[modelnum].state_linear_transform_cl->input_size = input_size;
        models[modelnum].state_linear_transform_cl->output_size = output_size;

        linear_transform_cl_wrapper(models[modelnum].state_linear_transform_cl);

        models[modelnum].state_linear_transform_cl->setparams = 1;
        models[modelnum].state_linear_transform_cl->y  = (float *) malloc_wrapper(&models[modelnum].state_linear_transform_cl->total_malloc, sizeof(float) * models[modelnum].state_linear_transform_cl->WVSIZE);        
        
        models[modelnum].state_linear_transform_cl->y  = (float *) malloc_wrapper(&models[modelnum].state_linear_transform_cl->total_malloc, sizeof(float) * 1);
        models[modelnum].state_linear_transform_cl->weights  = weights;


        if (bias != NULL)
        {
            models[modelnum].state_linear_transform_cl->bias = bias;
        }
        else
        {
            models[modelnum].state_linear_transform_cl->bias = (float *) malloc_wrapper(&models[modelnum].state_linear_transform_cl->total_malloc, sizeof(float) * models[modelnum].state_linear_transform_cl->input_size);
            memset(models[modelnum].state_linear_transform_cl->bias, 0, sizeof(float) * models[modelnum].state_linear_transform_cl->input_size);
        }
        
        models[modelnum].state_linear_transform_cl->set_output = 1;
        models[modelnum].state_linear_transform_cl->output = output;    
        models[modelnum].state_linear_transform_cl->set_weights = 1;
        models[modelnum].state_linear_transform_cl->set_bias = 1;
        models[modelnum].state_linear_transform_cl->set_input_size = 1;
        models[modelnum].state_linear_transform_cl->set_output_size = 1;            
        



        models[modelnum].state_linear_transform_cl->set_WVSIZE = 1;
        models[modelnum].state_linear_transform_cl->set_CTXSIZE = 1;
        models[modelnum].state_linear_transform_cl->set_HEADSIZE = 1;
        models[modelnum].state_linear_transform_cl->set_NUMHEADS = 1;
        models[modelnum].state_linear_transform_cl->set_NUMLAYERS = 1;
        models[modelnum].state_linear_transform_cl->set_layeridx = 1;
        models[modelnum].state_linear_transform_cl->set_closest_power_of_2 = 1;
        models[modelnum].state_linear_transform_cl->set_y = 1;
        models[modelnum].state_linear_transform_cl->set_here = 1;        
    }


    models[modelnum].state_linear_transform_cl->setparams = 1;
    models[modelnum].state_linear_transform_cl->set_input = 1;    
    models[modelnum].state_linear_transform_cl->input = input;    
    models[modelnum].state_linear_transform_cl->set_y = 1;
    models[modelnum].state_linear_transform_cl->y[0] = 0;    
    linear_transform_cl_wrapper(models[modelnum].state_linear_transform_cl); 

    models[modelnum].state_linear_transform_cl->execute = 1;
    stopwatch_start(&begin_glob);
    models[modelnum].state_linear_transform_cl->numCores_local = 256;
    models[modelnum].state_linear_transform_cl->numCores_global = 16384;        
    models[modelnum].state_linear_transform_cl->get_output = 1;
    linear_transform_cl_wrapper(models[modelnum].state_linear_transform_cl);       
    stopwatch_end("greedy lmlogit transform", begin_glob, false);

    //memcpy(output, models[modelnum].state_linear_transform_cl->output, sizeof(float)* models[modelnum].state_linear_transform_cl->output_size);
    //exit(0);
    

}