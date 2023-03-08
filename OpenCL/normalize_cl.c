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

#define WV_SIZE (14336)

typedef struct opencl_kernel_normalize_cl_t
{
    int err;                            // error code returned from api calls

    float *data;              // original data set given to device
    float *b_data;            // original b data set given to device
    float *g_data;            // original g data set given to device
    float eps_data[1];                  // original g data set given to device
    float *mean_val_data;             // original g data set given to device
    float *rstd_val_data;           // original g data set given to device
    float *results;           // results returned from device

    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    cl_mem b;
    cl_mem g;
    cl_mem eps;
    cl_mem mean_val;
    cl_mem rstd_val;

    unsigned long count;


    cl_uint numPlatforms; //the NO. of platforms
    cl_platform_id platform; //the chosen platform    

    cl_int status;

    int length;

    cl_uint numDevices;
    int gpu;
    cl_device_id        *devices;   
    int populate_data_for_test;

} opencl_kernel_normalize_cl_t;


void initialize_normalize_cl(opencl_kernel_normalize_cl_t *state);
void set_parameters_normalize_cl(opencl_kernel_normalize_cl_t *state);
void execute_normalize_cl(opencl_kernel_normalize_cl_t *state);
void release_normalize_cl(opencl_kernel_normalize_cl_t *state);

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
    opencl_kernel_normalize_cl_t state={0};
    initialize_normalize_cl(&state);
    set_parameters_normalize_cl(&state);
    printf ("initialization complete\n");
    fflush(stdout);
    execute_normalize_cl(&state);
    printf ("execution complete\n");
    fflush(stdout);    
    release_normalize_cl(&state);    
}

void initialize_normalize_cl(opencl_kernel_normalize_cl_t *state)
{




    state->numPlatforms; //the NO. of platforms
    state->platform = NULL; //the chosen platform    

    state->numDevices = 0;
    state->gpu = 1;
   

    
    
    // Fill our data set with random float values
    //
    int i = 0;

    if (state->populate_data_for_test == 0)
    {
        state->count = WV_SIZE;

        state->data = (float *) malloc(sizeof(float) * state->count);
        state->b_data = (float *) malloc(sizeof(float) * state->count);
        state->g_data = (float *) malloc(sizeof(float) * state->count);
        state->results = (float *) malloc(sizeof(float) * state->count);
        state->mean_val_data = (float *) malloc(sizeof(float) * state->count);
        state->rstd_val_data = (float *) malloc(sizeof(float) * state->count);    
        state->eps_data[0]=0.00001;                  // original g data set given to device

        for(i = 0; i < state->count; i++)
        {
            state->data[i] = 1;//rand() / (float)RAND_MAX;
            state->b_data[i] = i;//rand() / (float)RAND_MAX;
            state->g_data[i] = 1;//rand() / (float)RAND_MAX;
            state->mean_val_data[i]=0;             // original g data set given to device
            state->rstd_val_data[i]=0;           // original g data set given to device        
        }
    }

    KernelSource = readfile("normalize_cl.cl", &state->length, ".");

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
    state->context = clCreateContext(0, 1, state->devices, NULL, NULL, &state->err);
    if (!state->context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    state->commands = clCreateCommandQueue(state->context, state->devices[0], 0, &state->err);
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
    state->kernel = clCreateKernel(state->program, "normalize_cl", &state->err);
    if (!state->kernel || state->err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    state->input = clCreateBuffer(state->context,  CL_MEM_READ_ONLY,  sizeof(float) * state->count, NULL, NULL);
    state->output = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float) * state->count, NULL, NULL);
    state->b = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float) * state->count, NULL, NULL);
    state->g = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float) * state->count, NULL, NULL);
    state->eps = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);
    state->mean_val = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float)* state->count, NULL, NULL);
    state->rstd_val = clCreateBuffer(state->context, CL_MEM_WRITE_ONLY, sizeof(float)* state->count, NULL, NULL);
    if (!state->input || !state->output || !state->b || !state->g || !state->eps || !state->mean_val || !state->rstd_val)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    state->err = clEnqueueWriteBuffer(state->commands, state->input, CL_TRUE, 0, sizeof(float) * state->count, state->data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    state->err = clEnqueueWriteBuffer(state->commands, state->b, CL_TRUE, 0, sizeof(float) * state->count, state->b_data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    state->err = clEnqueueWriteBuffer(state->commands, state->g, CL_TRUE, 0, sizeof(float) * state->count, state->g_data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    state->err = clEnqueueWriteBuffer(state->commands, state->eps, CL_TRUE, 0, sizeof(float), state->eps_data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    state->err = clEnqueueWriteBuffer(state->commands, state->mean_val, CL_TRUE, 0, sizeof(float)* state->count, state->mean_val_data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    state->err = clEnqueueWriteBuffer(state->commands, state->rstd_val, CL_TRUE, 0, sizeof(float)* state->count, state->rstd_val_data, 0, NULL, NULL);
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }            
}


void set_parameters_normalize_cl(opencl_kernel_normalize_cl_t *state)
{

    // Set the arguments to our compute kernel
    //
    state->err = 0;
    state->err  = clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &state->input);
    state->err |= clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &state->output);
    state->err |= clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &state->b);
    state->err |= clSetKernelArg(state->kernel, 3, sizeof(cl_mem), &state->g);
    state->err |= clSetKernelArg(state->kernel, 4, sizeof(cl_mem), &state->eps);
    state->err |= clSetKernelArg(state->kernel, 5, sizeof(cl_mem), &state->mean_val);
    state->err |= clSetKernelArg(state->kernel, 6, sizeof(cl_mem), &state->rstd_val);
    state->err |= clSetKernelArg(state->kernel, 7, sizeof(unsigned long), &state->count);
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

void execute_normalize_cl(opencl_kernel_normalize_cl_t *state)
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
    state->err = clEnqueueReadBuffer( state->commands, state->output, CL_TRUE, 0, sizeof(float) * state->count, state->results, 0, NULL, NULL );  
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", state->err);
        exit(1);
    }

    // Read back the results from the device to verify the output
    //
    state->err = clEnqueueReadBuffer( state->commands, state->mean_val, CL_TRUE, 0, sizeof(float)* state->count, state->mean_val_data, 0, NULL, NULL );  
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", state->err);
        exit(1);
    }

    // Read back the results from the device to verify the output
    //
    state->err = clEnqueueReadBuffer( state->commands, state->rstd_val, CL_TRUE, 0, sizeof(float)* state->count, state->rstd_val_data, 0, NULL, NULL );  
    if (state->err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", state->err);
        exit(1);
    }        
    
}


    
void release_normalize_cl(opencl_kernel_normalize_cl_t *state)
{
    // Shutdown and cleanup
    //
    clReleaseMemObject(state->input);
    clReleaseMemObject(state->output);
    clReleaseMemObject(state->b);
    clReleaseMemObject(state->g);
    clReleaseMemObject(state->eps);
    clReleaseMemObject(state->mean_val);
    clReleaseMemObject(state->rstd_val);
        
    clReleaseProgram(state->program);
    clReleaseKernel(state->kernel);
    clReleaseCommandQueue(state->commands);
    clReleaseContext(state->context);

    free(state->data);
    free(state->b_data);
    free(state->g_data);
    free(state->results);
    free(state->mean_val_data);
    free(state->rstd_val_data);

    return 0;
}

