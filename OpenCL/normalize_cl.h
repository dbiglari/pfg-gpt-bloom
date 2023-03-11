
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

    int useDeviceNum;
    int execute;
    int initialize;
    int cleanup;    

} opencl_kernel_normalize_cl_t;



void initialize_normalize_cl(opencl_kernel_normalize_cl_t *state);
void set_parameters_normalize_cl(opencl_kernel_normalize_cl_t *state);
void execute_normalize_cl(opencl_kernel_normalize_cl_t *state);
void release_normalize_cl(opencl_kernel_normalize_cl_t *state);