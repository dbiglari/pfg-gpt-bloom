
typedef struct opencl_kernel_dot_cl_t
{
    int err;                            // error code returned from api calls

    float *v_data;              // original data set given to device
    float *m_data;            // original b data set given to device
    float *a_data;           // results returned from device

    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem v;                       // device memory used for the input array
    cl_mem a;                      // device memory used for the output array
    cl_mem m;
    unsigned long m_size;
    unsigned long m_start;
    unsigned long wdt;

    unsigned long count;


    cl_uint numPlatforms; //the NO. of platforms
    cl_platform_id platform; //the chosen platform    

    cl_int status;

    int length;

    cl_uint numDevices;
    int gpu;
    cl_device_id        *devices;   
    int populate_data_for_test;

    int set_v;
    int set_a;
    int set_m;
    int set_m_start;        
        
    int set_wdt;  
    int useDeviceNum;
    int execute;
    int initialize;
    int cleanup;
    int setparams;

} opencl_kernel_dot_cl_t;


void initialize_dot_cl(opencl_kernel_dot_cl_t *state);
void set_parameters_dot_cl(opencl_kernel_dot_cl_t *state);
void execute_dot_cl(opencl_kernel_dot_cl_t *state);
void release_dot_cl(opencl_kernel_dot_cl_t *state);
opencl_kernel_dot_cl_t *dot_cl_wrapper(opencl_kernel_dot_cl_t *state, float a, float *v, float *m, long long wdt);