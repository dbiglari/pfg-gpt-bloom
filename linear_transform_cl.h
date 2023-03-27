#include <CL/cl.h>
typedef struct opencl_kernel_model_linear_transform_cl_t
{
    int err;                            // error code returned from api calls

    // layer input
    float *x;
    // layer data
    float *xn;

    float *input;
    float *output;
    float *weights;
    float *bias;
    unsigned int input_size;
    unsigned int output_size;

 

    // model parameters
    unsigned int WVSIZE;
    unsigned int CTXSIZE;
    unsigned int HEADSIZE;
    unsigned int NUMHEADS;
    unsigned int NUMLAYERS;    
    unsigned int layeridx;
    float closest_power_of_2;
    // layer output
    float *y;
    unsigned int here;


    // flags for set_parameters
    int set_x;
    int set_xn;
    int set_input;
    int set_output;
    int set_weights;
    int set_bias;
    int set_input_size;
    int set_output_size;

    int set_WVSIZE;
    int set_CTXSIZE;
    int set_HEADSIZE;
    int set_NUMHEADS;
    int set_NUMLAYERS;    
    int set_layeridx;
    int set_closest_power_of_2;
    int set_y;
    int set_here;
    int get_max_workgroup;
    int get_output;
    int get_y;
    int get_x;
    int get_xn;    

    // opencl specific structures
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    // opencl arrays
    cl_mem x_data;
    cl_mem xn_data;

    cl_mem input_data;
    cl_mem output_data;
    cl_mem weights_data;
    cl_mem bias_data;

    cl_mem y_data;


    cl_uint numPlatforms; //the NO. of platforms
    cl_platform_id platform; //the chosen platform    
    cl_int status;
    int length;
    cl_uint numDevices;
    int gpu;
    cl_device_id *devices;   
    int populate_data_for_test;

    // opencl action flow control
    int numCores_local;
    int numCores_global;
    int useDeviceNum;
    int execute;
    int initialize;
    int cleanup;
    int setparams;

    size_t total_malloc;

} opencl_kernel_model_linear_transform_cl_t;


void linear_transform_cl(float *input, float *output, float *weights, float *bias, long long input_size, long long output_size, int modelnum, int querynum);
int initialize_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state);
void set_parameters_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state);
int execute_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state);
int release_linear_transform_cl(opencl_kernel_model_linear_transform_cl_t *state);
opencl_kernel_model_linear_transform_cl_t *linear_transform_cl_wrapper(opencl_kernel_model_linear_transform_cl_t *state);
void runAllLayers_cl(float *x, int here, int modelnum, int querynum);
void runlinear_transform_cl(float *input, float *output, float *weights, float *bias, long long input_size, long long output_size, int modelnum, int querynum);
void Initialize_OpenCL_For_Model_linear_transform_cl(int modelnum);
