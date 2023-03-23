#include <CL/cl.h>
typedef struct opencl_kernel_model_conv1dline_cl_t
{
    int err;                            // error code returned from api calls

    // layer input
    float b;
    float *v;
    float *s_attn_cattn_b;
    float *s_attn_cattn_w;
    float *scratch;
    float *output;
    int WVSIZE;
    int size;
    int m_offset;
    int arraychoice;
    float a;

    // flags for set_parameters
    int set_b;
    int set_v;
    int set_s_attn_cattn_b;
    int set_s_attn_cattn_w;
    int set_scratch;
    int set_output;
    int get_scratch;
    int set_size;
    int set_m_offset;
    int set_arraychoice;
    int get_v;    
    int get_a;
    int get_output;

    // opencl specific structures
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    // opencl arrays
    cl_mem v_data;
    cl_mem s_attn_cattn_b_data;
    cl_mem s_attn_cattn_w_data;
    cl_mem scratch_data;
    cl_mem output_data;

    cl_uint numPlatforms; //the NO. of platforms
    cl_platform_id platform; //the chosen platform    
    cl_int status;
    int length;
    cl_uint numDevices;
    int gpu;
    cl_device_id *devices;   
    int populate_data_for_test;
    int get_max_workgroup;

    // opencl action flow control
    int numCores;
    int useDeviceNum;
    int execute;
    int initialize;
    int cleanup;
    int setparams;

    size_t total_malloc;

} opencl_kernel_model_conv1dline_cl_t;


float conv1dline_cl(float a, float *v, int m_offset, int size, int arraychoice, int modelnum, int layeridx, int thr);
int initialize_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state);
void set_parameters_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state);
int execute_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state);
int release_conv1dline_cl(opencl_kernel_model_conv1dline_cl_t *state);
opencl_kernel_model_conv1dline_cl_t *conv1dline_cl_wrapper(opencl_kernel_model_conv1dline_cl_t *state);
void runconv1dline_cl(float *x, int layeridx, int here, int modelnum, int querynum);
void Initialize_OpenCL_For_Model(int modelnum);
