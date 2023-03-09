
typedef struct opencl_kernel_layer_cl_t
{
    int err;                            // error code returned from api calls

    // layer input
    float *x;
    // layer data
    float *s_ln1_b;
    float *s_ln1_g;
    float *s_ln2_b;
    float *s_ln2_g;
    float *s_mlp_cfc_b;
    float *s_mlp_cfc_w;
    float *s_mlp_cproj_b;
    float *s_mlp_cproj_w;
    float *s_attn_cattn_b;
    float *s_attn_cattn_w;
    float *s_attn_cproj_b;
    float *s_attn_cproj_w;
    float *att;
    float *attentions;
    float *attentions_presoftmax;
    // model parameters
    unsigned int WVSIZE;
    unsigned int CTXSIZE;
    unsigned int HEADSIZE;
    unsigned int NUMHEADS;
    unsigned int NUMLAYERS;    
    float closest_power_of_2;
    // layer output
    float *y;


    // flags for set_parameters
    int set_x;
    int set_s_ln1_b;
    int set_s_ln1_g;
    int set_s_ln2_b;
    int set_s_ln2_g;
    int set_s_mlp_cfc_b;
    int set_s_mlp_cfc_w;
    int set_s_mlp_cproj_b;
    int set_s_mlp_cproj_w;
    int set_s_attn_cattn_b;
    int set_s_attn_cattn_w;
    int set_s_attn_cproj_b;
    int set_s_attn_cproj_w;
    int set_att;
    int set_attentions;
    int set_attentions_presoftmax;
    int set_WVSIZE;
    int set_CTXSIZE;
    int set_HEADSIZE;
    int set_NUMHEADS;
    int set_NUMLAYERS;    
    int set_closest_power_of_2;
    int set_y;

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
    cl_mem s_ln1_b_data;
    cl_mem s_ln1_g_data;
    cl_mem s_ln2_b_data;
    cl_mem s_ln2_g_data;
    cl_mem s_mlp_cfc_b_data;
    cl_mem s_mlp_cfc_w_data;
    cl_mem s_mlp_cproj_b_data;
    cl_mem s_mlp_cproj_w_data;
    cl_mem s_attn_cattn_b_data;
    cl_mem s_attn_cattn_w_data;
    cl_mem s_attn_cproj_b_data;
    cl_mem s_attn_cproj_w_data;
    cl_mem att_data;
    cl_mem attentions_data;
    cl_mem attentions_presoftmax_data;
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
    int useDeviceNum;
    int execute;
    int initialize;
    int cleanup;
    int setparams;

    size_t total_malloc;

} opencl_kernel_layer_cl_t;


void initialize_layer_cl(opencl_kernel_layer_cl_t *state);
void set_parameters_layer_cl(opencl_kernel_layer_cl_t *state);
void execute_layer_cl(opencl_kernel_layer_cl_t *state);
void release_layer_cl(opencl_kernel_layer_cl_t *state);
opencl_kernel_layer_cl_t *layer_cl_wrapper(opencl_kernel_layer_cl_t *state);