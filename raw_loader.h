#define BLOOM_560m "/media/dbiglari/ML_Data/bloom/models--bigscience--bloom-560m/snapshots/afe2e6f33eb135d254df849c74bb83322b53641c"
#define BLOOMZ_560m "/media/dbiglari/ML_Data/bloom/models--bigscience--bloomz-560m/snapshots/e183ebbd45c58bdd406d19e907ed5f36b526f752"
#define BLOOM_3b "/media/silicon-admin/37f14fc8-6e78-4c4b-a7ac-36716515787f/data/bloom/models--bigscience--bloom-3b/snapshots/515ae965cc83b9ebbf0054de106c434bd4ec35dc"
#define BLOOMZ_3b "/media/silicon-admin/37f14fc8-6e78-4c4b-a7ac-36716515787f/data/bloom/models--bigscience--bloomz-3b/snapshots/2d4a819e9aa8cc96718a5be4cb0c350b5642c3f0/"
#define BLOOM_7b1 "/media/silicon-admin/37f14fc8-6e78-4c4b-a7ac-36716515787f/data/bloom/models--bigscience--bloom-7b1/snapshots/850ba1758a7744fedae78caadc152625133b1677"
#define BLOOMZ_7b1 "/media/silicon-admin/37f14fc8-6e78-4c4b-a7ac-36716515787f/data/bloom/models--bigscience--bloomz-7b1/snapshots/229c446c3c5263165f376b4b19ddedbeeab3b575"
#define BLOOM_175b "/media/silicon-admin/6e27c8dd-118e-4cae-a3d4-e12a286a3cc1/data/bloom/models--bigscience--bloom/snapshots/f0e3b92526b687d0b1efe041876c806b6316c1e0"


#define NUMMODELS 9

typedef struct model_path_t
{
    char modelname[256];
    char modelpath[2048];
} model_path_t;

int load_huggingface_bloom_model_folder(char *path, int modelindex);
void load_layer_container(int modelindex, int layernum);
void unload_layer_container(int modelindex, int layernum);
char *lookup_model_path(char *modelname);
char *read_file(char *path);
void load_layer_container_thr(int modelindex, int layernum, int thr);
int load_model_paths(char *configfile);