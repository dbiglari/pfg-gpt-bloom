
typedef struct {
    int index;
    int memoryUsage;
    int totalMemory;
} GPUInfo;

int findMaxRAMIndex(const GPUInfo gpus[], int numGPUs);
int GetNVIDIALinuxGPUInfo(GPUInfo **gpus, int *rows);
GPUInfo *parseGPUInfo(const char *input, int *rows);
int countLines(const char *str);
int extractFirstInteger(const char *str);
