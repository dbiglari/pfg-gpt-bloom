#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "getGPUinfo.h"

#define MAX_GPUS 1024  // Adjust this value based on the maximum number of GPUs in your system



int extractFirstInteger(const char *str) {
    const char *ptr = str;

    // Skip leading non-digit characters
    while (*ptr && (*ptr < '0' || *ptr > '9')) {
        ++ptr;
    }

    // Use atoi to convert the first encountered integer
    return atoi(ptr);
}


int countLines(const char *str) {
    int lines = 0;

    while (*str) {
        if (*str == '\n') {
            ++lines;
        }
        ++str;
    }

    // If the last line doesn't end with a newline, add one more line
    if (str[-1] != '\n' && str[-1] != '\r') {
        ++lines;
    }

    return lines;
}


GPUInfo *parseGPUInfo(const char *input, int *rows) {

    *rows =  countLines(input);
    GPUInfo *gpus = malloc(sizeof(GPUInfo) * (*rows));
    int numGPUs = 0;
    int index = 0;
    int state = 0;  // State machine: 0 - looking for index and name, 1 - looking for RAM info

    const char *token = strtok((char *)input, "\n");
    char *totalMemoryStr;
    while (token != NULL) {
        char name[50];

        char *ramindex = strstr(token, " | ");
        if (strstr(token, " | ") != NULL) {
           
           gpus[numGPUs].index = index++;
           
            // Find memory usage and total memory
            gpus[numGPUs].memoryUsage = extractFirstInteger(ramindex);
            
            totalMemoryStr = strstr(ramindex, "/");
            gpus[numGPUs].totalMemory = extractFirstInteger(totalMemoryStr);

            numGPUs++;
        }        

        token = strtok(NULL, "\n");
    }


    return gpus;
}


int GetNVIDIALinuxGPUInfo(GPUInfo **gpus, int *rows) {
    FILE *fp;
    char buffer[4096];  // Adjust buffer size as needed

    // Run the "nvidia-smi" command and capture its output
    fp = popen("nvidia-smi | grep Default", "r");
    if (fp == NULL) {
        perror("Error opening pipe");
        return -1;
    }

    // Read the command output into the buffer
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), fp);
    buffer[bytesRead] = '\0';  // Null-terminate the string

    // Close the pipe
    pclose(fp);

    // Parse and display GPU information (for debug purposes)
    // printf ("%s\n", buffer);
    *gpus = parseGPUInfo(buffer, rows);

    return 0;
}


int GetNVIDIAWindowsGPUInfo(GPUInfo **gpus, int *rows) {
    FILE *fp;
    char buffer[4096];  // Adjust buffer size as needed

    // Run the "nvidia-smi" command and capture its output
    fp = popen("nvidia-smi | findstr Default", "r");
    if (fp == NULL) {
        perror("Error opening pipe");
        return -1;
    }

    // Read the command output into the buffer
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), fp);
    buffer[bytesRead] = '\0';  // Null-terminate the string

    // Close the pipe
    pclose(fp);

    // Parse and display GPU information (for debug purposes)
    // printf ("%s\n", buffer);
    *gpus = parseGPUInfo(buffer, rows);

    return 0;
}

int findMaxRAMIndex(const GPUInfo gpus[], int numGPUs) {
    if (numGPUs <= 0) {
        // Handle the case where the array is empty
        return -1;
    }

    int maxIndex = 0;
    int maxRAM = gpus[0].totalMemory - gpus[0].memoryUsage;

    for (int i = 1; i < numGPUs; i++) {
        if ((gpus[i].totalMemory - gpus[i].memoryUsage) > maxRAM) {
            maxRAM = gpus[i].totalMemory - gpus[i].memoryUsage;
            maxIndex = i;
        }
    }

    return maxIndex;
}

// Future work, add ATI, Intel other GPU support

// int main(void)
// {
//     GPUInfo *gpus;
//     int rows;
//     GetNVIDIALinuxGPUInfo(&gpus, &rows);

//     printf("Detected %d GPUs:\n", rows);
//     for (int i = 0; i < rows; i++) {
//         printf("GPU %d: %dMiB / %dMiB\n", gpus[i].index, 
//                gpus[i].memoryUsage, gpus[i].totalMemory);
//     }    
// }