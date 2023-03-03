#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <ctype.h>
#ifdef HAVE_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#endif

#include "structs.h"
#include "unpickler.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void Unpickler_load_data_pkl(layerfiles_t *layerfiles, uint8_t *buffer, int size)
    {

        unpickler::Unpickler unpickler;
        unpickler::PickleObject *object = unpickler.loads((const char *)buffer, size);

        // get indices of each file
        for (int j = 0; j < layerfiles->numfiles; j++)
        {
            layerfiles->tmpindex[j] = -1;
            layerfiles->index[j] = -1;
        }
        for (int j = 0; j < layerfiles->numfiles; j++)
        {
            if (j == 121)
            {
                int q = 0;
                q++;
            }
            if (j == 238)
            {
                int q = 0;
                q++;
            }
            if (layerfiles->files[j] != NULL)
            {
                for (int i = 0; i < object->frames.size(); i++)
                {

                    if (object->frames[i]->opcode == 'c' ||
                        object->frames[i]->opcode == 'X')
                    {
                        if (object->frames[i]->content != NULL)
                        {
                            char *tempstr = (char *)malloc(object->frames[i]->frameSize+1);
                            memset(tempstr,0, object->frames[i]->frameSize+1);
                            memcpy(tempstr,object->frames[i]->content, object->frames[i]->frameSize );
                            if (strstr(tempstr, layerfiles->files[j]) != 0)
                            {
                                // get the index of the file
                                layerfiles->tmpindex[j] = i;
                            }
                            free(tempstr);

                        }
                    }
                }
            }
        }

        // // sort the list by buffer index
        for (int k = 0; k < layerfiles->numfiles; k++)
        {
            int minindex = -1;
            int minval = -1;
            int lastminval = -1;

            for (int j = 0; j < layerfiles->numfiles; j++)
            {
                if (layerfiles->tmpindex[j] != -1 && (minval == -1 || layerfiles->tmpindex[j] < minval))
                {
                    minval = layerfiles->tmpindex[j];
                    minindex = j;
                }
            }

            if (minval != -1)
            {
                lastminval = minval;
                layerfiles->index[minindex] = k;
                layerfiles->tmpindex[minindex] = -1;
            }
        }

        return;
    }

#ifdef __cplusplus
}
#endif
