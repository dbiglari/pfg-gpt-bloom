#include "common.h"
#include "fp32to8bit.h"


int8_t convertfloatto8bit(float val, float maxfloatval)
{
    int8_t val_int = (int8_t)((val/maxfloatval)/255.0);
    return val_int;
}

int8_t *convert1dfloatarrayto8bit(float *array, int size, float maxfloatval)
{
    int8_t *outputarray = (int8_t *) malloc(size);
    for (int i=0;i<size;i++)
    {
        convertfloatto8bit(array[i], maxfloatval);
    }
    return outputarray;
}


int8_t *convert2dlinearfloatarrayto8bit(float *array, int sizex, int sizey, float maxfloatval)
{
    int8_t *outputarray = (int8_t *) malloc(sizex*sizey);
    for (int i=0;i<sizex*sizey;i++)
    {
        convertfloatto8bit(array[i], maxfloatval);
    }
    return outputarray;
}