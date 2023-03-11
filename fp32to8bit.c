#include <limits.h>
#include "common.h"
#include "fp32to8bit.h"


float g_schar_max = ((float)SCHAR_MAX);
float g_sint16_max = ((float)SHRT_MAX);

int8_t convertfloatto8bit(float val, float maxfloatval)
{
    int8_t val_int = (int8_t)((val/maxfloatval)*g_schar_max);
    //printf ("%d ", val_int );
    return val_int;
}

int8_t *convert1dfloatarrayto8bit(float *array, int size, float maxfloatval, int8_t *outputarray)
{
    if (outputarray==NULL)
        outputarray = (int8_t *) malloc(size);
    for (int i=0;i<size;i++)
    {
        outputarray[i] = convertfloatto8bit(array[i], maxfloatval);
    }
    return outputarray;
}


int8_t *convert2dlinearfloatarrayto8bit(float *array, int sizex, int sizey, float maxfloatval, int8_t *outputarray)
{
    if (outputarray==NULL)
        outputarray = (int8_t *) malloc(sizex*sizey);
    for (int i=0;i<sizex*sizey;i++)
    {
        outputarray[i] = convertfloatto8bit(array[i], maxfloatval);
    }
    return outputarray;
}



float convert8bittofloat(int8_t val, float maxfloatval)
{
    float val_float = (float)((((float)val)/g_schar_max)*maxfloatval);
    return val_float;
}

float *convert1d8bitarraytofloat(int8_t *array, int size, float maxfloatval, float *outputarray)
{
    if (outputarray==NULL)
        outputarray = (float *) malloc(sizeof(float)*size);
    for (int i=0;i<size;i++)
    {
        outputarray[i] = convert8bittofloat(array[i], maxfloatval);
    }
    return outputarray;
}


float *convert2dlinear8bitarraytofloat(int8_t *array, int sizex, int sizey, float maxfloatval, float *outputarray)
{
    if (outputarray==NULL)
        outputarray = (float *) malloc(sizeof(float)*sizex*sizey);
    for (int i=0;i<sizex*sizey;i++)
    {
        outputarray[i] = convert8bittofloat(array[i], maxfloatval);
    }
    return outputarray;
}








int16_t convertfloatto16bit(float val, float maxfloatval)
{
    int16_t val_int = (int16_t)((val/maxfloatval)*g_sint16_max);
    //printf ("%d ", val_int );
    return val_int;
}

int16_t *convert1dfloatarrayto16bit(float *array, int size, float maxfloatval, int16_t *outputarray)
{
    if (outputarray==NULL)
        outputarray = (int16_t *) malloc(sizeof(int16_t)*size);
    for (int i=0;i<size;i++)
    {
        outputarray[i] = convertfloatto16bit(array[i], maxfloatval);
    }
    return outputarray;
}


int16_t *convert2dlinearfloatarrayto16bit(float *array, int sizex, int sizey, float maxfloatval, int16_t *outputarray)
{
    if (outputarray==NULL)
        outputarray = (int16_t *) malloc(sizeof(int16_t)*sizex*sizey);
    for (int i=0;i<sizex*sizey;i++)
    {
        outputarray[i] = convertfloatto16bit(array[i], maxfloatval);
    }
    return outputarray;
}



float convert16bittofloat(int16_t val, float maxfloatval)
{
    float val_float = (float)((((float)val)/g_sint16_max)*maxfloatval);
    return val_float;
}


float *convert1d16bitarraytofloat(int16_t *array, int size, float maxfloatval, float *outputarray)
{
    if (outputarray==NULL)
        outputarray = (float *) malloc(sizeof(float)*size);
    for (int i=0;i<size;i++)
    {
        outputarray[i] = convert16bittofloat(array[i], maxfloatval);
    }
    return outputarray;
}


float *convert2dlinear16bitarraytofloat(int8_t *array, int sizex, int sizey, float maxfloatval, float *outputarray)
{
    if (outputarray==NULL)
        outputarray = (float *) malloc(sizeof(float)*sizex*sizey);
    for (int i=0;i<sizex*sizey;i++)
    {
        outputarray[i] = convert16bittofloat(array[i], maxfloatval);
    }
    return outputarray;
}