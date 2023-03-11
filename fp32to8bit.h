int8_t convertfloatto8bit(float val, float maxfloatval);
int8_t *convert1dfloatarrayto8bit(float *array, int size, float maxfloatval, int8_t *outputarray);
int8_t *convert2dlinearfloatarrayto8bit(float *array, int sizex, int sizey, float maxfloatval, int8_t *outputarray);
float convert8bittofloat(int8_t val, float maxfloatval);
float *convert1d8bitarraytofloat(int8_t *array, int size, float maxfloatval, float *outputarray);
float *convert2dlinear8bitarraytofloat(int8_t *array, int sizex, int sizey, float maxfloatval, float *outputarray);



int16_t convertfloatto16bit(float val, float maxfloatval);
int16_t *convert1dfloatarrayto16bit(float *array, int size, float maxfloatval, int16_t *outputarray);
int16_t *convert2dlinearfloatarrayto16bit(float *array, int sizex, int sizey, float maxfloatval, int16_t *outputarray);
float convert16bittofloat(int16_t val, float maxfloatval);
float *convert1d16bitarraytofloat(int16_t *array, int size, float maxfloatval, float *outputarray);
float *convert2dlinear16bitarraytofloat(int8_t *array, int sizex, int sizey, float maxfloatval, float *outputarray);
