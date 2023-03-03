#include <ctype.h>

typedef struct layerfiles_t
{
    char **files;
    int *index;
    int *tmpindex;
    int numfiles;
    int numlayers;
    uint8_t *pkl_file;
    int pkl_file_size;
    char *zipfile;
} layerfiles_t;