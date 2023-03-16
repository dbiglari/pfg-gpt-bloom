#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include "json-c/json.h"
#include "zip.h"
#include "common.h"
#include "raw_loader.h"
#include "unpickler.h"
#include "bf16.h"


model_path_t model_definitions[MAXNUMMODELS] = {
};



int load_model_paths(char *configfile)
{
  // open the config file
  FILE *infile = NULL;
  
  if ((infile = fopen(configfile, "r")) == NULL)
  {
    // unable to open config file
    return -1;
  }

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  int count = 0;
  while ((read = getline(&line, &len, infile)) != -1) {
      printf("Retrieved line of length %zu:\n", read);
      printf("%s", line);
      // parse string to the first space, that becomes the model name, remainder of string becomes the modelpath
      int spacechar=-1;
      int numchars = strlen(line);
      for (int i=0;i<numchars;i++)
      {
        if (line[i] == ' ')
        {
          spacechar = i;
          line[i] = '\0';
        }
        if (line[i] == '\n')
        {
          line[i] = '\0';
        }        
      }
      if (spacechar == -1)
        return -2;

      strcpy(model_definitions[count].modelname, line);
      strcpy(model_definitions[count].modelpath, &(line[spacechar+1]));
      count++;

  }  

  return count;
}

/**
 * @brief  Lookup model path
 * @note   
 * @param  *modelname: 
 * @retval 
 */
char *lookup_model_path(char *modelname)
{
  for (int i = 0; i < NUMMODELS; i++)
  {
    if (strcmp(model_definitions[i].modelname, modelname) == 0)
    {
      return model_definitions[i].modelpath;
    }
  }
  return NULL;
}

/**
 * @brief  Load contents of file as char array
 * @note   
 * @param  *path: 
 * @retval 
 */
char *read_file(char *path)
{
  FILE *file = fopen(path, "r");
  if (file == NULL)
  {
    fprintf(stderr, "Expected file \"%s\" not found", path);
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  long len = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *buffer = malloc(len + 1);

  if (buffer == NULL)
  {
    fprintf(stderr, "Unable to allocate memory for file");
    fclose(file);
    return NULL;
  }

  size_t bytesread = fread(buffer, 1, len, file);
  if (bytesread != len)
  {
    fprintf(stderr, "bytesread != null in %s\n", (char *)path);
  }
  buffer[len] = '\0';

  return (char *)buffer;
}

/**
 * @brief  Load model tokens
 * @note   
 * @param  modelindex: 
 * @param  *path: 
 * @retval None
 */
void load_tokens(int modelindex, char *path)
{
  FILE *tokenbinaryfile = NULL;
  struct json_object *parsed_json;
  struct json_object *model;
  struct json_object *vocab;

  char *string = read_file(path);

  parsed_json = json_tokener_parse(string);

  // grab the list of tokens
  json_object_object_get_ex(parsed_json, "model", &model);
  json_object_object_get_ex(model, "vocab", &vocab);

  char nullterminator = '\0';

  int tok = 0;
  models[modelindex].nummodeltokens = json_object_object_length(vocab);
  models[modelindex].numwtetokens = models[modelindex].nummodeltokens;
  models[modelindex].numtokens = models[modelindex].numwtetokens;
  int tokenstrings_size = (models[modelindex].numtokens = models[modelindex].numwtetokens + 1 + MAXUSERTOKENS) * sizeof(char *);
  models[modelindex].tokenstrings = malloc(tokenstrings_size);
  memset(models[modelindex].tokenstrings, 0, tokenstrings_size);
  json_object_object_foreach(vocab, key, val)
  {
    models[modelindex].tokenstrings[tok] = strdup(key);
    tok++;
  }
  for (int i=tok;i<tokenstrings_size;i++)
  {
    models[modelindex].tokenstrings[tok] = strdup("");
  }
  models[modelindex].emptytoken = tokenize("</s>", modelindex);

  free(string);
}

/**
 * @brief  extract file from shard zip
 * @note   
 * @param  *path: 
 * @param  *shard: 
 * @param  load_to_ram: 
 * @retval None
 */
void extract_zip_shard(char *path, shard_t *shard, bool load_to_ram)
{
  void *buf = NULL;
  size_t bufsize;

  struct zip_t *zip = zip_open(path, 0, 'r');
  int i, n = zip_entries_total(zip);

  shard->bufs = malloc(sizeof(uint8_t *) * n);
  // shard->indices = malloc(sizeof(uint32_t) * n);

  strcpy(shard->filename, path);

  int fileindex = 0;
  for (i = 0; i < n; ++i)
  {
    zip_entry_openbyindex(zip, i);
    {
      const char *name = zip_entry_name(zip);

      zip_entry_open(zip, name);

      if (strstr(name, "archive/data/"))
      {
        int fileindex = atoi(&(name[13]));

        if (load_to_ram)
        {
          zip_entry_read(zip, shard->bufs[i], &bufsize);
        }

        // shard->indices = fileindex;
      }

      zip_entry_close(zip);

      zip_close(zip);

      int isdir = zip_entry_isdir(zip);
      unsigned long long size = zip_entry_size(zip);
      unsigned int crc32 = zip_entry_crc32(zip);
    }
    zip_entry_close(zip);
  }
}

/**
 * @brief  Extract specified file inside zip file into memory
 * @note   
 * @param  *path: 
 * @param  *file: 
 * @param  **buf: 
 * @retval 
 */
long long extract_zip_file_to_ram(char *path, char *file, uint8_t **buf)
{
  size_t bufsize;
  uint8_t *buf_local;

  struct zip_t *zip = zip_open(path, 0, 'r');

  int i, n = zip_entries_total(zip);
  for (i = 0; i < n; ++i)
  {
    zip_entry_openbyindex(zip, i);

    const char *name = zip_entry_name(zip);
    // printf ("%s\n", name);
    // fflush(stdout);
    if (strcmp(name, file) == 0)
    {
      bufsize = zip_entry_size(zip);
      buf_local = (uint8_t *)calloc(sizeof(uint8_t), bufsize);
      if (buf_local == NULL)
      {
        int q = 0;
        q++;
      }

      zip_entry_noallocread(zip, (void *)buf_local, bufsize);

      // do something with the data!
      // if (unload_after)
      //  free(buf);
      *buf = buf_local;
      zip_entry_close(zip);
      zip_close(zip);
      return bufsize;
    }

    int isdir = zip_entry_isdir(zip);
    unsigned long long size = zip_entry_size(zip);
    unsigned int crc32 = zip_entry_crc32(zip);

    zip_entry_close(zip);
  }

  zip_close(zip);
}

int Get_Layer_Index(int modelindex, char *filename, layerfiles_t *layerfiles)
{

  for (int i = 0; i < layerfiles->numfiles; i++)
  {
    if (layerfiles->files[i] != NULL)
    {
      if (strstr(layerfiles->files[i], filename) != NULL)
      {
        return layerfiles->index[i];
      }
    }
  }

  return -1;
}

void get_zipfile_for_weightfile(int modelindex, char *layer_fn, char *zipfile, bool useshards)
{

  if (useshards == false)
  {
    strcpy(zipfile, models[modelindex].path_to_zip);
    return;
  }

  struct json_object *weight_map = (struct json_object *)models[modelindex].weight_map;

  json_object_object_foreach(weight_map, key, val)
  {
    const char *str = json_object_get_string(val);

    char temp[1024];

    // get layer
    // printf ("get_zipfile_for_weightfile: %s : %s\n", key, val);
    if (useshards)
    {
      if (strcmp(layer_fn, key) == 0)
      {
        sprintf(zipfile, "%s/%s", models[modelindex].path, str);
        return;
      }
    }
  }
}

void get_zipfile_for_layer(int modelindex, int layernum, char *zipfile, bool useshards)
{
  if (useshards == false)
  {
    strcpy(zipfile, models[modelindex].path_to_zip);
    return;
  }

  struct json_object *weight_map = (struct json_object *)models[modelindex].weight_map;

  json_object_object_foreach(weight_map, key, val)
  {
    const char *str = json_object_get_string(val);

    char temp[1024];

    // get layer
    // printf ("get_zipfile_for_layer: %s : %s\n", key, val);
    if (isdigit(key[2]))
    {
      // supports up to 99 layers, if more layers need to change this code
      // currently 175 bloom model is 69 layers.
      temp[0] = key[2];
      temp[1] = key[3];
      temp[2] = 0;
      if (temp[1] == '.')
      {
        temp[1] = 0;
      }
      int layer = atoi(temp);
      if (layer == layernum)
      {
        sprintf(zipfile, "%s/%s", models[modelindex].path, str);
        return;
      }
    }
  }
}

/**
 * @brief  Compute checksum
 * @note   
 * @param  *addr: 
 * @param  count: 
 * @retval 
 */
uint16_t checksum(uint8_t *addr, uint32_t count)
{
  register uint32_t sum = 0;

  // Main summing loop
  while (count > 1)
  {
    sum = sum + *((uint16_t *)addr);
    addr += sizeof(uint16_t);
    count = count - 2;
  }

  // Add left-over uint8_t, if any
  if (count > 0)
    sum = sum + *((uint8_t *)addr);

  // Fold 32-bit sum to 16 bits
  while (sum >> 16)
    sum = (sum & 0xFFFF) + (sum >> 16);

  return (~sum);
}

void strippath(char *file_with_path, char *file_without_path)
{
  char *p = strrchr(file_with_path, '/');
  if (!p)
  {
    p = strrchr(file_with_path, '\\');
  }
  if (p)
  {
    strcpy(file_without_path, p + 1);
  }
  else
  {
    strcpy(file_without_path, file_with_path);
  }
}

void extract_shard_data(int modelindex, layerfiles_t *shard_layer_files, char *zipfile)
{
  char zipfileonly[1024];
  int fileindex = 0;

  if (models[modelindex].useshards)
  {

    for (int i = 0; i < shard_layer_files->numfiles; i++)
    {
      shard_layer_files->index[i] = -1;
      shard_layer_files->files[i] = NULL;
    }

    struct json_object *weight_map = (struct json_object *)models[modelindex].weight_map;

    strippath(zipfile, zipfileonly);

    char temp[1024];

    json_object_object_foreach(weight_map, key, val)
    {
      const char *str = json_object_get_string(val);
      if (strcmp(zipfileonly, str) == 0)
      {
        // parse shard number string to get shard number
        shard_layer_files->index[fileindex] = fileindex;
        shard_layer_files->files[fileindex++] = strdup(key);
      }
    }

    Unpickler_load_data_pkl(shard_layer_files, shard_layer_files->pkl_file, shard_layer_files->pkl_file_size);
  }
  else
  {
    for (int i = 0; i < shard_layer_files->numfiles; i++)
    {
      shard_layer_files->index[i] = i;
    }
  }
}

int get_shard_index_for_zipfile(char *zipfile)
{
  int index = 0;

  // loop through and find the maximum shard numbers
  char temp[1024];
  char *ret = strstr(zipfile, "pytorch_model");
  zipfile = ret;
  while (ret != NULL)
  {
    ret = strstr(zipfile + 1, "pytorch_model");
    if (ret != NULL)
      zipfile = ret;
  }
  strcpy(temp, zipfile);
  temp[19] = 0;
  index = atoi(&temp[17]);

  return index - 1;
}

void load_layer_container_thr(int modelindex, int layernum, int thr)
{

  //  printf ("loading layer %d\n", layernum);
  //  fflush(stdout);
  if (layernum == 19)
  {
    int q = 0;
    q++;
  }
  float min, max;
  char layer_fn[1024];
  char zipfile[1024];
  bloom_precision FP16_size = 2.0;
  int WVSIZE = models[modelindex].WVSIZE;
  int CTXSIZE = models[modelindex].CTXSIZE;
  int closest_power_of_2 = models[modelindex].closest_power_of_2;
  int HEADSIZE = models[modelindex].HEADSIZE;
  int NUMHEADS = models[modelindex].NUMHEADS;
  int NUMLAYERS = models[modelindex].NUMLAYERS;
  char fn[1024];
  long long sz;
  long long dcnt;
  long long dsz;
  int size;
  int filenum;
  uint16_t chk;
  layerfiles_t *layerfiles;

  // if (models[modelindex].layers[layernum].fp16_ln1_g == NULL)

  {

    if (thr == 0 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.input_layernorm.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)(uint8_t **)&models[modelindex].layers[layernum].fp16_ln1_g);
      // printf ("raw_loader: ln1_g sz %d\n", sz);
      // fflush(stdout);

#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded ln1_g: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].ln1_g == NULL)
      {
        models[modelindex].layers[layernum].ln1_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln1_g;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].ln1_g, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln1_g;
            models[modelindex].layers[layernum].ln1_g[j] = half_to_float(*((unsigned short *)(ptr + j))).f;          
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_ln1_g == NULL)
          models[modelindex].layers[layernum].q8_ln1_g = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln1_g, dcnt, 1,NULL);


        // computeminmax(models[modelindex].layers[layernum].ln1_g, dcnt, &models[modelindex].layers[layernum].ln1_g_min, &models[modelindex].layers[layernum].ln1_g_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln1_g, dcnt, models[modelindex].layers[layernum].ln1_g_max,NULL);
        // models[modelindex].layers[layernum].ln1_g = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].ln1_g_max,models[modelindex].layers[layernum].ln1_g);        
      }


      if (models[modelindex].layers[layernum].fp16_ln1_g != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_ln1_g);
        models[modelindex].layers[layernum].fp16_ln1_g = NULL;
      }
#endif
    }

    if (thr == 1 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.input_layernorm.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      // models[modelindex].layers[layernum].s_ln1_b=readfile(fn,&sz,path);
      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_ln1_b);
// printf ("raw_loader: ln1_b sz %d\n", sz);
// fflush(stdout);
// printf ("loaded ln1_b: %s %s %d\n", zipfile, fn, sz);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].ln1_b == NULL)
      {
        models[modelindex].layers[layernum].ln1_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln1_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].ln1_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln1_b;
            models[modelindex].layers[layernum].ln1_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_ln1_b==NULL)
          models[modelindex].layers[layernum].q8_ln1_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln1_b, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].ln1_b, dcnt, &models[modelindex].layers[layernum].ln1_b_min, &models[modelindex].layers[layernum].ln1_b_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln1_b, dcnt, models[modelindex].layers[layernum].ln1_b_max,NULL);
        // models[modelindex].layers[layernum].ln1_b = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].ln1_b_max,models[modelindex].layers[layernum].ln1_b);        

      }      

      if (models[modelindex].layers[layernum].fp16_ln1_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_ln1_b);
        models[modelindex].layers[layernum].fp16_ln1_b = NULL;
      }
#endif
    }

    if (thr == 2 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.post_attention_layernorm.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      // models[modelindex].layers[layernum].s_ln2_g=readfile(fn,&sz,path);
      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_ln2_g);
// printf ("raw_loader: ln2_g sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded ln2_g: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].ln2_g == NULL)
      {
        models[modelindex].layers[layernum].ln2_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln2_g;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].ln2_g, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln2_g;
            models[modelindex].layers[layernum].ln2_g[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_ln2_g==NULL)
          models[modelindex].layers[layernum].q8_ln2_g = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_g, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].ln2_g, dcnt, &models[modelindex].layers[layernum].ln2_g_min, &models[modelindex].layers[layernum].ln2_g_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_g, dcnt, models[modelindex].layers[layernum].ln2_g_max,NULL);
        // models[modelindex].layers[layernum].ln2_g = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].ln2_g_max,models[modelindex].layers[layernum].ln2_g);        
           
      }    

      if (models[modelindex].layers[layernum].fp16_ln2_g != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_ln2_g);
        models[modelindex].layers[layernum].fp16_ln2_g = NULL;
      }
#endif
    }

    if (thr == 3 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.post_attention_layernorm.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_ln2_b);
// printf ("raw_loader: ln2_b sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded ln2_b: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].ln2_b == NULL)
      {
        models[modelindex].layers[layernum].ln2_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln2_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].ln2_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_ln2_b;
            models[modelindex].layers[layernum].ln2_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_ln2_b==NULL)
          models[modelindex].layers[layernum].q8_ln2_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_b, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].ln2_b, dcnt, &models[modelindex].layers[layernum].ln2_b_min, &models[modelindex].layers[layernum].ln2_b_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].ln2_b, dcnt, models[modelindex].layers[layernum].ln2_b_max,NULL);
        // models[modelindex].layers[layernum].ln2_b = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].ln2_b_max,models[modelindex].layers[layernum].ln2_b);        
      }          

      if (models[modelindex].layers[layernum].fp16_ln2_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_ln2_b);
        models[modelindex].layers[layernum].fp16_ln2_b = NULL;
      }
#endif
    }

    if (thr == 4 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.mlp.dense_h_to_4h.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_mlp_cfc_w);
// printf ("raw_loader: mlp_cfc_w sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded mlp_cfc_w: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].mlp_cfc_w == NULL)
      {
        models[modelindex].layers[layernum].mlp_cfc_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cfc_w;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].mlp_cfc_w, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cfc_w;
            models[modelindex].layers[layernum].mlp_cfc_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_mlp_cfc_w==NULL)
          models[modelindex].layers[layernum].q8_mlp_cfc_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_w, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].mlp_cfc_w, dcnt, &models[modelindex].layers[layernum].mlp_cfc_w_min, &models[modelindex].layers[layernum].mlp_cfc_w_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_w, dcnt, models[modelindex].layers[layernum].mlp_cfc_w_max,NULL);
        // models[modelindex].layers[layernum].mlp_cfc_w = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].mlp_cfc_w_max,models[modelindex].layers[layernum].mlp_cfc_w);        

      }          

      if (models[modelindex].layers[layernum].fp16_mlp_cfc_w != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cfc_w);
        models[modelindex].layers[layernum].fp16_mlp_cfc_w = NULL;
      }
#endif
    }

    if (thr == 5 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.mlp.dense_h_to_4h.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_mlp_cfc_b);
// printf ("raw_loader: mlp_cfc_b sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded mlp_cfc_b: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].mlp_cfc_b == NULL)
      {
        models[modelindex].layers[layernum].mlp_cfc_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cfc_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].mlp_cfc_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cfc_b;
            models[modelindex].layers[layernum].mlp_cfc_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_mlp_cfc_b==NULL)
          models[modelindex].layers[layernum].q8_mlp_cfc_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, &models[modelindex].layers[layernum].mlp_cfc_b_min, &models[modelindex].layers[layernum].mlp_cfc_b_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cfc_b, dcnt, models[modelindex].layers[layernum].mlp_cfc_b_max,NULL);
        // models[modelindex].layers[layernum].mlp_cfc_b = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].mlp_cfc_b_max,models[modelindex].layers[layernum].mlp_cfc_b);        

      }                

      if (models[modelindex].layers[layernum].fp16_mlp_cfc_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cfc_b);
        models[modelindex].layers[layernum].fp16_mlp_cfc_b = NULL;
      }
#endif
    }

    if (thr == 6 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.mlp.dense_4h_to_h.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_mlp_cproj_w);
// printf ("raw_loader: mlp_cproj_w sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded mlp_cproj_w: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].mlp_cproj_w == NULL)
      {
        models[modelindex].layers[layernum].mlp_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cproj_w;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].mlp_cproj_w, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cproj_w;
            models[modelindex].layers[layernum].mlp_cproj_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_mlp_cproj_w==NULL)
          models[modelindex].layers[layernum].q8_mlp_cproj_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cproj_w, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].mlp_cproj_w, dcnt, &models[modelindex].layers[layernum].mlp_cproj_w_min, &models[modelindex].layers[layernum].mlp_cproj_w_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cproj_w, dcnt, models[modelindex].layers[layernum].mlp_cproj_w_max,NULL);
        // models[modelindex].layers[layernum].mlp_cproj_w = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].mlp_cproj_w_max,models[modelindex].layers[layernum].mlp_cproj_w);        

      }      

      if (models[modelindex].layers[layernum].fp16_mlp_cproj_w != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cproj_w);
        models[modelindex].layers[layernum].fp16_mlp_cproj_w = NULL;
      }
#endif
    }

    if (thr == 7 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.mlp.dense_4h_to_h.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_mlp_cproj_b);
// printf ("raw_loader: mlp_cproj_b sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded mlp_cproj_b: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].mlp_cproj_b == NULL)
      {
        models[modelindex].layers[layernum].mlp_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cproj_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].mlp_cproj_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_mlp_cproj_b;
            models[modelindex].layers[layernum].mlp_cproj_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_mlp_cproj_b==NULL)
          models[modelindex].layers[layernum].q8_mlp_cproj_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cproj_b, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].mlp_cproj_b, dcnt, &models[modelindex].layers[layernum].mlp_cproj_b_min, &models[modelindex].layers[layernum].mlp_cproj_b_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].mlp_cproj_b, dcnt, models[modelindex].layers[layernum].mlp_cproj_b_max,NULL);
        // models[modelindex].layers[layernum].mlp_cproj_b = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].mlp_cproj_b_max,models[modelindex].layers[layernum].mlp_cproj_b);        

      }            

      if (models[modelindex].layers[layernum].fp16_mlp_cproj_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_mlp_cproj_b);
        models[modelindex].layers[layernum].fp16_mlp_cproj_b = NULL;
      }
#endif
    }

    if (thr == 8 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.self_attention.dense.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_attn_cproj_w);
// printf ("raw_loader: attn_cproj_w sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded attn_cproj_w: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].attn_cproj_w == NULL)
      {
        models[modelindex].layers[layernum].attn_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cproj_w;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].attn_cproj_w, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cproj_w;
            models[modelindex].layers[layernum].attn_cproj_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_attn_cproj_w==NULL)
          models[modelindex].layers[layernum].q8_attn_cproj_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_w, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].attn_cproj_w, dcnt, &models[modelindex].layers[layernum].attn_cproj_w_min, &models[modelindex].layers[layernum].attn_cproj_w_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_w, dcnt, models[modelindex].layers[layernum].attn_cproj_w_max,NULL);
        // models[modelindex].layers[layernum].attn_cproj_w = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].attn_cproj_w_max,models[modelindex].layers[layernum].attn_cproj_w);        

      }                

      if (models[modelindex].layers[layernum].fp16_attn_cproj_w != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_attn_cproj_w);
        models[modelindex].layers[layernum].fp16_attn_cproj_w = NULL;
      }
#endif
    }

    if (thr == 9 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.self_attention.dense.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_attn_cproj_b);
// printf ("raw_loader: attn_cproj_b sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded attn_cproj_b: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].attn_cproj_b == NULL)
      {
        models[modelindex].layers[layernum].attn_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cproj_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].attn_cproj_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cproj_b;
            models[modelindex].layers[layernum].attn_cproj_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_attn_cproj_b==NULL)
          models[modelindex].layers[layernum].q8_attn_cproj_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_b, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].attn_cproj_b, dcnt, &models[modelindex].layers[layernum].attn_cproj_b_min, &models[modelindex].layers[layernum].attn_cproj_b_max);
        // int8_t *q8_temp = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cproj_b, dcnt, models[modelindex].layers[layernum].attn_cproj_b_max,NULL);
        // models[modelindex].layers[layernum].attn_cproj_b = convert1d8bitarraytofloat(q8_temp, dcnt, models[modelindex].layers[layernum].attn_cproj_b_max,models[modelindex].layers[layernum].attn_cproj_b);        

      }                      

      if (models[modelindex].layers[layernum].fp16_attn_cproj_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_attn_cproj_b);
        models[modelindex].layers[layernum].fp16_attn_cproj_b = NULL;
      }
#endif
    }

    if (thr == 10 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.self_attention.query_key_value.weight", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_attn_cattn_w);
// printf ("raw_loader: attn_cattn_w sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded attn_cattn_w: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].attn_cattn_w == NULL)
      {
        models[modelindex].layers[layernum].attn_cattn_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cattn_w;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].attn_cattn_w, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cattn_w;
            models[modelindex].layers[layernum].attn_cattn_w[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_attn_cattn_w==NULL)
          models[modelindex].layers[layernum].q8_attn_cattn_w = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cattn_w, dcnt, 1,NULL);

        // computeminmax(models[modelindex].layers[layernum].attn_cattn_w, dcnt, &models[modelindex].layers[layernum].attn_cattn_w_min, &models[modelindex].layers[layernum].attn_cattn_w_max);
        // int16_t *q16_temp = convert1dfloatarrayto16bit(models[modelindex].layers[layernum].attn_cattn_w, dcnt, models[modelindex].layers[layernum].attn_cattn_w_max,NULL);
        // models[modelindex].layers[layernum].attn_cattn_w = convert1d16bitarraytofloat(q16_temp, dcnt, models[modelindex].layers[layernum].attn_cattn_w_max,models[modelindex].layers[layernum].attn_cattn_w);        

      }               

      if (models[modelindex].layers[layernum].fp16_attn_cattn_w != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_attn_cattn_w);
        models[modelindex].layers[layernum].fp16_attn_cattn_w = NULL;
      }
#endif
    }

    if (thr == 11 || (global_numthreads < 12 && thr == 0) || thr == -1)
    {
      sprintf(layer_fn, "h.%d.self_attention.query_key_value.bias", layernum);
      get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
      if (models[modelindex].useshards)
      {
        layerfiles = models[modelindex].shard_layerfiles[get_shard_index_for_zipfile(zipfile)];
      }
      else
      {
        layerfiles = &models[modelindex].layerfiles;
      }
      filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
      sprintf(fn, "archive/data/%d", filenum);

      sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].layers[layernum].fp16_attn_cattn_b);
// printf ("raw_loader: attn_cattn_b sz %d\n", sz);
// fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
      // printf ("loaded attn_cattn_b: %s %s %d\n", zipfile, fn, sz);
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      if (models[modelindex].layers[layernum].attn_cattn_b == NULL)
      {
        models[modelindex].layers[layernum].attn_cattn_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dcnt);
        // copy values
        if (models[modelindex].use_bfloat16)
        {
          uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cattn_b;
          BFloat16ToFloat((uint16_t *)ptr, models[modelindex].layers[layernum].attn_cattn_b, dcnt);
        }
        else
        {
          for (long long j = 0; j < dcnt; j++)
          {
            uint16_t *ptr = (uint16_t *)models[modelindex].layers[layernum].fp16_attn_cattn_b;
            models[modelindex].layers[layernum].attn_cattn_b[j] = half_to_float(*((unsigned short *)(ptr + j))).f;
          }
        }
      }

      if (models[modelindex].use_8bit==true)
      {
        if (models[modelindex].layers[layernum].q8_attn_cattn_b==NULL)
          models[modelindex].layers[layernum].q8_attn_cattn_b = convert1dfloatarrayto8bit(models[modelindex].layers[layernum].attn_cattn_b, dcnt, 1,NULL);
      }       

      if (models[modelindex].layers[layernum].fp16_attn_cattn_b != NULL)
      {
        free(models[modelindex].layers[layernum].fp16_attn_cattn_b);
        models[modelindex].layers[layernum].fp16_attn_cattn_b = NULL;
      }
#endif
    }

    // fprintf (stderr, "file load complete\n");
    // fflush(stderr);
  }
}

void load_layer_container(int modelindex, int layernum)
{
  load_layer_container_thr(modelindex, layernum, -1);
}

void unload_layer_container(int modelindex, int layernum)
{
  // free fp16 float data
  if (models[modelindex].layers[layernum].fp16_ln1_g != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_ln1_g);
    models[modelindex].layers[layernum].fp16_ln1_g = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_ln1_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_ln1_b);
    models[modelindex].layers[layernum].fp16_ln1_b = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_ln2_g != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_ln2_g);
    models[modelindex].layers[layernum].fp16_ln2_g = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_ln2_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_ln2_b);
    models[modelindex].layers[layernum].fp16_ln2_b = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_mlp_cfc_w != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_mlp_cfc_w);
    models[modelindex].layers[layernum].fp16_mlp_cfc_w = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_mlp_cfc_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_mlp_cfc_b);
    models[modelindex].layers[layernum].fp16_mlp_cfc_b = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_mlp_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_mlp_cproj_w);
    models[modelindex].layers[layernum].fp16_mlp_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_mlp_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_mlp_cproj_b);
    models[modelindex].layers[layernum].fp16_mlp_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_attn_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_attn_cproj_w);
    models[modelindex].layers[layernum].fp16_attn_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_attn_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_attn_cproj_b);
    models[modelindex].layers[layernum].fp16_attn_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_attn_cattn_w != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_attn_cattn_w);
    models[modelindex].layers[layernum].fp16_attn_cattn_w = NULL;
  }
  if (models[modelindex].layers[layernum].fp16_attn_cattn_b != NULL)
  {
    free(models[modelindex].layers[layernum].fp16_attn_cattn_b);
    models[modelindex].layers[layernum].fp16_attn_cattn_b = NULL;
  }

  // free 32bit float data
  if (models[modelindex].layers[layernum].ln1_g != NULL)
  {
    free(models[modelindex].layers[layernum].ln1_g);
    models[modelindex].layers[layernum].ln1_g = NULL;
  }
  if (models[modelindex].layers[layernum].ln1_b != NULL)
  {
    free(models[modelindex].layers[layernum].ln1_b);
    models[modelindex].layers[layernum].ln1_b = NULL;
  }
  if (models[modelindex].layers[layernum].ln2_g != NULL)
  {
    free(models[modelindex].layers[layernum].ln2_g);
    models[modelindex].layers[layernum].ln2_g = NULL;
  }
  if (models[modelindex].layers[layernum].ln2_b != NULL)
  {
    free(models[modelindex].layers[layernum].ln2_b);
    models[modelindex].layers[layernum].ln2_b = NULL;
  }
  if (models[modelindex].layers[layernum].mlp_cfc_w != NULL)
  {
    free(models[modelindex].layers[layernum].mlp_cfc_w);
    models[modelindex].layers[layernum].mlp_cfc_w = NULL;
  }
  if (models[modelindex].layers[layernum].mlp_cfc_b != NULL)
  {
    free(models[modelindex].layers[layernum].mlp_cfc_b);
    models[modelindex].layers[layernum].mlp_cfc_b = NULL;
  }
  if (models[modelindex].layers[layernum].mlp_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].mlp_cproj_w);
    models[modelindex].layers[layernum].mlp_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].mlp_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].mlp_cproj_b);
    models[modelindex].layers[layernum].mlp_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].attn_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].attn_cproj_w);
    models[modelindex].layers[layernum].attn_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].attn_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].attn_cproj_b);
    models[modelindex].layers[layernum].attn_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].attn_cattn_w != NULL)
  {
    free(models[modelindex].layers[layernum].attn_cattn_w);
    models[modelindex].layers[layernum].attn_cattn_w = NULL;
  }
  if (models[modelindex].layers[layernum].attn_cattn_b != NULL)
  {
    free(models[modelindex].layers[layernum].attn_cattn_b);
    models[modelindex].layers[layernum].attn_cattn_b = NULL;
  }


  // free 8bit data
  if (models[modelindex].layers[layernum].q8_ln1_g != NULL)
  {
    free(models[modelindex].layers[layernum].q8_ln1_g);
    models[modelindex].layers[layernum].q8_ln1_g = NULL;
  }
  if (models[modelindex].layers[layernum].q8_ln1_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_ln1_b);
    models[modelindex].layers[layernum].q8_ln1_b = NULL;
  }
  if (models[modelindex].layers[layernum].q8_ln2_g != NULL)
  {
    free(models[modelindex].layers[layernum].q8_ln2_g);
    models[modelindex].layers[layernum].q8_ln2_g = NULL;
  }
  if (models[modelindex].layers[layernum].q8_ln2_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_ln2_b);
    models[modelindex].layers[layernum].q8_ln2_b = NULL;
  }
  if (models[modelindex].layers[layernum].q8_mlp_cfc_w != NULL)
  {
    free(models[modelindex].layers[layernum].q8_mlp_cfc_w);
    models[modelindex].layers[layernum].q8_mlp_cfc_w = NULL;
  }
  if (models[modelindex].layers[layernum].q8_mlp_cfc_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_mlp_cfc_b);
    models[modelindex].layers[layernum].q8_mlp_cfc_b = NULL;
  }
  if (models[modelindex].layers[layernum].q8_mlp_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].q8_mlp_cproj_w);
    models[modelindex].layers[layernum].q8_mlp_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].q8_mlp_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_mlp_cproj_b);
    models[modelindex].layers[layernum].q8_mlp_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].q8_attn_cproj_w != NULL)
  {
    free(models[modelindex].layers[layernum].q8_attn_cproj_w);
    models[modelindex].layers[layernum].q8_attn_cproj_w = NULL;
  }
  if (models[modelindex].layers[layernum].q8_attn_cproj_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_attn_cproj_b);
    models[modelindex].layers[layernum].q8_attn_cproj_b = NULL;
  }
  if (models[modelindex].layers[layernum].q8_attn_cattn_w != NULL)
  {
    free(models[modelindex].layers[layernum].q8_attn_cattn_w);
    models[modelindex].layers[layernum].q8_attn_cattn_w = NULL;
  }
  if (models[modelindex].layers[layernum].q8_attn_cattn_b != NULL)
  {
    free(models[modelindex].layers[layernum].q8_attn_cattn_b);
    models[modelindex].layers[layernum].q8_attn_cattn_b = NULL;
  }  
}

void load_word_embeddings(int modelindex)
{
  char layer_fn[1024];
  char zipfile[1024];
  bloom_precision FP16_size = 2.0;

  int WVSIZE = models[modelindex].WVSIZE;
  int CTXSIZE = models[modelindex].CTXSIZE;
  int closest_power_of_2 = models[modelindex].closest_power_of_2;
  int HEADSIZE = models[modelindex].HEADSIZE;
  int NUMHEADS = models[modelindex].NUMHEADS;
  int NUMLAYERS = models[modelindex].NUMLAYERS;
  char fn[1024];
  int filenum;
  long long dsz;
  long long sz;
  long long dcnt;
  layerfiles_t *layerfiles;

  sprintf(layer_fn, "word_embeddings.weight");
  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
  if (models[modelindex].useshards)
  {
    layerfiles = models[modelindex].shard_wtefiles;
  }
  else
  {
    layerfiles = &models[modelindex].layerfiles;
  }

  filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
  sprintf(fn, "archive/data/%d", filenum);

  sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].fp16_wte);
  //  printf ("raw loader: wte sz: %ld\n", sz);
  //  fflush(stdout);
#ifndef EXTRACT_WEIGHTS_ON_DEMAND
  // printf ("loaded wte: %s %s %d\n", zipfile, fn, sz);
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  models[modelindex].wte = (wte_t *)malloc(sizeof(bloom_precision) * (dcnt + MAXUSERTOKENS * WVSIZE));
  // copy values
  // memset(models[modelindex].wte, 0, sizeof(bloom_precision)*(dcnt+MAXUSERTOKENS*WVSIZE));
  memset(&(models[modelindex].wte[dcnt]), 0, sizeof(bloom_precision) * (MAXUSERTOKENS * WVSIZE));

  if (models[modelindex].use_bfloat16)
  {
    uint16_t *ptr = (uint16_t *)models[modelindex].fp16_wte;
    BFloat16ToFloat((uint16_t *)ptr, models[modelindex].wte, dcnt);
  }
  else
  {
    for (long long i = 0; i < dcnt; i++)
    {
      uint16_t *ptr = (uint16_t *)models[modelindex].fp16_wte;
      // FP16 temp;
      // memcpy(&temp, (ptr + i), 2);
      models[modelindex].wte[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
      // models[modelindex].wte[i] = half_to_float(temp).f;
    }
  }

  if (models[modelindex].use_8bit==true)
  {
    if (models[modelindex].q8_wte==NULL)
      models[modelindex].q8_wte = convert1dfloatarrayto8bit(models[modelindex].wte, dcnt, 1,NULL);
  }  

  if (models[modelindex].fp16_wte != NULL)
  {
    free(models[modelindex].fp16_wte);
    models[modelindex].fp16_wte = NULL;
  }
#endif

  sprintf(layer_fn, "word_embeddings_layernorm.weight");
  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
  filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
  sprintf(fn, "archive/data/%d", filenum);

  sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].fp16_welw);
  // printf ("loaded welw: %s %s %d\n", zipfile, fn, sz);
  int dwesz = sz * FP16_size;
  dcnt = sz / FP16_size;
  models[modelindex].welw = (pkdflt *)malloc(sizeof(bloom_precision) * dcnt);
  // copy values
  if (models[modelindex].use_bfloat16)
  {
    uint16_t *ptr = (uint16_t *)models[modelindex].fp16_welw;
    BFloat16ToFloat((uint16_t *)ptr, models[modelindex].welw, dcnt);
  }
  else
  {
    for (long long i = 0; i < dcnt; i++)
    {
      uint16_t *ptr = (uint16_t *)models[modelindex].fp16_welw;
      models[modelindex].welw[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
    }
  }

  if (models[modelindex].use_8bit==true)
  {
    if(models[modelindex].q8_welw==NULL)
      models[modelindex].q8_welw = convert1dfloatarrayto8bit(models[modelindex].welw, dcnt, 1,NULL);
  }    

  if (models[modelindex].fp16_welw != NULL)
  {
    free(models[modelindex].fp16_welw);
    models[modelindex].fp16_welw = NULL;
  }

  sprintf(layer_fn, "word_embeddings_layernorm.bias");
  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
  filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
  sprintf(fn, "archive/data/%d", filenum);

  sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].fp16_welb);
  // printf ("loaded welb: %s %s %d\n", zipfile, fn, sz);
  dwesz = sz * FP16_size;
  dcnt = sz / FP16_size;
  models[modelindex].welb = (pkdflt *)malloc(sizeof(bloom_precision) * dcnt);
  // copy values
  if (models[modelindex].use_bfloat16)
  {
    uint16_t *ptr = (uint16_t *)models[modelindex].fp16_welb;
    BFloat16ToFloat((uint16_t *)ptr, models[modelindex].welb, dcnt);
  }
  else
  {
    for (long long i = 0; i < dcnt; i++)
    {
      uint16_t *ptr = (uint16_t *)models[modelindex].fp16_welb;
      models[modelindex].welb[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
    }
  }

  if (models[modelindex].use_8bit==true)
  {
    if (models[modelindex].q8_welb == NULL)
      models[modelindex].q8_welb = convert1dfloatarrayto8bit(models[modelindex].welb, dcnt, 1,NULL);
  }      

  if (models[modelindex].fp16_welb != NULL)
  {
    free(models[modelindex].fp16_welb);
    models[modelindex].fp16_welb = NULL;
  }
}

void load_final_layer_normalization(int modelindex)
{
  char layer_fn[1024];
  char zipfile[1024];
  bloom_precision FP16_size = 2.0;
  int WVSIZE = models[modelindex].WVSIZE;
  int CTXSIZE = models[modelindex].CTXSIZE;
  int closest_power_of_2 = models[modelindex].closest_power_of_2;
  int HEADSIZE = models[modelindex].HEADSIZE;
  int NUMHEADS = models[modelindex].NUMHEADS;
  int NUMLAYERS = models[modelindex].NUMLAYERS;
  char fn[1024];
  int filenum;
  long long dsz;
  long long sz;
  layerfiles_t *layerfiles;

  sprintf(layer_fn, "ln_f.weight");
  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);
  if (models[modelindex].useshards)
  {
    layerfiles = models[modelindex].shard_lnfiles;
  }
  else
  {
    layerfiles = &models[modelindex].layerfiles;
  }

  sprintf(layer_fn, "ln_f.weight");
  filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
  sprintf(fn, "archive/data/%d", filenum);

  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);

  sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].fp16_lnf_g);
  // printf ("loaded lnf_g: %s %s %d\n", zipfile, fn, sz);
  dsz = sz * FP16_size;
  long long dcnt = sz / FP16_size;
  models[modelindex].lnf_g = (wte_t *)malloc(sizeof(bloom_precision) * dcnt);
  // copy values
  if (models[modelindex].use_bfloat16)
  {
    uint16_t *ptr = (uint16_t *)models[modelindex].fp16_lnf_g;
    BFloat16ToFloat((uint16_t *)ptr, models[modelindex].lnf_g, dcnt);
  }
  else
  {
    for (long long i = 0; i < dcnt; i++)
    {
      uint16_t *ptr = (uint16_t *)models[modelindex].fp16_lnf_g;
      models[modelindex].lnf_g[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
    }
  }


  if (models[modelindex].use_8bit==true)
  {
    if (models[modelindex].q8_lnf_g == NULL)
      models[modelindex].q8_lnf_g = convert1dfloatarrayto8bit(models[modelindex].lnf_g, dcnt, 1,NULL);
  }      


  if (models[modelindex].fp16_lnf_g != NULL)
  {
    free(models[modelindex].fp16_lnf_g);
    models[modelindex].fp16_lnf_g = NULL;
  }

  sprintf(layer_fn, "ln_f.bias");
  filenum = Get_Layer_Index(modelindex, layer_fn, layerfiles);
  sprintf(fn, "archive/data/%d", filenum);

  get_zipfile_for_weightfile(modelindex, layer_fn, zipfile, models[modelindex].useshards);

  sz = extract_zip_file_to_ram(zipfile, fn, (uint8_t **)&models[modelindex].fp16_lnf_b);
  // printf ("loaded lnf_b: %s %s %d\n", zipfile, fn, sz);
  dsz = sz * FP16_size;
  dcnt = sz / FP16_size;
  models[modelindex].lnf_b = (wte_t *)malloc(sizeof(bloom_precision) * dcnt);
  // copy values
  if (models[modelindex].use_bfloat16)
  {
    uint16_t *ptr = (uint16_t *)models[modelindex].fp16_lnf_b;
    BFloat16ToFloat((uint16_t *)ptr, models[modelindex].lnf_b, dcnt);
  }
  else
  {
    for (long long i = 0; i < dcnt; i++)
    {
      uint16_t *ptr = (uint16_t *)models[modelindex].fp16_lnf_b;
      models[modelindex].lnf_b[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
    }
  }


  if (models[modelindex].use_8bit==true)
  {
    if (models[modelindex].q8_lnf_b == NULL)
      models[modelindex].q8_lnf_b = convert1dfloatarrayto8bit(models[modelindex].lnf_b, dcnt, 1,NULL);
  }      


  if (models[modelindex].fp16_lnf_b != NULL)
  {
    free(models[modelindex].fp16_lnf_b);
    models[modelindex].fp16_lnf_b = NULL;
  }
}

int get_shard_for_layer(int modelindex, int layer)
{
  struct json_object *weight_map = (struct json_object *)models[modelindex].weight_map;

  char temp[1024];
  sprintf(temp, "h.%d.", layer);

  json_object_object_foreach(weight_map, key, val)
  {
    const char *str = json_object_get_string(val);
    if (strstr(temp, key) == 0)
    {
      // parse shard number string to get shard number
      strcpy(temp, str);
      temp[19] = 0;
      int shard = atoi(&(temp[14]));
    }
  }

  // couldn't find shard
  return -1;
}

int load_shard_map(int modelindex, char *path_to_json)
{
  int num_shards;
  char *string = read_file(path_to_json);
  struct json_object *parsed_json = json_tokener_parse(string);
  struct json_object *weight_map;

  json_object_object_get_ex(parsed_json, "weight_map", &weight_map);
  int array_size = json_object_object_length(weight_map);

  models[modelindex].weight_map = (void *)weight_map;

  free(string);

  // loop through and find the maximum shard numbers
  int maxshard = 0;
  json_object_object_foreach(weight_map, key, val)
  {
    const char *str = json_object_get_string(val);

    char temp[1024];
    strcpy(temp, str);
    temp[19] = 0;
    int shardnum = atoi(&temp[17]);

    if (shardnum > maxshard)
    {
      maxshard = shardnum;
    }
  }

  return maxshard;
}

// path points to a directory with config.json
/**
 * @brief  Load model
 * @note   
 * @param  *path: 
 * @param  modelindex: 
 * @retval 
 */
int load_huggingface_bloom_model_folder(char *path, int modelindex)
{

  char path_to_json[2048];

  char path_to_tokenizerjson[2048];
  char shard_file_bin[2048];

  sprintf(path_to_json, "%s/config.json", path);
  // open config file so we can see what we're dealing with

  char *string = read_file(path_to_json);
  struct json_object *parsed_json = json_tokener_parse(string);

  struct json_object *vocab;
  struct json_object *temp_json_obj;

  free(string);

  strcpy(models[modelindex].path, path);
  // set all the model hyper parameters
  json_object_object_get_ex(parsed_json, "n_embed", &temp_json_obj);
  models[modelindex].WVSIZE = json_object_get_int(temp_json_obj);
  if (models[modelindex].WVSIZE == 0)
  {
    json_object_object_get_ex(parsed_json, "hidden_size", &temp_json_obj);
    models[modelindex].WVSIZE = json_object_get_int(temp_json_obj);
  }
  json_object_object_get_ex(parsed_json, "n_layer", &temp_json_obj);
  models[modelindex].NUMLAYERS = json_object_get_int(temp_json_obj);
  json_object_object_get_ex(parsed_json, "num_attention_heads", &temp_json_obj);
  models[modelindex].NUMHEADS = json_object_get_int(temp_json_obj);
  if (models[modelindex].NUMHEADS == 0)
  {
    json_object_object_get_ex(parsed_json, "n_head", &temp_json_obj);
    models[modelindex].NUMHEADS = json_object_get_int(temp_json_obj);
  }
  models[modelindex].HEADSIZE = (models[modelindex].WVSIZE / models[modelindex].NUMHEADS);
  models[modelindex].RSQRT_HEADSIZE = (1 / sqrt(models[modelindex].HEADSIZE));

  models[modelindex].CTXSIZE = 4096; // load from config or command line?

  models[modelindex].useshards = false;
  // if pytorch_model.bin.index.json exists, open it as a map for shards (7b1 and 175b models)
  sprintf(path_to_json, "%s/pytorch_model.bin.index.json", path);
  if (access(path_to_json, F_OK) == 0)
  {
    models[modelindex].useshards = true;

    int ind = 1;
    // load map of which shards contain which files
    models[modelindex].num_shards = load_shard_map(modelindex, path_to_json);

    // load all shard files if models[modelindex].shard_layerfiles == NULL
    if (models[modelindex].shard_layerfiles == NULL)
    {
      char shard_zipfile[4096];

      models[modelindex].shard_layerfiles = malloc(sizeof(layerfiles_t *) * models[modelindex].num_shards);
      for (int i = 0; i < models[modelindex].num_shards; i++)
      {
        sprintf(shard_zipfile, "%s/pytorch_model-000%02d-of-000%02d.bin", models[modelindex].path, i + 1, models[modelindex].num_shards);
        // check for the existance of the file
        if (access(shard_zipfile, F_OK) != 0)
        {
          sprintf(shard_zipfile, "%s/pytorch_model_000%02d-of-000%02d.bin", models[modelindex].path, i + 1, models[modelindex].num_shards);
        }

        models[modelindex].shard_layerfiles[i] = NULL;

        int fileindex = 0;
        models[modelindex].shard_layerfiles[i] = malloc(sizeof(layerfiles_t));
        models[modelindex].shard_layerfiles[i]->pkl_file_size = extract_zip_file_to_ram(shard_zipfile, "archive/data.pkl", &models[modelindex].shard_layerfiles[i]->pkl_file);
        int files_per_layer = FILES_PER_LAYER;
        models[modelindex].shard_layerfiles[i]->numlayers = models[modelindex].NUMLAYERS;
        models[modelindex].shard_layerfiles[i]->zipfile = strdup(shard_zipfile);
        models[modelindex].shard_layerfiles[i]->numfiles = models[modelindex].shard_layerfiles[i]->numlayers * files_per_layer + 5;
        models[modelindex].shard_layerfiles[i]->index = malloc(sizeof(int) * models[modelindex].shard_layerfiles[i]->numfiles);
        models[modelindex].shard_layerfiles[i]->tmpindex = malloc(sizeof(int) * models[modelindex].shard_layerfiles[i]->numfiles);
        models[modelindex].shard_layerfiles[i]->files = malloc(sizeof(char *) * models[modelindex].shard_layerfiles[i]->numfiles);
        extract_shard_data(modelindex, models[modelindex].shard_layerfiles[i], shard_zipfile);
      }
    }

    if (models[modelindex].shard_wtefiles == NULL)
    {
      char shard_zipfile[4096];
      char layer_fn[1024];
      sprintf(layer_fn, "word_embeddings.weight");
      get_zipfile_for_weightfile(modelindex, layer_fn, shard_zipfile, models[modelindex].useshards);
      // sprintf(shard_zipfile, "%s/%s", models[modelindex].path, shard_zipfile);
      models[modelindex].shard_wtefiles = malloc(sizeof(layerfiles_t));
      models[modelindex].shard_wtefiles->pkl_file_size = extract_zip_file_to_ram(shard_zipfile, "archive/data.pkl", &models[modelindex].shard_wtefiles->pkl_file);
      int files_per_layer = FILES_PER_LAYER;
      models[modelindex].shard_wtefiles->zipfile = strdup(shard_zipfile);
      models[modelindex].shard_wtefiles->numlayers = models[modelindex].NUMLAYERS;
      models[modelindex].shard_wtefiles->numfiles = models[modelindex].shard_wtefiles->numlayers * files_per_layer + 5;
      models[modelindex].shard_wtefiles->index = malloc(sizeof(int) * models[modelindex].shard_wtefiles->numfiles);
      models[modelindex].shard_wtefiles->tmpindex = malloc(sizeof(int) * models[modelindex].shard_wtefiles->numfiles);
      models[modelindex].shard_wtefiles->files = malloc(sizeof(char *) * models[modelindex].shard_wtefiles->numfiles);
      extract_shard_data(modelindex, models[modelindex].shard_wtefiles, shard_zipfile);
    }

    if (models[modelindex].shard_lnfiles == NULL)
    {
      char shard_zipfile[4096];
      char layer_fn[1024];
      sprintf(layer_fn, "ln_f.weight");
      get_zipfile_for_weightfile(modelindex, layer_fn, shard_zipfile, models[modelindex].useshards);
      // sprintf(shard_zipfile, "%s/%s", models[modelindex].path, shard_zipfile);
      models[modelindex].shard_lnfiles = malloc(sizeof(layerfiles_t));
      models[modelindex].shard_lnfiles->zipfile = strdup(shard_zipfile);
      models[modelindex].shard_lnfiles->pkl_file_size = extract_zip_file_to_ram(shard_zipfile, "archive/data.pkl", &models[modelindex].shard_lnfiles->pkl_file);
      int files_per_layer = FILES_PER_LAYER;
      models[modelindex].shard_lnfiles->numlayers = models[modelindex].NUMLAYERS;
      models[modelindex].shard_lnfiles->numfiles = models[modelindex].shard_lnfiles->numlayers * files_per_layer + 5;
      models[modelindex].shard_lnfiles->index = malloc(sizeof(int) * models[modelindex].shard_lnfiles->numfiles);
      models[modelindex].shard_lnfiles->tmpindex = malloc(sizeof(int) * models[modelindex].shard_lnfiles->numfiles);
      models[modelindex].shard_lnfiles->files = malloc(sizeof(char *) * models[modelindex].shard_lnfiles->numfiles);

      extract_shard_data(modelindex, models[modelindex].shard_lnfiles, shard_zipfile);
    }
  }
  else
  {
    // otherwise, just unzip pytorch_model.bin as a zip file
    sprintf(models[modelindex].path_to_zip, "%s/pytorch_model.bin", path);

    // extract data.pkl from zip
    uint8_t *pkl_file;
    int size = extract_zip_file_to_ram(models[modelindex].path_to_zip, "archive/data.pkl", (uint8_t **)&pkl_file);

    int files_per_layer = 12;

    models[modelindex].layerfiles.numlayers = models[modelindex].NUMLAYERS;
    models[modelindex].layerfiles.numfiles = models[modelindex].layerfiles.numlayers * files_per_layer + 5;
    models[modelindex].layerfiles.index = malloc(sizeof(int) * models[modelindex].layerfiles.numfiles);
    models[modelindex].layerfiles.tmpindex = malloc(sizeof(int) * models[modelindex].layerfiles.numfiles);
    models[modelindex].layerfiles.files = malloc(sizeof(char *) * models[modelindex].layerfiles.numfiles);

    int fileindex = 0;

    models[modelindex].layerfiles.files[fileindex++] = strdup("word_embeddings.weight");
    models[modelindex].layerfiles.files[fileindex++] = strdup("word_embeddings_layernorm.weight");
    models[modelindex].layerfiles.files[fileindex++] = strdup("word_embeddings_layernorm.bias");

    for (int i = 0; i < models[modelindex].NUMLAYERS; i++)
    {
      char bufferfilename[1024];
      sprintf(bufferfilename, "h.%d.input_layernorm.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.input_layernorm.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.self_attention.query_key_value.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.self_attention.query_key_value.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.self_attention.dense.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.self_attention.dense.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.post_attention_layernorm.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.post_attention_layernorm.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.mlp.dense_h_to_4h.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.mlp.dense_h_to_4h.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.mlp.dense_4h_to_h.weight", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
      sprintf(bufferfilename, "h.%d.mlp.dense_4h_to_h.bias", i);
      models[modelindex].layerfiles.files[fileindex++] = strdup(bufferfilename);
    }

    models[modelindex].layerfiles.files[fileindex++] = strdup("ln_f.weight");
    models[modelindex].layerfiles.files[fileindex++] = strdup("ln_f.bias");

    extract_shard_data(modelindex, &models[modelindex].layerfiles, models[modelindex].path_to_zip);

    // pickle file maps layer names to file numbers
    Unpickler_load_data_pkl(&models[modelindex].layerfiles, pkl_file, size);
  }

  // load tokens
  sprintf(path_to_tokenizerjson, "%s/tokenizer.json", path);
  load_tokens(modelindex, path_to_tokenizerjson);

  // allocate space for layer container structures
  int layer_size = models[modelindex].NUMLAYERS * sizeof(hlayer);
  models[modelindex].layers = malloc(layer_size);
  memset(models[modelindex].layers, 0, layer_size);
  if (!models[modelindex].layers)
  {
    fprintf(stderr, "error allocating memory for layer container!\n");
    exit(1);
  }

  int WVSIZE = models[modelindex].WVSIZE;

  // load word embeddings
  load_word_embeddings(modelindex);

  for (int i = 0; i < models[modelindex].NUMLAYERS; i++)
  {
#ifndef LOAD_WEIGHTS_ON_DEMAND
    load_layer_container(modelindex, i);
#else
    models[modelindex].layers[i].fp16_ln1_g = NULL;
    models[modelindex].layers[i].fp16_ln1_b = NULL;
    models[modelindex].layers[i].fp16_ln2_g = NULL;
    models[modelindex].layers[i].fp16_ln2_b = NULL;
    models[modelindex].layers[i].fp16_mlp_cfc_w = NULL;
    models[modelindex].layers[i].fp16_mlp_cfc_b = NULL;
    models[modelindex].layers[i].fp16_mlp_cproj_w = NULL;
    models[modelindex].layers[i].fp16_mlp_cproj_b = NULL;
    models[modelindex].layers[i].fp16_attn_cproj_w = NULL;
    models[modelindex].layers[i].fp16_attn_cproj_b = NULL;
    models[modelindex].layers[i].fp16_attn_cattn_w = NULL;
    models[modelindex].layers[i].fp16_attn_cattn_b = NULL;

    models[modelindex].layers[i].ln1_g = NULL;
    models[modelindex].layers[i].ln1_b = NULL;
    models[modelindex].layers[i].ln2_g = NULL;
    models[modelindex].layers[i].ln2_b = NULL;
    models[modelindex].layers[i].mlp_cfc_w = NULL;
    models[modelindex].layers[i].mlp_cfc_b = NULL;
    models[modelindex].layers[i].mlp_cproj_w = NULL;
    models[modelindex].layers[i].mlp_cproj_b = NULL;
    models[modelindex].layers[i].attn_cproj_w = NULL;
    models[modelindex].layers[i].attn_cproj_b = NULL;
    models[modelindex].layers[i].attn_cattn_w = NULL;
    models[modelindex].layers[i].attn_cattn_b = NULL;
#endif

    models[modelindex].layers[i].k = malloc(models[modelindex].CTXSIZE * WVSIZE * sizeof(bloom_precision));
    models[modelindex].layers[i].v = malloc(models[modelindex].CTXSIZE * WVSIZE * sizeof(bloom_precision));
  }

  // load layernorms
  load_final_layer_normalization(modelindex);

  models[modelindex].matchlist = (match_t *)malloc(MAXNUMMATCHES * sizeof(match_t));

  fprintf(stderr, "initial load complete.\nlayer data size: %ld\n", models[modelindex].WVSIZE * sizeof(bloom_precision));
  fflush(stderr);

  return 0;
}

// int raw_loader_test_main(void)
// {
//   // raw data for bfloat16's
//   char rawdata[8] = {0x0e, 0xa1, 0xed, 0x9c, 0xad, 0xa1, 0x54, 0x16};

//   double z = FP16_buf_to_double(rawdata);

//   load_huggingface_bloom_model_folder(BLOOM_560m, 0);
//   printf("Hello world!\n");
// }
