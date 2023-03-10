#include "common.h"
#include "server.h"
#include "raw_loader.h"
#include "client.h"
#include "utf8.h"

/* standardized benchmark run */
extern model_path_t model_definitions[9];

void benchmark(char *prompt, int lgt)
{

}

void purgeoldcontext(int p, int modelnum, int querynum)
{
  int i, j;

  for (i = 0; i < models[modelnum].CTXSIZE - p; i++)
    queries[querynum].context[i] = queries[querynum].context[i + p];
}

int breaks_called = 0;

int isBreaker(char *s)
{
  int o = 0;
  while (*s)
  {
    if (*s == '\n')
      o = 2;
    if (*s == '.')
      o = 2;
    if (*s == '!')
      o = 2;
    if (*s == '?')
      o = 2;
    if (*s == ',')
      o = 1;
    if (*s == ':')
      o = 1;
    if (*s == ';')
      o = 1;
    if (*s == '"')
      o = 1;
    if (*s == '(')
      o = 1;
    if (*s == ')')
      o = 1;
    if (*s >= 'a' && *s <= 'z')
      o = 0;
    s++;
  }
  return o;
}

bloom_precision tuneTemperatureByContext(int i, int querynum, int modelnum)
{
  bloom_precision lowtemp = queries[querynum].temperature;
  bloom_precision hightemp = queries[querynum].temperature_alt;
  if (hightemp <= 0)
    return lowtemp;
  if (!(i & 15))
    return hightemp;
  if (isBreaker(models[modelnum].tokenstrings[queries[querynum].context[i]]))
  {
    return hightemp;
  }
  int j = i - 3;
  while (j > 3)
  {
    if (queries[querynum].context[i] == queries[querynum].context[j])
    {
      if (queries[querynum].context[i - 1] == queries[querynum].context[j - 1] &&
          queries[querynum].context[i - 2] == queries[querynum].context[j - 2] &&
          queries[querynum].context[i - 3] == queries[querynum].context[j - 3])
      {
        return hightemp;
      }
    }
    j--;
  }
  return lowtemp;
}

float getMaxValue(bloom_precision *array, int size)
{
  int index = -1;
  bloom_precision max = 0;
  for (int i = 0; i < size; i++)
  {
    if (array[i] > max)
    {
      index = i;
      max = array[i];
    }
  }
  return index;
}

float getMaxValueReplace(bloom_precision *array, int size)
{
  int index = -1;
  bloom_precision max = 0;
  for (int i = 0; i < size; i++)
  {
    if (array[i] > max)
    {
      index = i;
      max = array[i];
    }
  }
  if (index != -1)
    array[index] = 0;

  return index;
}

void print_query_parameters(int querynum)
{
  fprintf(stderr, "----------------query #%d parameters---------------\n", querynum);
  fprintf(stderr, "temperature: %lf\n", queries[querynum].temperature);
  fprintf(stderr, "temperature_alt: %lf\n", queries[querynum].temperature_alt);
  fprintf(stderr, "minp: %lf\n", queries[querynum].minp);
  fprintf(stderr, "mode: %d\n", queries[querynum].mode);
  fprintf(stderr, "hardmax_gen: %d\n", queries[querynum].hardmax_gen);
  fprintf(stderr, "grammarmax_gen: %d\n", queries[querynum].grammarmax_gen);
  fprintf(stderr, "paragrammarmax_gen: %d\n", queries[querynum].paragrammarmax_gen);
  fprintf(stderr, "force_gen_tokens: %d\n", queries[querynum].force_gen_tokens);
  fprintf(stderr, "seed: %d\n", queries[querynum].seed);
  fprintf(stderr, "---------------------------------------------------\n");
}

void print_model_parameters(int modelnum)
{
  fprintf(stderr, "----------------model #%d parameters---------------\n", modelnum);
  fprintf(stderr, "modelname: %s\n", models[modelnum].modelname);
  fprintf(stderr, "numthreads: %d\n", models[modelnum].numthreads);
  fprintf(stderr, "WVSIZE: %d\n", models[modelnum].WVSIZE);
  fprintf(stderr, "CTXSIZE: %d\n", models[modelnum].CTXSIZE);
  fprintf(stderr, "NUMLAYERS: %d\n", models[modelnum].NUMLAYERS);
  fprintf(stderr, "NUMHEADS: %d\n", models[modelnum].NUMHEADS);
  fprintf(stderr, "HEADSIZE: %f\n", models[modelnum].HEADSIZE);
  fprintf(stderr, "RSQRT_HEADSIZE: %f\n", models[modelnum].RSQRT_HEADSIZE);
  fprintf(stderr, "---------------------------------------------------\n");
}

int countparagraphs(char *inputsentence)
{
  if (inputsentence == NULL)
    return 0;

  int paragraphcount = 0;
  for (int i=0;i<strlen(inputsentence);i++)
  {
    if (inputsentence[i] == '\n') // && inputsentence[i-1] == '\n')
    {
      paragraphcount++;
    }
  }
  return paragraphcount;
}

// int countsentences(char *inputsentence)
// {
//   if (inputsentence == NULL)
//     return 0;

//   int sentencecount = 0;
//   for (int i=0;i<strlen(inputsentence);i++)
//   {
//     if (inputsentence[i] == '.' || inputsentence[i] == '?' || inputsentence[i] == '!' )
//     {
//       sentencecount++;
//     }
//   }
//   return sentencecount;
// }

char *abbreviations[] = {"Mr.", "Mrs.", "Ms.", "Dr."}; // array of common abbreviations

int is_abbreviation(char *text, char *word) {
    // go back to the start of the current word
    int goback=1;
    while (goback==1)
    {
        if (word == text)
        {
            goback = 0;
            continue;
        }
        if (*word == ' ')
        {
            goback = 0;
            word++;
            continue;
        }
        word--;
    }
    int n = sizeof(abbreviations) / sizeof(abbreviations[0]);
    for (int i = 0; i < n; i++) {
        const char *abbr = abbreviations[i];
        int abbr_len = strlen(abbr);
        if (strncmp(word, abbr, abbr_len - 1) == 0 && word[abbr_len - 1] == '.') {
            return 1;
        }
    }
    return 0;
}

int countsentences(char *text) {
    if (text == NULL)
      return 0;
    int count = 0;
    int is_sentence_end = 1; // set to 1 to detect first sentence
    char *p = text;
    while (*p) {
        if (*p == '.' || *p == '?' || *p == '!') { // possible sentence end
            if (is_sentence_end) { // already at sentence end, ignore
                p++;
            } else {
                char *q = p - 1;
                while (q >= text && *q == ' ') { // skip trailing spaces
                    q--;
                }
                if (q >= text && is_abbreviation(text, q+1)) { // ignore if it's an abbreviation
                    p++;
                } else { // sentence ends here
                    count++;
                    is_sentence_end = 1;
                    p++;
                }
            }
        } else {
            is_sentence_end = 0;
            p++;
        }
    }
    return count;
}




// int has_repeat_ngram(int *context, int context_length, int proposed_token, int no_repeat_ngrams, int max_context_len)  {
//     if (no_repeat_ngrams <= 1 || context_length == 0) {
//         return 0;
//     }
//     int ngram[no_repeat_ngrams];
//     int i, j;
//     for (i = 0; i < no_repeat_ngrams-1; i++) {
//         ngram[i] = context[context_length-no_repeat_ngrams+1+i];
//     }
//     ngram[no_repeat_ngrams-1] = proposed_token;
//     for (i = 0; i < context_length-no_repeat_ngrams+2; i++) {
//         for (j = i+1; j < context_length-no_repeat_ngrams+2; j++) {
//             int k;
//             for (k = 0; k < no_repeat_ngrams; k++) {
//                 if (context[i+k] != ngram[k]) {
//                     break;
//                 }
//             }
//             if (k == no_repeat_ngrams) {
//                 return 1;
//             }
//         }
//     }
//     return 0;
// }

int has_repeat_ngram(int *context, int context_length, int proposed_token, int no_repeat_ngrams, int max_context_len)  {
    if (no_repeat_ngrams <= 1 || context_length == 0) {
        return 0;
    }
    int ngram[no_repeat_ngrams];
    int i, j;
    for (i = 0; i < no_repeat_ngrams-1; i++) {
        ngram[i] = context[context_length-no_repeat_ngrams+1+i];
    }
    ngram[no_repeat_ngrams-1] = proposed_token;
    int searchstart = context_length-max_context_len;
    if (searchstart < 0 || max_context_len == -1)
    {
      searchstart = 0;
    }
    for (i = searchstart; i < context_length-no_repeat_ngrams+2; i++) {
        for (j = i+1; j < context_length-no_repeat_ngrams+2; j++) {
            int k;
            for (k = 0; k < no_repeat_ngrams; k++) {
                if (context[i+k] != ngram[k]) {
                    break;
                }
            }
            if (k == no_repeat_ngrams) {
                return 1;
            }
        }
    }
    return 0;
}


void generate(int start, int genstart_, int genend_, int modelnum, int querynum, bool displayprompt)
{
  

  int tok = -1;
  print_query_parameters(querynum);
  print_model_parameters(modelnum);

  int hardmax_gen = queries[querynum].hardmax_gen;
  int grammarmax_gen = queries[querynum].grammarmax_gen;
  int paragrammarmax_gen = queries[querynum].paragrammarmax_gen;
  int no_repeat_ngrams = queries[querynum].no_repeat_ngrams;
  int stop_after_ngram_repeats = queries[querynum].stop_after_ngram_repeats;
  int start_n_gram_search_on_current_response = queries[querynum].start_n_gram_search_on_current_response;
  int ngram_repeats_detected = 0;
  


  int genstart_token_index=-1;
  int paragraphs = 0;
  int gentokensleft = queries[querynum].force_gen_tokens;
  int tokens_generated = 0;
  if (gentokensleft == -1)
  {
    gentokensleft = genend_ / 2;
  }

  long long dsz;
  long long dcnt;
  long long size;
  long long sz;
  bloom_precision FP16_size = 2.0;
  int modelindex = modelnum;

  queries[querynum].currslot = start;
  queries[querynum].genstart = genstart_;
  queries[querynum].genend = genend_;
  char reloading = 0;

  while ((queries[querynum].hardmax_gen <= 0 || queries[querynum].currslot < queries[querynum].genend) &&
         (queries[querynum].grammarmax_gen <= 0 || countsentences(queries[querynum].response)  < queries[querynum].grammarmax_gen) &&
         (queries[querynum].paragrammarmax_gen <= 0 || countparagraphs(queries[querynum].response)  < queries[querynum].paragrammarmax_gen) &&
         (queries[querynum].hardmax_gen <= 0 || (queries[querynum].hardmax_gen > 0 && tokens_generated < queries[querynum].hardmax_gen)) &&
         tok != 2)
  {

    runModel(queries[querynum].currwv, queries[querynum].currslot, modelnum, querynum);
#ifdef EXTRACT_WEIGHTS_ON_DEMAND
    if (models[modelindex].wte == NULL)
    {
      sz = models[modelindex].WVSIZE * (((long long)models[modelnum].numwtetokens) + NUMUSERTOKENS) * FP16_size;
      dsz = sz * FP16_size;
      dcnt = sz / FP16_size;
      models[modelindex].wte = (wte_t *)malloc(sizeof(bloom_precision) * (dcnt + MAXUSERTOKENS * ((long long)models[modelindex].WVSIZE)));
      // copy values
      memset(&(models[modelindex].wte[dcnt]), 0, sizeof(bloom_precision) * (MAXUSERTOKENS * ((long long)models[modelindex].WVSIZE)));
      if (models[modelnum].use_bfloat16)
      {
        uint16_t *ptr = models[modelindex].fp16_wte;
        BFloat16ToFloat((uint16_t *)ptr, models[modelindex].wte, dcnt);
      }
      else
      {
        for (long long i = 0; i < dcnt; i++)
        {
          uint16_t *ptr = models[modelindex].fp16_wte;
          models[modelindex].wte[i] = half_to_float(*((unsigned short *)(ptr + i))).f;
        }
      }
    }
#endif

#ifdef MEASURE_TOKEN_TIME
    clock_t token_t;
    token_t = clock();
#endif

    tok = queries[querynum].context[queries[querynum].currslot];
    if (tok >= 0 && queries[querynum].currslot < queries[querynum].genstart)
    {
      if (!reloading || models[modelnum].verbose >= 1)
      {
        char *buffer_new = str_replace(models[modelnum].tokenstrings[tok], "Ġ", " ");
        char *buffer_new2 = str_replace(buffer_new, "Ċ", "\n");
        printf("%s", buffer_new2);
        fflush(stdout);
        free(buffer_new);
        free(buffer_new2);
      }

      if (queries[querynum].lm_logits == NULL)
      {
        queries[querynum].lm_logits = (bloom_precision *)malloc(models[modelnum].numwtetokens * sizeof(bloom_precision));
      }
    }

    queries[querynum].currslot++;
    if (queries[querynum].currslot >= queries[querynum].genstart)
    {
      bool allowspecial = true;
      if (queries[querynum].force_gen_tokens >= 0)
      {
        allowspecial = false;
        if (gentokensleft == 0)
        {
          queries[querynum].temperature *= 0.9999;
          queries[querynum].temperature_alt *= 0.9;
          queries[querynum].minp *= 0.9;
          allowspecial = true;
        }
      }

      
      if (queries[querynum].mode == 0)
      {
        linear_transform(queries[querynum].currwv, queries[querynum].lm_logits, models[modelnum].wte, NULL, models[modelnum].WVSIZE, models[modelnum].numwtetokens);
        tok = (int)getMaxValueReplace(queries[querynum].lm_logits, models[modelnum].numwtetokens);
        if (tok == 2)
        {
          tok = (int)getMaxValueReplace(queries[querynum].lm_logits, models[modelnum].numwtetokens);
        }       
        if (genstart_token_index >=0)
        {          
          int start_n_gram_search = -1;
          if (start_n_gram_search_on_current_response==1)
          {
            start_n_gram_search = genstart_token_index;
          }
          int repeats = has_repeat_ngram(&(queries[querynum].context[genstart_token_index]), tokens_generated, tok, queries[querynum].no_repeat_ngrams, start_n_gram_search );
          if (repeats == 1)
          {
            ngram_repeats_detected++;
          }
          while (repeats == 1)
          {            
            tok = (int)getMaxValueReplace(queries[querynum].lm_logits, models[modelnum].numwtetokens);        
            repeats = has_repeat_ngram(&(queries[querynum].context[genstart_token_index]), tokens_generated+1, tok, queries[querynum].no_repeat_ngrams, start_n_gram_search );
          }
        }
      }
      else if (queries[querynum].mode == 1)
      {
        int match;
        reloading = 0;

        bloom_precision tunedTemp = tuneTemperatureByContext(queries[querynum].currslot - 1, querynum, modelnum);
        matchToTokens(queries[querynum].currwv, models[modelnum].matchlist, queries[querynum].nummatches, tunedTemp, modelnum);

        match = pickmatch(models[modelnum].matchlist, queries[querynum].nummatches, queries[querynum].minp, allowspecial, modelnum);
        tok = models[modelnum].matchlist[match].tok;
        if (tok == 2 && (queries[querynum].force_gen_tokens == -2 || countsentences(queries[querynum].response) == 0))
        {
            match++;
            tok = models[modelnum].matchlist[match].tok;          
        }
        //tok = replacetoken(tok, modelnum);

        if (genstart_token_index >=0)
        {          
          int start_n_gram_search = -1;
          if (start_n_gram_search_on_current_response==1)
          {
            start_n_gram_search = genstart_token_index;
          }
          int repeats = has_repeat_ngram(&(queries[querynum].context[genstart_token_index]), tokens_generated, tok, queries[querynum].no_repeat_ngrams, -1 );
          if (repeats == 1)
          {
            ngram_repeats_detected++;
          }
          while (repeats == 1)
          {            
            //tok = (int)getMaxValueReplace(queries[querynum].lm_logits, models[modelnum].numwtetokens);        
            match++;
            tok = models[modelnum].matchlist[match].tok;
            repeats = has_repeat_ngram(&(queries[querynum].context[genstart_token_index]), tokens_generated+1, tok, queries[querynum].no_repeat_ngrams, -1 );
          }
        }        
      }
      tokens_generated++;

      if (stop_after_ngram_repeats != -1 && ngram_repeats_detected >= stop_after_ngram_repeats)
      {
        // if we have stop after ngram repeats set, then we can make the
        grammarmax_gen = countsentences(queries[querynum].response)+1;
        queries[querynum].grammarmax_gen = countsentences(queries[querynum].response)+1;
      }

      if (tok == 1 || tok == 2)
      {
        queries[querynum].context[queries[querynum].currslot] = 3;
      }
      else
      {
        queries[querynum].context[queries[querynum].currslot] = tok;
      }


      if (queries[querynum].force_gen_tokens >= 0)
      {
        if (gentokensleft > 0)
          gentokensleft--;
      }

      if (genstart_token_index == -1)
      {
        genstart_token_index = queries[querynum].currslot;
      }
      if (tok >= 4)
      {
        int q=0;
        q++;

        char *buffer_new = str_replace(models[modelnum].tokenstrings[tok], "Ġ", " ");
        char *buffer_new2 = str_replace(buffer_new, "Ċ", "\n");
        char *buffer_new3 = str_replace(buffer_new2, "âĢĻ", "'");
        char *buffer_new4 = str_replace(buffer_new3, "âĢĺ", "'");
        char *buffer_new5 = str_replace(buffer_new4, "âĢľ", "\"");
        char *buffer_new6 = str_replace(buffer_new5, "âĢĿ", "\"");
        char *buffer_new7 = str_replace(buffer_new6, "Ã©", "é");
        char *buffer_new8 = str_replace(buffer_new7, "Ã³", "ó");


        printf("%s", buffer_new8);
        fflush(stdout);

        int len = 0;
        if (queries[querynum].response != NULL)
        {
          len = strlen(queries[querynum].response) + 1;
        }
        char *temp = malloc(len + strlen(buffer_new8) + 1);
        if (queries[querynum].response != NULL)
        {
          sprintf(temp, "%s%s", queries[querynum].response, buffer_new8);
          free(queries[querynum].response);
        }
        else
        {
          sprintf(temp, "%s", buffer_new8);
        }

        queries[querynum].response = temp;
        free(buffer_new);
        free(buffer_new2);
        free(buffer_new3);
        free(buffer_new4);
        free(buffer_new5);
        free(buffer_new6);
        free(buffer_new7);
        free(buffer_new8);

      }
    }

    if (queries[querynum].currslot >= models[modelnum].CTXSIZE)
    {
      purgeoldcontext(models[modelnum].CTXSIZE / 2, modelnum, querynum);
      queries[querynum].currslot -= models[modelnum].CTXSIZE / 2;
      queries[querynum].genstart -= models[modelnum].CTXSIZE / 2;
      queries[querynum].genend -= models[modelnum].CTXSIZE / 2;
    }
    if (breaks_called)
    {
      breaks_called = 0;
      ttyui();
    }

#ifdef MEASURE_TOKEN_TIME
    token_t = clock() - token_t;
    double token_time_taken = ((double)token_t) / CLOCKS_PER_SEC; // in seconds

    fprintf(stderr, "\ntoken lookup: %fs \n", token_time_taken);
    fflush(stderr);
#endif
  }
}

/**
 * @brief  Release resources allocated by model
 * @note   
 * @param  modelnum: 
 * @retval None
 */
void freeModel(int modelnum)
{
  free(models[modelnum].matchlist);
  free(models[modelnum].alibi);
  free(models[modelnum].tokenflags);

  models[modelnum].matchlist = NULL;
  models[modelnum].alibi = NULL;
  models[modelnum].tokenflags = NULL;
}

/**
 * @brief  Initialize Model
 * @note   
 * @param  *modelpath: 
 * @param  modelnum: 
 * @retval None
 */
int initModel(char *modelpath, int modelnum)
{
  char path_to_usebfloat16[2048];
  fprintf(stderr, "loading model from %s...\n", modelpath);

  sprintf(path_to_usebfloat16, "%s/usebfloat16", modelpath);

  // for now
  models[modelnum].numthreads = global_numthreads;

  if (access(path_to_usebfloat16, F_OK) == 0)
  {
    fprintf(stderr, "using bfloat16s\n");
    models[modelnum].use_bfloat16 = true;
  }
  else
  {
    models[modelnum].use_bfloat16 = false;
  }

  load_huggingface_bloom_model_folder(modelpath, modelnum);

  models[modelnum].matchlist = (match_t *)malloc(MAXNUMMATCHES * sizeof(match_t));
  models[modelnum].closest_power_of_2 = pow(2, floor(log2((bloom_precision)models[modelnum].NUMHEADS)));
  models[modelnum].base = pow(2, (-(pow(2, -(log2(models[modelnum].closest_power_of_2) - 3)))));
  models[modelnum].alibi = malloc(sizeof(bloom_precision) * models[modelnum].closest_power_of_2 * models[modelnum].CTXSIZE);

  models[modelnum].tokenflags = (char *)malloc((models[modelnum].numtokens + MAXUSERTOKENS) * sizeof(char));
  memset(models[modelnum].tokenflags, 0, (models[modelnum].numtokens + MAXUSERTOKENS) * sizeof(char));
  models[modelnum].nummodeltokens = models[modelnum].numtokens;
  models[modelnum].isInitialized = true;
  models[modelnum].inUse = true;
  models[modelnum].verbose = 0;
  models[modelnum].quanter_wte = 1.0;
  fprintf(stderr, "model load complete\n");

  return modelnum;
}

/**
 * @brief  Release resources allocated by query
 * @note   
 * @param  querynum: 
 * @retval None
 */
void freeQuery(int querynum)
{
  free(queries[querynum].context);
  free(queries[querynum].currwv);
  free(queries[querynum].attentions);
  free(queries[querynum].attentions_presoftmax);
  free(queries[querynum].attention_arrange_tensor);
  free(queries[querynum].attention_mask);
  free(queries[querynum].att);

  queries[querynum].context = NULL;
  queries[querynum].currwv = NULL;
  queries[querynum].attentions = NULL;
  queries[querynum].attentions_presoftmax = NULL;
  queries[querynum].attention_arrange_tensor = NULL;
  queries[querynum].attention_mask = NULL;
  queries[querynum].att = NULL;
}

/**
 * @brief  Initialize Query
 * @note   
 * @param  modelnum: 
 * @param  querynum: 
 * @retval None
 */
void initQuery(int modelnum, int querynum)
{
  fprintf(stderr, "initializing query #%d\n", querynum);
  queries[querynum].context = (token_t *)malloc(models[modelnum].CTXSIZE * sizeof(token_t));
  queries[querynum].currwv = (bloom_precision *)malloc(models[modelnum].WVSIZE * sizeof(bloom_precision));
  queries[querynum].attentions = malloc(models[modelnum].CTXSIZE * models[modelnum].NUMLAYERS * models[modelnum].NUMHEADS * sizeof(bloom_precision));
  queries[querynum].attentions_presoftmax = malloc(models[modelnum].CTXSIZE * models[modelnum].NUMLAYERS * models[modelnum].NUMHEADS * sizeof(bloom_precision));
  queries[querynum].attention_arrange_tensor = malloc(sizeof(bloom_precision) * models[modelnum].CTXSIZE);
  queries[querynum].attention_mask = malloc(sizeof(bloom_precision) * models[modelnum].CTXSIZE);
  queries[querynum].no_repeat_ngrams = 4;
  // stop at the end of the sentence after the first repeat
  queries[querynum].stop_after_ngram_repeats = 1;
  queries[querynum].start_n_gram_search_on_current_response = 1;
  
  for (int i = 0; i < models[modelnum].CTXSIZE; i++)
    queries[querynum].context[i] = models[modelnum].emptytoken;

  queries[querynum].att = malloc(sizeof(bloom_precision) * models[modelnum].closest_power_of_2 * models[modelnum].CTXSIZE * models[modelnum].NUMHEADS + models[modelnum].CTXSIZE);

  queries[querynum].isInitialized = true;

  fprintf(stderr, "query initialization complete\n");
}

int matchcolortopalette(int r, int g, int b)
{
  int i;
  bloom_precision min;
  int where;
  for (i = 0; i < models[0].numtokens; i++)
  {
    bloom_precision *p = palette + i * 3;
    bloom_precision d2 = (p[0] - r) * (p[0] - r) +
                         (p[1] - g) * (p[1] - g) +
                         (p[2] - b) * (p[2] - b);
    if (i == 0 || d2 < min)
    {
      min = d2;
      where = i;
    }
  }
  return where;
}

#ifdef HAVE_SDL
int tokenize_image(char *fn)
{
  int ctxdim = sqrt(CTXSIZE);
  int i, j, w, h;
  SDL_Surface *img = IMG_Load(fn);
  if (!img)
    return 0;
  fprintf(stderr, "loading image %s: %d x %d", fn, img->w, img->h);
  w = img->w;
  if (w > ctxdim)
    w = ctxdim;
  h = img->h;
  if (h > ctxdim)
    h = ctxdim;
  fprintf(stderr, " -> %d x %d\n", w, h);
  for (j = 0; j < h; j++)
    for (i = 0; i < w; i++)
    {
      Uint8 r, g, b;
      int c = *((int *)(img->pixels + img->pitch * j + img->format->BytesPerPixel * i));
      SDL_GetRGB(c, img->format, &r, &g, &b);
      context[j * ctxdim + i] = matchcolortopalette(r, g, b);
    }
  fprintf(stderr, "loaded\n");
  return (h - 1) * ctxdim + w;
  // SDL_FreeSurface(img);
}
#endif

#ifdef ENABLE_TTYUI
int handlesignal(int s)
{
  breaks_called++;
  if (breaks_called >= 3)
  {
    breaks_called = 0;
    ttyui();
  }

  return 0;
}
#endif

void configcmd(char *cmd, char *param)
{
  char changed = 0;
  if (!strncmp(cmd, "model", 5))
  {
    models[0].modelpath = strndup(param, strchr(param, '\n') - param);
    return;
  }
  for (int i = 0; i < NUMVARS; i++)
    if (!strncmp(cmd, settingvars[i].name, param - 1 - cmd))
    {
      if (settingvars[i].type)
        *((int *)settingvars[i].ptr) = strtol(param, NULL, 10);
      else
        *((bloom_precision *)settingvars[i].ptr) = strtof(param, NULL);
      changed = 1;
    }
  if (!changed)
  {
    fprintf(stderr, "couldn't parse configline `%s'\n", cmd);
    exit(1);
  }
}

void flagTokenForReplace(int t1, int t2, int flag)
{
  if (!models[0].tokenrepls)
  {
    models[0].tokenrepls = (token_t *)malloc(models[0].numtokens * sizeof(token_t));
  }
  models[0].tokenflags[t1] = flag;
  models[0].tokenrepls[t1] = t2;
  //  fprintf(stderr,"flagged %d->%d (%d)\n",t1,t2,flag);
}

void flagTokenSetForReplace(char *t1s, char *t2s, int flag)
{
  int t1t = tokenize(t1s,0);
  int t2t = tokenize(t2s,0);
  if (!strcmp(models[0].tokenstrings[t1t], t1s) &&
      !strcmp(models[0].tokenstrings[t2t], t2s))
  {
    flagTokenForReplace(t1t, t2t, flag);
  }

  if (*t1s >= 'a' && *t1s <= 'z')
  {
    char *t1b = strdup(t1s);
    char *t2b = strdup(t2s);
    t1b[0] += 'A' - 'a';
    if (*t2s >= 'a' && *t2s <= 'z')
      t2b[0] += 'A' - 'a';
    flagTokenSetForReplace(t1b, t2b, flag);
    free(t1b);
    free(t2b);
  }
  if (*t1s == ' ' && *t2s == ' ')
    flagTokenSetForReplace(t1s + 1, t2s + 1, flag);
}

void readconfig(char *s, int dotokens)
{
  while (*s)
  {
    if (*s != '#')
    {
      char *linestart;
      while (*s == ' ' || *s == '\t')
        s++;
      linestart = s;
      if (*linestart == '*' && (linestart[1] >= '0' && linestart[1] <= '9'))
      {
        // massflag tokens by substring
        if (dotokens)
        {
          int flag = linestart[1] - '0' - 1;
          int i;
          int linelgt = strchr(linestart, '\n') - linestart;
          char *lineend = strchr(s, '\n');
          if (!lineend)
            lineend = s + strlen(s);
          lineend--;
          char *needle = strndup(linestart + 2, lineend - linestart - 2 + 1);
          fprintf(stderr, "massflagging *%s* to %d...\n", needle, flag);
          for (i = 0; i < models[0].numtokens; i++)
          {
            if (strstr(models[0].tokenstrings[i], needle))
            {
              // fprintf(stderr,"token %d (%s) flagged to %d\n",i,tokenstrings[i],flag);
              models[0].tokenflags[i] = flag;
            }
          }
          free(needle);
        }
      }
      else if (*linestart >= '0' && *linestart <= '9')
      {
        // flag single token
        if (dotokens)
        {
          int flag = *linestart - '0' - 1;
          int token = tokenize(linestart + 1,0);
          if (token >= 0)
            models[0].tokenflags[token] = flag;
        }
      }
      else if (*linestart == 'r' || *linestart == 'R' || *linestart == 'x' ||
               *linestart == 'X')
      {
        if (dotokens)
        {
          char *ss;
          int flag = isupper(*linestart) ? 5 : 4;
          ss = linestart + 1;
          int t1 = tokenize(ss,0);
          ss += strlen(models[0].tokenstrings[t1]);
          int t2 = tokenize(ss,0);
          ss += strlen(models[0].tokenstrings[t2]);
          if (*ss != '\n' && *ss != '\0')
          {
            fprintf(stderr, "error: need exactly two tokens for r/R/x/X!\n"
                            "(%d %d %s ends with %d)\n",
                    t1, t2, linestart, *ss);
          }
          else
          {
            flagTokenSetForReplace(models[0].tokenstrings[t1], models[0].tokenstrings[t2], flag);
            if (*s == 'x' || *s == 'X')
              flagTokenSetForReplace(models[0].tokenstrings[t2], models[0].tokenstrings[t1], flag);
          }
        }
      }
      else
      {
        // other commands (set parameters)
        while (*s && *s != '=' && *s != ' ' && *s != '\n')
          s++;
        if (*s == ' ' || *s == '=')
          configcmd(linestart, s + 1);
      }
    }
    while (*s && *s != '\n')
      s++;
    if (*s)
      s++;
  }
}

int(analyzetokens_cmp)(const void *a, const void *b)
{
  return (((int *)a)[0] > ((int *)b)[0]) ? -1 : 1;
}

void analyzetokens()
{
  int i, j;
  int tab[models[0].numtokens * 2];
  for (j = 0; j < 768; j++)
  {
    int howmany = 0;
    for (i = 0; i < models[0].numtokens; i++)
    {
      if (models[0].tokenstrings[i][0] == ' ')
      {
        tab[howmany * 2 + 0] = models[0].wte[i * 768 + j];
        tab[howmany * 2 + 1] = i;
        howmany++;
      }
    }
    qsort(tab, howmany, sizeof(int) * 2, analyzetokens_cmp);
    printf("component %d:", j);
    for (i = 0; i < 5; i++)
      printf("%s[%d]", models[0].tokenstrings[tab[i * 2 + 1]],
             tab[i * 2]);
    printf(" (not:");
    for (i = 0; i < 5; i++)
      printf("%s[%d]", models[0].tokenstrings[tab[(howmany - 1 - i) * 2 + 1]],
             tab[(howmany - 1 - i) * 2]);
    printf(")\n");
  }
}

char *str_replace(const char *in, const char *pattern, const char *by)
{
  size_t outsize = strlen(in) + 1;
  // TODO maybe avoid reallocing by counting the non-overlapping occurences of pattern
  char *res = malloc(outsize);
  // use this to iterate over the output
  size_t resoffset = 0;

  char *needle;
  while (needle = strstr(in, pattern))
  {
    // copy everything up to the pattern
    memcpy(res + resoffset, in, needle - in);
    resoffset += needle - in;

    // skip the pattern in the input-string
    in = needle + strlen(pattern);

    // adjust space for replacement
    outsize = outsize - strlen(pattern) + strlen(by);
    res = realloc(res, outsize);

    // copy the pattern
    memcpy(res + resoffset, by, strlen(by));
    resoffset += strlen(by);
  }

  // copy the remaining input
  strcpy(res + resoffset, in);

  return res;
}

void testutf8()
{
  

}

int main(int argc, char **argv)
{

  testutf8();
  char *loadDefaultModel=NULL;
  global_numthreads = 12;
  int models_size = sizeof(model_t) * MAXNUMMODELS;
  models = (model_t *)malloc(models_size);
  memset(models, 0, models_size);
  int queries_size = sizeof(query_t) * MAXNUMQUERIES;
  queries = (query_t *)malloc(sizeof(query_t) * MAXNUMQUERIES);
  memset(queries, 0, queries_size);

  settingvars = (settingvars_t *)malloc(sizeof(settingvars_t) * NUMVARS);

  settingvars[0].name = strdup("temperature");
  settingvars[0].ptr = &queries[0].temperature;
  settingvars[0].type = 0;
  settingvars[1].name = strdup("temperature_alt");
  settingvars[1].ptr = &queries[0].temperature_alt;
  settingvars[1].type = 0;
  settingvars[2].name = strdup("minp");
  settingvars[2].ptr = &queries[0].minp;
  settingvars[2].type = 0;
  settingvars[3].name = strdup("nummatches");
  settingvars[3].ptr = &queries[0].nummatches;
  settingvars[3].type = 1;
  settingvars[4].name = strdup("genend");
  settingvars[4].ptr = &queries[0].genend;
  settingvars[4].type = 1;
  settingvars[5].name = strdup("numthreads");
  settingvars[5].ptr = &models[0].numthreads;
  settingvars[5].type = 1;

  models[0].quanter_wte = 1.0;

  // settings & defaults
  queries[0].temperature = 1.0;
  queries[0].temperature_alt = 1.0;
  queries[0].nummatches = 80;
  queries[0].mode = 0;
  queries[0].minp = 0;
  queries[0].seed = 0;
  queries[0].hardmax_gen = -1;
  queries[0].grammarmax_gen = -1;
  queries[0].force_gen_tokens = -2;

  models[0].numthreads = 1;
  models[0].verbose = 0;

  // other runtime options
  char *prompt = NULL;
  char *promptfile = NULL;
  char *configfile = NULL;
  char *clientServerAddress = NULL;
  int clientServerPort = 8081;
  bool doClient = false;
  char *samplingmode = NULL;
  char *packedfiletosave = NULL;
  int lengthtogen = 0;
  char wannastartui = 0;
  char wannabenchmark = 0;

  int i, here = 0;
  bool temp_alt_set = false;
  startServer = false;

  for (i = 1; i < argc; i++)
  {
    if (argv[i][0] == '-')
    {
      char *s = argv[i] + 1;
      while (*s == '-')
        s++;
      for (; *s; s++)
      {
        if (*s == 'h')
        {
          fprintf(stderr,
                  "Usage: %s <options> [modelpath]\n"
                  "-h          show this help\n"
                  "-d port     start server on port\n"
                  "-f file.txt read prompt from file\n"
                  "-F file.txt pfg_gpt_config file location\n"
                  "-l 512      set maximum length of text to output (in tokens)\n"
                  "-t 1.0      set noise temperature for match randomization\n"
                  "-a 1.0      set alternative temperature used at sentence boundaries etc\n"
                  "-T 4        set number of threads\n"
                  "-m mode     set sampling mode, (greedy/sampling)\n"
                  "-M model    use specified modelname\n"
                  "-s 123456   set random number seed (0 = use timer)\n"
                  "-c server   connect to a server as a client\n"
                  "-p port     use specified port when connecting to server (default 8081 if unspecified)\n"
                  "-u          start ui even with -p and -f\n"
                  "-b          run benchmark\n"
                  "-g n        force generate at least n tokens\n"
                  "-x n        hard maximum n tokens\n"
                  "-y n        generate at least n sentences\n"
                  "-Y n        generate at least n paragraphs\n"
                  "-v          verbose/debug output\n"
                  "-z m        set minp value to m (float)\n"
                  "-L          start lua interpreter\n",
                  argv[0]);
          exit(1);
        }
        if (i < argc - 1)
        {
          if (*s == 'p')
          {
            clientServerPort = atoi(argv[++i]);
          }
          if (*s == 'f')
            promptfile = argv[++i];
          if (*s == 'F')
            configfile = argv[++i];
          if (*s == 't')
          {
            queries[0].temperature = atof(argv[++i]);
            if (temp_alt_set==false)
            {
              // so we don't have to always set temperature alt
              queries[0].temperature_alt = queries[0].temperature;
            }
          }
          if (*s == 'a')
          {
            queries[0].temperature_alt = atof(argv[++i]);
            temp_alt_set = true;
          }
          if (*s == 'z')
            queries[0].minp = atof(argv[++i]);            
          if (*s == 'T')
            global_numthreads = atoi(argv[++i]);
          if (*s == 's')
            queries[0].seed = atoi(argv[++i]);
          if (*s == 'd')
          {
            serverPort = atoi(argv[++i]);
            startServer = true;
          }
          if (*s == 'g')
          {         
            queries[0].force_gen_tokens = atoi(argv[++i]);
          }
          if (*s == 'x')
          {
            queries[0].hardmax_gen = atoi(argv[++i]);
          }
          if (*s == 'y')
          {
            queries[0].grammarmax_gen = atoi(argv[++i]);
          }      
          if (*s == 'Y')
          {
            queries[0].paragrammarmax_gen = atoi(argv[++i]);
          }                                         
          if (*s == 'm')
          {
            samplingmode = argv[++i];
            if (strcmp(samplingmode, "greedy") == 0)
            {
              queries[0].mode = 0;
            }
            else if (strcmp(samplingmode, "sampling") == 0)
            {
              queries[0].mode = 1;
            }
          }
          if (*s == 'M')
          {
            loadDefaultModel = argv[++i];
          }          
          if (*s == 'c')
          {
            clientServerAddress = argv[++i];
            doClient = true;
          }
          if (*s == 'l')
            lengthtogen = atoi(argv[++i]);
          // if (*s == 'Z')
          //   packedfiletosave = argv[++i];
        }
        if (*s == 'H')
          wannastartui = 3;
        if (*s == 'L')
          wannastartui = 2;
        if (*s == 'u')
          wannastartui = 1;
        if (*s == 'v')
          models[0].verbose++;
        if (*s == 'b')
        {
          wannabenchmark = 1;
          queries[0].temperature = 1.2;
          queries[0].seed = 70177;
          lengthtogen = 256;
          queries[0].nummatches = 40;
          queries[0].minp = 0;
          prompt = " Suddenly, a magical floppy disk";
        }
      }
    }
    else
    {
      models[0].modelpath = argv[i];
    }
  }

  // load model location config file
  if (configfile == NULL)
  {
    // use default config file location
    configfile = strdup("./pfg_gpt_config");
  }
  if (access(configfile, F_OK) == 0)
  {
    load_model_paths(configfile);
  }    

  if (loadDefaultModel == NULL)
  {
    loadDefaultModel = strdup("bloom-560m");
  }
  strcpy(models[0].modelname, loadDefaultModel);  

  if (doClient)
  {
    client_main(clientServerAddress, clientServerPort);
  }

  if (startServer)
  {
    models[0].modelpath = lookup_model_path(models[0].modelname);
    initModel(models[0].modelpath, 0);    
  }
  else
  {
    models[0].modelpath = lookup_model_path(models[0].modelname);
    initModel(models[0].modelpath, 0);
  }
  initQuery(0, 0);
  queries[0].in_use = true; //reserve the 0th query
  if (packedfiletosave)
  {
    fprintf(stderr, "saving packed model to file %s...\n", packedfiletosave);
    savepackedmodel(packedfiletosave,0);
    exit(0);
  }
  if (!queries[0].seed)
  {
    queries[0].seed = time(NULL);
    fprintf(stderr, "seed from time(): %d\n", queries[0].seed);
  }
  srand(queries[0].seed);

  if (wannastartui == 2)
  {
#ifdef HAVE_LUA
    vzlua(NULL);
#endif
    return 0;
  }

  if (promptfile && !palette)
  {
    prompt = readtextfile(promptfile, NULL);
    prompt[strlen(prompt) - 1] = 0;
  }

#ifdef HAVE_SDL
#ifdef ENABLE_SDLUI
  if (wannastartui || !prompt)
    ui_init();
#endif
#endif
  // iqtest();

  int promptlgt = 0;
  char *prompt_new = NULL;
  char *prompt_new2 = NULL;
  if (prompt)
  {
    prompt_new = str_replace(prompt, " ", "Ġ");
    prompt_new2 = str_replace(prompt_new, "\n", "Ċ");
    promptlgt = tokenize_to_context(prompt_new2, 0, 0, 0);
    free(prompt_new);
    free(prompt_new2);
  }
#ifdef HAVE_SDL
  if (palette && promptfile)
    promptlgt = tokenize_image(promptfile);
#endif
  fprintf(stderr, "Prompt length: %d tokens\n", promptlgt);

#ifdef HAVE_SDL
#ifdef ENABLE_SDLUI
  if (wannastartui || !prompt)
  {
    ui_run();
    return 0;
  }
#endif
#endif

#ifdef ENABLE_TTYUI
  signal(SIGINT, handlesignal);
#endif
  if (lengthtogen <= 0)
    lengthtogen = models[0].CTXSIZE + 1;
  if (startServer)
  {
    fprintf(stderr, "starting server...\n");
    fflush(stderr);
    // start the server
    ServerStart(serverPort);
  }

  // queries[0].force_gen_tokens = 10;
  generate(0, promptlgt, lengthtogen, 0, 0, true);
  printf("\n");
  return 0;
}

/*

configfile:
param value

*/
