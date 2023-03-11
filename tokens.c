#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <regex.h>
#include "common.h"
#include "utf8.h"

/*** load the token dictionary (for normal text-based gpt-2 models) ***/

int loadtokens_from_tokendata(char *tokendata, int numtoks)
{
  char *s = tokendata;
  models[0].tokenstrings = malloc((numtoks + 1 + MAXUSERTOKENS) * sizeof(char *));
  int i = 0;
  while (i < numtoks)
  {
    models[0].tokenstrings[i] = s;
    s += strlen(s) + 1;
    i++;
  }
  models[0].tokenstrings[i] = NULL;
  fprintf(stderr, "%d tokenstrings retrieved\n", i);
  models[0].numtokens = i;
  return 0;
}

int loadtokens(char *path)
{
  int tokendatalgt;
  models[0].tokendata = (char *)readfile("tokens.dat", &tokendatalgt, path);
  if (!models[0].tokendata)
  {
    fprintf(stderr, "couldn't load tokens.dat...\n");
    return 1;
  }
  char *s = models[0].tokendata;
  int numtoks = 0;
  while (s < models[0].tokendata + tokendatalgt)
  {
    int slen = strlen(s);
    s += slen + 1;
    numtoks++;
  }
  return loadtokens_from_tokendata(models[0].tokendata, numtoks);
}

/*** load the color palette (for imagegpt models) ***/

int loadpalette(char *path)
{
  int sz, i;
  palette = (bloom_precision *)readfile("kmeans_centers.npy.raw", &sz, path);
  if (!sz)
  {
    fprintf(stderr, "couldn't find color clusters\n");
    return 1;
  }
  models[0].numtokens = sz / (3 * sizeof(bloom_precision));

  // normalize to 0..255
  bloom_precision min = 0, max = 0;
  for (i = 0; i < models[0].numtokens * 3; i++)
  {
    if (palette[i] < min)
      min = palette[i];
    if (palette[i] > max)
      max = palette[i];
  }
  for (i = 0; i < models[0].numtokens * 3; i++)
  {
    int a = round(255 * (palette[i] - min) / (max - min));
    palette[i] = a;
  }

  // create pseudotokens
  models[0].tokenstrings = malloc((models[0].numtokens + 1 + MAXUSERTOKENS) * sizeof(char *));
  for (i = 0; i < models[0].numtokens; i++)
  {
    char buf[8];
    sprintf(buf, " %02x%02x%02x",
            (int)(palette[i * 3 + 0]),
            (int)(palette[i * 3 + 1]),
            (int)(palette[i * 3 + 2]));
    models[0].tokenstrings[i] = strdup(buf);
  }
  models[0].tokenstrings[models[0].numtokens] = NULL;
  fprintf(stderr, "%d colors imported\n", models[0].numtokens);
  return 0;
}

/*** helper functions ***/

// TODO fix for PKD
int allocusertoken(bloom_precision *wv, char *name)
{
  int WVSIZE = models[0].WVSIZE;

  int tok;
  if (models[0].numtokens >= models[0].nummodeltokens + MAXUSERTOKENS)
    return -1;
  tok = models[0].numtokens;
  models[0].numtokens++;
  models[0].userwte[tok - models[0].nummodeltokens] = malloc(WVSIZE * sizeof(pkdflt));
  models[0].tokenstrings[tok] = strdup(name ? name : "UNNAMED");
  models[0].tokenstrings[tok + 1] = NULL;
  memcpy(&models[0].userwte[tok - models[0].nummodeltokens], wv, models[0].WVSIZE * sizeof(bloom_precision));
  return tok;
}


int8_t *getwv_q8(long long token, int modelindex)
{
  if (token < 0 || token >= models[modelindex].numtokens)
     return 0;
  if (token < models[modelindex].nummodeltokens)
  {
    long long offset = models[modelindex].WVSIZE * token;
    return models[modelindex].q8_wte + models[modelindex].WVSIZE * token;
  }
  // return userwte+WVSIZE*(token-nummodeltokens);
  return models[modelindex].q8_wte + models[modelindex].WVSIZE * (token - models[modelindex].nummodeltokens);
}

wte_t *getwv(long long token, int modelindex)
{
  if (token < 0 || token >= models[modelindex].numtokens)
    return models[modelindex].sos;
  if (token < models[modelindex].nummodeltokens)
  {
    long long offset = models[modelindex].WVSIZE * token;
    return models[modelindex].wte + models[modelindex].WVSIZE * token;
  }
  // return userwte+WVSIZE*(token-nummodeltokens);
  return models[modelindex].wte + models[modelindex].WVSIZE * (token - models[modelindex].nummodeltokens);
}

wte_t *getwv_final(long long token, int modelindex)
{
  int WVSIZE = models[modelindex].WVSIZE;
  if (!models[modelindex].wtet)
    return getwv(token, modelindex);
  if (token < 0 || token >= models[modelindex].numtokens)
    return models[modelindex].sos;
  if (token < models[modelindex].nummodeltokens)
    return models[modelindex].wtet + WVSIZE * token;
  // return userwte+WVSIZE*(token-nummodeltokens);
  return models[modelindex].wte + WVSIZE * (token - models[modelindex].nummodeltokens);
}

void nametoken(int tok, char *name)
{
  if (tok >= models[0].nummodeltokens)
    free(models[0].tokenstrings[tok]);
  // ^ slight memory leak here if renaming pre-renamed model token.
  // should rather disable the modeltoken and replace it with usertoken
  models[0].tokenstrings[tok] = strdup(name);
}

// int strmatchlgt(char*s0,char*s1)
// {
//   int i=0;
//   while(*s0 && *s1)
//   {
//     if(!*s0) return i;
//     if(*s0!=*s1) return 0;
//     s0++;
//     s1++;
//     i++;
//   }
//   if(!*s0) return i;
//   return 0;
// }

// int tokenize(char*src) // slow! (but not too slow)
// {
//   int i;
//   int best=0,where=-1;

//   for(i=0;;i++)
//   {
//     if(!tokenstrings[i]) break;
//     int matchlgt=strmatchlgt(tokenstrings[i],src);
//     if(matchlgt>best)
//     {
//       best=matchlgt;
//       where=i;
//     }
//   }
//   //fprintf(stderr,"%s bestmatch %d lgt %d\n",src,where,best);
//   return where;
// }

int count_single_byte_chars(char *str) {
    int count = 0;
    unsigned char *p = (unsigned char *)str;
    while (*p) {
        if ((*p & 0x80) == 0x00) {
            count++;
            p++;
        } else if ((*p & 0xE0) == 0xC0) {
            p += 2;
        } else if ((*p & 0xF0) == 0xE0) {
            p += 3;
        } else if ((*p & 0xF8) == 0xF0) {
            p += 4;
        } else {
            // invalid utf-8 sequence
            break;
        }
    }
    return count;
}

int strmatchlgt(char *s1, char *s2)
{
  int matchlgt = 0;
  while (*s1 && *s2)
  {
    if (*s1 == *s2)
    {
      matchlgt++;
    }
    else
    {
      matchlgt = 0;
      break;
    }
    if (*s1 == -60 && *(s1 + 1) == -96 && *(s1 + 2) == '\0')
    {
      matchlgt--;
      return 0;
    } 
    s1++;
    s2++;
  }

  return matchlgt;
}

int get_next_word_length(char *word)
{
  int len = strlen(word);
  int count = 0;

  for (int i=0;i<len;i++)
  {
    if (*(word+i) == '\0' || (*(word+i)==-60 && *(word+i+1)==-96) || *(word+i) == ',' || *(word+i) == '.' || *(word+i) == '!' || *(word+i) == '?')
    {
      return count;

    }
    count++;
  }
  return count;
}

int tokenize(char *src, int modelnum) // slow! (but not too slow)
{
  int max_length=5;
  int i;
  int best = -1, where = -1;
  
  // first try to find a perfect match
  for (i = 0;; i++)
  {
    if (!models[modelnum].tokenstrings[i])
      break;
    int matchlgt = strmatchlgt(models[modelnum].tokenstrings[i], src);
    int token_len = strlen(models[modelnum].tokenstrings[i]);
    
    if (matchlgt > 0 && (*(src+token_len) == '\0' || (*(src+token_len)==-60 && *(src+token_len+1)==-96) || *(src+token_len) == ',' || *(src+token_len) == '.' || *(src+token_len) == '!' || *(src+token_len) == '?') )
    {
      best = matchlgt;
      where = i;      
      return where;
    }
  }
  // if no perfect match, try to tokenize the word into multiple sub tokens in the fewest number of tokens, 
  // preferring tokens less than 5 characters long
  int numbesttokens=0;
  int besttokens[4096] = {-1};
  
  
  
  for (int k=3;k<16;k++)
  {
    max_length = k;
    int currenttokens[4096] = {-1};

    int offset = 0;
    char *src_temp = src;
    int worddone = 0;
    int j=0;
    while (worddone == 0)
    {
      best = -1;
      where = -1;
      for (i = 0;; i++)
      {
        if (!models[modelnum].tokenstrings[i])
          break;
        int matchlgt = strmatchlgt(models[modelnum].tokenstrings[i], src_temp);
        int token_len = strlen(models[modelnum].tokenstrings[i]);
        int next_word_length = 0;

        if (matchlgt > 0 && (*(src_temp+token_len) == '\0' || (*(src_temp+token_len)==-60 && *(src_temp+token_len+1)==-96) || *(src_temp+token_len) == ',' || *(src_temp+token_len) == '.' || *(src_temp+token_len) == '!' || *(src_temp+token_len) == '?') )
        {
          best = matchlgt;
          where = i;      
          worddone = 1;
          currenttokens[j] = i;
          break;
        }

        if ((best == -1 && matchlgt > 0 && matchlgt > best) || (matchlgt > 0 && matchlgt > best && (strlen(models[modelnum].tokenstrings[i])<max_length)))
        {
          best = matchlgt;
          where = i;
          currenttokens[j] = i;
        }
      }
      if (where == -1)
      {
        break;
      }
      
      src_temp+=strlen(models[modelnum].tokenstrings[where]);
      j++;
      if (worddone == 1)
      {
        if (numbesttokens==0 || j<numbesttokens)
        {
          memcpy(besttokens, currenttokens, sizeof(int)*4096);
          numbesttokens = j;
        }        
      }
    }
  }  

  if (numbesttokens != 0)
  {
    return besttokens[0];
  }
  best = -1;
  where = -1;
  // fall back to match best length
  for (i = 0;; i++)
  {
    if (!models[modelnum].tokenstrings[i])
      break;
    int matchlgt = strmatchlgt(models[modelnum].tokenstrings[i], src);
    int token_len = strlen(models[modelnum].tokenstrings[i]);
    
    if (matchlgt > 0  && matchlgt > best)
    {
      best = matchlgt;
      where = i;      
      return where;
    }
  }  
  return where;
}

int tokenize_to_context(char *src, int idx, int modelindex, int queryindex)
{
  int CTXSIZE = models[modelindex].CTXSIZE;
  char error_message_buffer[1024];
  regex_t pre_tokenize_regex;
  int return_value_of_regcomp = regcomp(&pre_tokenize_regex, " ?[^(\\s|[\\.,!?…。，、।۔،])]+", REG_EXTENDED);

  regmatch_t match;
  while (regexec(&pre_tokenize_regex, src, 1, &match, 0) == 0)
  {
    src[match.rm_so] = ' ';
    for (int i = match.rm_so + 1; i < match.rm_eo; i++)
    {
      src[i] = '\0';
    }
  }
  regfree(&pre_tokenize_regex);

  while (*src && idx < CTXSIZE)
  {
    int token = tokenize(src, modelindex);
    //printf ("%d ", token);
    if (token < 0)
    {
      //fflush(stdout);
      return idx;
    }
    queries[queryindex].context[idx] = token;
    src += strlen(models[modelindex].tokenstrings[token]);
    idx++;
  }

  //fflush(stdout);
  return idx;
}

void dumpwvstats(bloom_precision *wv)
{
  int WVSIZE = models[0].WVSIZE;
  bloom_precision mean = 0;
  bloom_precision max = 0;
  bloom_precision min = 0;
  int i;
  for (i = 0; i < WVSIZE; i++)
  {
    bloom_precision a = wv[i];
    mean += a;
    if (a > max)
      max = a;
    if (a < min)
      min = a;
    fprintf(stderr, "%f ", a);
  }
  mean /= WVSIZE;
  fprintf(stderr, "\nmean %f min %f max %f\n", mean, min, max);
}

#if (0)
#define CRUNCHSZ 256
void crunchvector(bloom_precision *o, bloom_precision *v, int lgt)
{
  int i, j;
  for (i = 0; i < CRUNCHSZ; i++)
  {
    bloom_precision a = 0;
    for (j = i; j < lgt; j += CRUNCHSZ)
      a += v[j];
    o[i] = a;
  }
}
#endif

int(matchToTokens_cmp)(const void *a, const void *b)
{
  return (((match_t *)b)->prob < ((match_t *)a)->prob) ? -1 : 1;
}

/*** matches a word vector against the token dictionary ***/

// optimization todo: wte_min (uses vec16) jolla top-80 tms sortattavaksi
void matchToTokens(bloom_precision *wv, match_t *o, int num, bloom_precision temp, int modelindex) // outputs tuples of (dist,token)
{
  long long i, j;
  //  fprintf(stderr,"numtokens=%d\n",numtokens);
  match_t *t = malloc(sizeof(match_t) * models[modelindex].numtokens);

#ifdef USE_PKD_WTE
  int32_t wv32[WVSIZE];
  for (i = 0; i < WVSIZE; i++)
  {
    int64_t a = wv[i] * models[modelindex].quanter_wte; // safe
    wv32[i] = a;
    // if(a<-128)a=-128;
    // if(a>127)a=127;
    // wv8[i]=a;
  }
#endif

  // "top_s" phase
  for (i = 0, j = 0; j < models[modelindex].numtokens; j++)
  {
    if (models[modelindex].tokenflags[j] >= 0)
    {
      wte_t *compwv = getwv_final(j, modelindex);
#ifdef Q8MODE_OUTWTE
      int cossim = conv1dline_ii(0, wv8, wte8 + j * WVSIZE, WVSIZE);
#else
#ifdef USE_PKD_WTE
      int64_t cossim = conv1dline_pkdwte(0, wv32, compwv, WVSIZE);
#else
      bloom_precision cossim = conv1dline(0, wv, compwv, models[modelindex].WVSIZE);
#endif
#endif
      t[i].prob = cossim / (temp * models[modelindex].quanter_wte * models[modelindex].quanter_wte);
      t[i].tok = j;
      i++;
    }
    // t[j*2+1]=j; // *(((int*)&t[j*2+1]))=
  }
  //  for(i=0;i<512;i++) fprintf(stderr,"%f ",t[i*2]);
  //  fprintf(stderr,"\n");

  qsort(t, i, sizeof(match_t), matchToTokens_cmp);

#if (0)
  if (targetwv)
  {
    for (i = 0; i < num; i++)
    {
      bloom_precision cossim = conv1dline(0, targetwv, wte + ((int)t[i].tok) * WVSIZE, WVSIZE);
      t[i].prob += cossim / (temp * models[0].quanter_wte * models[0].quanter_wte);
    }
    qsort(t, i, sizeof(match_t), matchToTokens_cmp);
  }
#endif

  //  for(i=0;i<num;i++)
  //  {
  //    if(tokenflags[(int)(t[i*2+1])]>0) t[i*2]=(t[i*2]*3+t[0])/4.0;
  //  }

  // softmax
  bloom_precision max = t[0].prob;
  for (i = 1; i < num; i++)
    if (t[i].prob > max)
      max = t[i].prob;
  bloom_precision sum = 0;
  for (i = 0; i < num; i++)
  {
    bloom_precision a = exp(t[i].prob - max);
    t[i].prob = a;
    sum += a;
  }
  bloom_precision sumr = 1.0 / sum;
  for (i = 0; i < num; i++)
  {
    o[i].prob = t[i].prob * sumr;
    o[i].tok = t[i].tok;
  }
  free(t);
}

/* picks a random match from the match list.
 * tokenflags[] allows for some token-specific options.
 */

int pickmatch_(match_t *list, int sz, bloom_precision minp, int modelindex)
{
  int i;
  bloom_precision a = frand();
  if (list[0].prob < minp || list[0].prob > 0.98)
    return 0;

  if (list[0].prob < 0.75 && (rand() & 1))
    for (i = 0; i < sz; i++)
    {
      int t = list[i].tok;
      if (models[modelindex].tokenflags[t] == 1 && list[i].prob > 0.002)
      {
        models[modelindex].tokenflags[t] = 0;
        // fprintf(stderr,"<>");
        return i;
      }
    }

  for (i = 0; i < sz; i++)
  {
    bloom_precision p = list[i].prob;
    if (p < minp)
    {
      i = 0;
      p = list[i].prob;
    }
    a -= p;
    if (a <= 0)
      return i;
  }
  return 0;
}

int replacetoken(int t, int modelnum)
{
  if (models[modelnum].tokenflags[t] == 4 || models[modelnum].tokenflags[t] == 5)
  {
    if (models[modelnum].tokenflags[t] == 4)
      models[modelnum].tokenflags[t] = 0;
    t = models[modelnum].tokenrepls[t];
    fprintf(stderr, "<R>");
  }
  return t;
}

int pickmatch(match_t *list, int sz, bloom_precision minp, bool allowspecial, int modelindex)
{
  int t = 0;
  int i = 0;

  while (t < 4)
  {
    i = pickmatch_(list, sz, minp, modelindex);
    t = list[i].tok;
    if (t < 4)
    {
      if (allowspecial)
      {
        return i;
      }
      else
      {
        i++;
        t = list[i].tok;
      }
    }
  }
  return i;
}

