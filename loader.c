#include "common.h"

/* file management functions */

void *readfile(char *fn, int *lgt_ret, char *path)
{
  char filename[2048];
  if (path != NULL)
    sprintf(filename, "%s/%s", path, fn);
  else
    strcpy(filename, fn);
  FILE *file = fopen(filename, "r");
  if (file == NULL)
  {
    fprintf(stderr, "Expected file \"%s\" not found", path);
    return NULL;
  }
  if (lgt_ret)
    *lgt_ret = 0;

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

  size_t readbytes = fread(buffer, 1, len, file);
  if (readbytes != len)
  {
    // something weird happened
    fprintf(stderr, "readbytes != len in %s\n", filename);
  }
  buffer[len] = '\0';
  fclose(file);

  if (lgt_ret)
    *lgt_ret = len;
  return (void *)buffer;
}

// void*readfile(char*fn,int*lgt_ret,char*path)
// {
//   void*buf=NULL;
//   int i=0;
//   FILE*f;
//   char fnb[120];
//   char*pathfn;

//   FILE *file = fopen(fn, "r");
//   if (file == NULL) {
//     fprintf(stderr, "Expected file \"%s\" not found", path);
//     return NULL;
//   }
//   fseek(file, 0, SEEK_END);
//   long len = ftell(file);
//   fseek(file, 0, SEEK_SET);
//   buf = malloc(len + 1);

//   if(lgt_ret) *lgt_ret=0;

//   f=fopen(pathfn,"rb");
//   if(!f)
//   {
//     fprintf(stderr,"file not found: %s\n",pathfn);
//     return NULL;
//   }
//   while(!feof(f))
//   {
//     buf=realloc(buf,i+1024);
//     if(!buf)
//     {
//       fprintf(stderr,"memory allocation failed when fetching file %s!\n",pathfn);
//       exit(1);
//     }
//     i+=fread(buf+i,1,1024,f);
//   }
//   fclose(f);
//   fprintf(stderr,"fetched file %s, lgt=%d\n",pathfn,i);
//   if(lgt_ret) *lgt_ret=i;
//   return buf;
// }

char *zeroterminate(char *s, int sz)
{
  s = realloc(s, sz + 1);
  s[sz] = '\0';
  return s;
}

char *readtextfile(char *fn, char *path)
{
  int sz;
  char *s = readfile(fn, &sz, path);
  return zeroterminate(s, sz);
}

void *readfile_mmap(char *fn, int *lgt_ret)
{
#ifdef HAVE_MMAP
  int sz;
  struct stat st;
  int fd = open(fn, O_RDONLY, 0);
  if (fd < 0)
    return NULL;
  stat(fn, &st);
  sz = st.st_size;
  if (lgt_ret)
    *lgt_ret = sz;
  fprintf(stderr, "mmap()\n");
  return mmap(NULL, sz, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
#else
  return readfile(fn, lgt_ret, NULL);
#endif
}


int loadpackedmodel(char *path)
{
  // int sz,i;
  // int paramsz,wteparamsz;
  // int paramsz2=sizeof(bloom_precision);
  // void*m=readfile_mmap(path,&sz); //readfile(path,&sz,NULL); // TODO use mmap!
  // if(!m) return 1;
  // fprintf(stderr,"file opened, size=%d\n",sz);

  // h=(header_t*)m; m+=256;
  // if(!(h->fileformat[0]=='V' && h->fileformat[1]=='Z' &&
  //      h->fileformat[2]=='G'))
  // {
  //   return 1;
  // }
  // if(WVSIZE!=h->wvsize)
  // {
  //   fprintf(stderr,"WVSIZE mismatch! file %d, exe %d\n",h->wvsize,WVSIZE);
  //   exit(1);
  // }
  // if(NUMLAYERS!=h->numlayers)
  // {
  //   fprintf(stderr,"NUMLAYERS mismatch! file %d, exe %d\n",h->numlayers,NUMLAYERS);
  //   exit(1);
  // }
  // if(NUMHEADS!=h->numheads)
  // {
  //   fprintf(stderr,"NUMHEADS mismatch! file %d, exe %d\n",h->numlayers,NUMLAYERS);
  //   exit(1);
  // }
  // numtokens=h->numtokens;
  // fprintf(stderr,"numtokens=%d\n",numtokens);
  // if(CTXSIZE!=h->ctxsize)
  // {
  //   fprintf(stderr,"CTXSIZE mismatch! file %d, exe %d\n",h->ctxsize,CTXSIZE);
  //   exit(1);
  // }
  // if(HEADSIZE!=h->headsize)
  // {
  //   fprintf(stderr,"HEADSIZE mismatch! file %d, exe %d\n",h->headsize,HEADSIZE);
  //   exit(1);
  // }
  // if(h->paramformat!=PFMT_BF16)
  // {
  //   fprintf(stderr,"paramformat must be PFMT_BF16!\n");
  //   exit(1);
  // }
  // paramsz=2;
  // if(h->wteformat!=PFMT_INT16)
  // {
  //   fprintf(stderr,"wteformat must be PFMT_INT16!\n");
  //   exit(1);
  // }
  // wteparamsz=2;
  // models[0].quanter_wte=h->models[0].quanter_wte;
  // //fprintf(stderr,"models[0].quanter_wte=%f\n",models[0].quanter_wte);

  // wte=m; m+=wteparamsz*numtokens*WVSIZE;
  // wpe=m; m+=paramsz*CTXSIZE*WVSIZE;
  // lnf_g=(pkdflt*)m; m+=paramsz2*WVSIZE;
  // if(h->flags&FLAG_HAVE_BASES)
  // {
  //   //fprintf(stderr,"we have bases\n");
  //   lnf_b=m; m+=paramsz2*WVSIZE;
  // }
  // if(h->flags&FLAG_HAVE_WTET)
  // {
  //   fprintf(stderr,"todo implement wtet\n");
  //   exit(1);
  // }
  // if(h->flags&FLAG_HAVE_SOS)
  // {
  //   fprintf(stderr,"todo implement sos\n");
  //   exit(1);
  // }
  // layers=malloc(sizeof(hlayer)*NUMLAYERS);
  // for(i=0;i<NUMLAYERS;i++)
  // {
  //   //fprintf(stderr,"layer %d starts at %d\n",i,m-(void*)h);
  //   layers[i].ln1_g=m; m+=paramsz2*WVSIZE;
  //   layers[i].ln2_g=m; m+=paramsz2*WVSIZE;
  //   layers[i].mlp_cfc_w=m; m+=paramsz*WVSIZE*WVSIZE*4;
  //   layers[i].mlp_cproj_w=m; m+=paramsz*WVSIZE*WVSIZE*4;
  //   layers[i].attn_cattn_w=m; m+=paramsz*WVSIZE*3*WVSIZE;
  //   layers[i].attn_cproj_w=m; m+=paramsz*WVSIZE*WVSIZE;
  //   if(h->flags&FLAG_HAVE_BASES)
  //   {
  //     layers[i].ln1_b=m; m+=paramsz2*WVSIZE;
  //     layers[i].ln2_b=m; m+=paramsz2*WVSIZE;
  //     layers[i].mlp_cfc_b=m; m+=paramsz2*WVSIZE*4;
  //     layers[i].mlp_cproj_b=m; m+=paramsz2*WVSIZE;
  //     layers[i].attn_cattn_b=m; m+=paramsz2*WVSIZE*3;
  //     layers[i].attn_cproj_b=m; m+=paramsz2*WVSIZE;
  //   }
  //   layers[i].k=malloc(CTXSIZE*WVSIZE*sizeof(bloom_precision));
  //   layers[i].v=malloc(CTXSIZE*WVSIZE*sizeof(bloom_precision));
  // }
  // if(h->flags&FLAG_HAVE_PALETTE)
  // {
  //   fprintf(stderr,"todo implement palette\n");
  //   exit(1);
  // }
  // if(h->flags&FLAG_HAVE_TOKENSTRINGS)
  //   loadtokens_from_tokendata(m,numtokens);
  // fprintf(stderr,"packed model loaded (size %d + tokendata)\n",m-(void*)h);
  return 0;
}

int savepackedmodel(char *fn, int modelindex)
{
  int i;
  header_t h;
  int flags = 0;
  int paramsz, wteparamsz;
  int paramsz2 = sizeof(bloom_precision);
  FILE *f = fopen(fn, "wb");
  if (!f)
    return 1;
  fprintf(stderr, "savepackedmodel: %s\n", fn);
  if (models[modelindex].wtet)
    flags |= FLAG_HAVE_WTET;
  if (models[modelindex].sos)
    flags |= FLAG_HAVE_SOS;
  if (!(flags & (FLAG_HAVE_WTET | FLAG_HAVE_SOS)))
    flags |= FLAG_HAVE_BASES | FLAG_HAVE_TOKENSTRINGS;
  else
  {
    fprintf(stderr, "looks like igpt\n");
    flags |= FLAG_HAVE_PALETTE;
  }
  h.fileformat[0] = 'V';
  h.fileformat[1] = 'Z';
  h.fileformat[2] = 'G';
  h.fileformat[3] = '0';
  h.wvsize = models[modelindex].WVSIZE;
  h.numlayers = models[modelindex].NUMLAYERS;
  h.numheads = models[modelindex].NUMHEADS;
  h.numtokens = models[modelindex].numtokens;
  h.ctxsize = models[modelindex].CTXSIZE;
  h.headsize = models[modelindex].HEADSIZE;
  h.flags = flags;
  h.paramformat = PFMT_BF16;
  paramsz = sizeof(pkdflt);
  h.wteformat = PFMT_INT16;
  wteparamsz = sizeof(wte_t);
  h.reserved0 = 0;
  h.quanter_wte = models[modelindex].quanter_wte;
  fprintf(stderr, "models[modelindex].quanter_wte=%f\n", models[modelindex].quanter_wte);
  fwrite(&h, sizeof(header_t), 1, f);
  for (i = sizeof(header_t); i < 256; i++)
    fputc(0, f);
  fwrite(models[modelindex].wte, wteparamsz * models[modelindex].numtokens * models[modelindex].WVSIZE, 1, f);
  fwrite(models[modelindex].wpe, paramsz * models[modelindex].CTXSIZE * models[modelindex].WVSIZE, 1, f);
  fwrite(models[modelindex].lnf_g, paramsz2 * models[modelindex].WVSIZE, 1, f);
  if (flags & FLAG_HAVE_BASES)
    fwrite(models[modelindex].lnf_b, paramsz2 * models[modelindex].WVSIZE, 1, f);
  if (flags & FLAG_HAVE_WTET)
    fwrite(models[modelindex].wtet, wteparamsz * models[modelindex].numtokens * models[modelindex].WVSIZE, 1, f);
  if (flags & FLAG_HAVE_SOS)
    fwrite(models[modelindex].sos, paramsz2 * models[modelindex].WVSIZE, 1, f);
  for (i = 0; i < models[modelindex].NUMLAYERS; i++)
  {
    fprintf(stderr, "layer %d starts at %ld\n", i, ftell(f));
    fwrite(models[modelindex].layers[i].ln1_g, paramsz2 * models[modelindex].WVSIZE, 1, f);
    fwrite(models[modelindex].layers[i].ln2_g, paramsz2 * models[modelindex].WVSIZE, 1, f);
    fwrite(models[modelindex].layers[i].mlp_cfc_w, paramsz * models[modelindex].WVSIZE * models[modelindex].WVSIZE * 4, 1, f);
    fwrite(models[modelindex].layers[i].mlp_cproj_w, paramsz * models[modelindex].WVSIZE * models[modelindex].WVSIZE * 4, 1, f);
    fwrite(models[modelindex].layers[i].attn_cattn_w, paramsz * models[modelindex].WVSIZE * 3 * models[modelindex].WVSIZE, 1, f);
    fwrite(models[modelindex].layers[i].attn_cproj_w, paramsz * models[modelindex].WVSIZE * models[modelindex].WVSIZE, 1, f);
    if (flags & FLAG_HAVE_BASES)
    {
      fwrite(models[modelindex].layers[i].ln1_b, paramsz2 * models[modelindex].WVSIZE, 1, f);
      fwrite(models[modelindex].layers[i].ln2_b, paramsz2 * models[modelindex].WVSIZE, 1, f);
      fwrite(models[modelindex].layers[i].mlp_cfc_b, paramsz2 * models[modelindex].WVSIZE * 4, 1, f);
      fwrite(models[modelindex].layers[i].mlp_cproj_b, paramsz2 * models[modelindex].WVSIZE, 1, f);
      fwrite(models[modelindex].layers[i].attn_cattn_b, paramsz2 * models[modelindex].WVSIZE * 3, 1, f); //!
      fwrite(models[modelindex].layers[i].attn_cproj_b, paramsz2 * models[modelindex].WVSIZE, 1, f);
    }
  }
  if (flags & FLAG_HAVE_PALETTE)
    fwrite(palette, models[modelindex].numtokens * 3 * sizeof(bloom_precision), 1, f);
  fprintf(stderr, "tokens start at %ld\n", ftell(f));
  if (flags & FLAG_HAVE_TOKENSTRINGS)
    for (i = 0; i < models[modelindex].numtokens; i++)
      fwrite(models[modelindex].tokenstrings[i], sizeof(char) * (strlen(models[modelindex].tokenstrings[i]) + 1), 1, f);
  fclose(f);
  fprintf(stderr, "packed file written to %s!\n", fn);
  return 0;
}

/* we may need to transpose some matrices when loading from raw dumps */

bloom_precision *transpose(bloom_precision *m, int w, int h)
{
  int i, j;
  bloom_precision *o = malloc(sizeof(bloom_precision) * w * h);
  for (i = 0; i < h; i++)
    for (j = 0; j < w; j++)
      o[j * h + i] = m[i * w + j];
  free(m);
  return o;
}

/* we also pack some matrices into more compact bloom_precision formats */

pkdflt *packtensor(bloom_precision *s, int lgt)
{
  int i;
  pkdflt *o = malloc(lgt * sizeof(pkdflt));
  // for (i = 0; i < lgt; i++)
  //   o[i] = PKFLT(s[i]);
  // free(s);
  return o;
}

/* here we load the model from separate raw files */

void importlayerdata(char *path)
{
  int i;
  int sz;
  char is_igpt = 0;
#ifdef QUANTIZE
  models[0].quanter_wte = 1.0;
  quanter_wpe = 1.0;
#endif
  models[0].s_lnf_g = readfile("lnf_g.raw", &sz, path);
  int dsz = sz / sizeof(float);
  models[0].lnf_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
  // copy values
  for (int i = 0; i < dsz; i++)
  {
    models[0].lnf_g[i] = models[0].s_lnf_g[i];
  }

  if (!models[0].lnf_g)
  {
    fprintf(stderr, "check if the directory is valid!\n");
    exit(1);
  }
#ifdef CONSTS_AS_VARS
  // models[0].WVSIZE=1024;//sz/sizeof(bloom_precision);
  // fprintf(stderr,"wordvector length: %d\n",models[0].WVSIZE);
  // models[0].NUMHEADS=16;
  // models[0].CTXSIZE=2048;
  // models[0].HEADSIZE=(models[0].WVSIZE/models[0].NUMHEADS);
  // models[0].RSQRT_HEADSIZE=(1/sqrt(models[0].HEADSIZE));
#else
  if (sz != WVSIZE * sizeof(float))
  {
    fprintf(stderr, "lnf_g doesn't match hardcoded WVSIZE=%d : %d!\n", WVSIZE, sz / sizeof(bloom_precision));
    exit(1);
  }
#endif

  int WVSIZE = models[0].WVSIZE;
  int CTXSIZE = models[0].CTXSIZE;
  int closest_power_of_2 = models[0].closest_power_of_2;
  int HEADSIZE = models[0].HEADSIZE;
  int NUMHEADS = models[0].NUMHEADS;
  int NUMLAYERS = models[0].NUMLAYERS;

  models[0].s_lnf_b = readfile("lnf_b.raw", &sz, path);
  dsz = sz / sizeof(float);
  models[0].lnf_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
  // copy values
  for (int i = 0; i < dsz; i++)
  {
    models[0].lnf_b[i] = models[0].s_lnf_b[i];
  }

  models[0].s_wte = readfile("wte.raw", &sz, path);
  dsz = sz / sizeof(float);
  models[0].wte = (wte_t *)malloc(sizeof(wte_t) * dsz);
  // copy values
  for (int i = 0; i < dsz; i++)
  {
    models[0].wte[i] = models[0].s_wte[i];
  }

  models[0].numwtetokens = sz / (WVSIZE * sizeof(float));
  fprintf(stderr, "wte size: %d wordvecs\n", models[0].numwtetokens);
  if (models[0].numtokens != models[0].numwtetokens)
  {
    // not an error
    // fprintf(stderr,"mismatch with vocabulary size %d!\n",models[0].numtokens);
    if (models[0].numtokens == 0)
    {
      models[0].numtokens = models[0].numwtetokens;
      fprintf(stderr, "using %d\n", models[0].numwtetokens);
    }
  }

  // wte = transpose(wte,WVSIZE,numtokens);
  int q = 0;
  q++;
#ifdef BLOOM
  int wesz = WVSIZE * sizeof(float);
  models[0].s_welw = readfile("welw.raw", &wesz, path);
  int dwesz = wesz / sizeof(float);
  models[0].welw = (pkdflt *)malloc(sizeof(bloom_precision) * dwesz);
  // copy values
  for (int i = 0; i < dwesz; i++)
  {
    models[0].welw[i] = models[0].s_welw[i];
  }

  if (wesz != WVSIZE * sizeof(float))
  {
    fprintf(stderr, "welw size mismatch!\n");
    exit(1);
  }

  models[0].s_welb = readfile("welb.raw", &wesz, path);
  dwesz = wesz / sizeof(float);
  models[0].welb = (pkdflt *)malloc(sizeof(bloom_precision) * dwesz);
  // copy values
  for (int i = 0; i < dwesz; i++)
  {
    models[0].welb[i] = models[0].s_welb[i];
  }

  if (wesz != WVSIZE * sizeof(float))
  {
    fprintf(stderr, "welb size mismatch!\n");
    exit(1);
  }
#else
  wpe = (pkdflt *)readfile("wpe.raw", &sz, path);
  if (sz != CTXSIZE * WVSIZE * sizeof(float))
  {
    fprintf(stderr, "wpe size mismatch!\n");
    exit(1);
  }
#endif
  if (is_igpt == 1)
  {
    models[0].wtet = (pkdflt *)readfile("wtet.raw", &sz, path); // igpt-only
    models[0].sos = (pkdflt *)readfile("sos.raw", &sz, path);   // igpt-only
  }
#ifdef BLOOM
#else

#ifdef USE_PKDFLT
  wpe = packtensor((bloom_precision *)wpe, CTXSIZE * WVSIZE);
#endif
#endif

  /* bbloom_precision16 causes regression in wte, so we use 16-bit ints there */
#ifdef USE_PKD_WTE
  {
    bloom_precision max = 0;
    bloom_precision avg = 0;
    for (i = 0; i < numtokens * WVSIZE; i++)
    {
      bloom_precision a = fabs(((bloom_precision *)wte)[i]);
      // avg+=a;
      if (a > max)
        max = a;
      // uint32_t a=((uint32_t*)wte)[i];
      // a&=0xfffffe00;
      //((uint32_t*)wte)[i]=a;
    }
    // avg/=(numtokens*WVSIZE);
    // fprintf(stderr,"wte max %f avg %f\n",max,avg);
    models[0].quanter_wte = 32767.5 / max;
    fprintf(stderr, "models[0].quanter_wte=%f\n", models[0].quanter_wte);
    wte_t *qwte = malloc(numtokens * WVSIZE * sizeof(wte_t));
    for (i = 0; i < numtokens * WVSIZE; i++)
    {
      int a = floor(((bloom_precision *)wte)[i] * models[0].quanter_wte);
      qwte[i] = a;
    }
    free(wte);
    wte = qwte;
  }
#endif

  for (i = 0; i < MAXNUMLAYERS; i++)
  {
    char fn[80];

    models[0].layers = realloc(models[0].layers, sizeof(hlayer) * (i + 1));
    if (!models[0].layers)
    {
      fprintf(stderr, "memory allocation error at layer %d!\n", i);
      exit(1);
    }

    sprintf(fn, "h%d_ln1_g.raw", i);
    models[0].layers[i].s_ln1_g = readfile(fn, &sz, path);
    if (!models[0].layers[i].s_ln1_g)
      break;
    dsz = sz / sizeof(float);
    models[0].layers[i].ln1_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].ln1_g[j] = models[0].layers[i].s_ln1_g[j];
    }

    sprintf(fn, "h%d_ln1_b.raw", i); // not in igpt
    models[0].layers[i].s_ln1_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].ln1_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].ln1_b[j] = models[0].layers[i].s_ln1_b[j];
    }

    sprintf(fn, "h%d_ln2_g.raw", i);
    models[0].layers[i].s_ln2_g = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].ln2_g = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].ln2_g[j] = models[0].layers[i].s_ln2_g[j];
    }

    sprintf(fn, "h%d_ln2_b.raw", i); // not in igpt
    models[0].layers[i].s_ln2_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].ln2_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].ln2_b[j] = models[0].layers[i].s_ln2_b[j];
    }

    sprintf(fn, "h%d_mlp_cfc_w.raw", i);
    models[0].layers[i].s_mlp_cfc_w = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].mlp_cfc_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].mlp_cfc_w[j] = models[0].layers[i].s_mlp_cfc_w[j];
    }

    sprintf(fn, "h%d_mlp_cfc_b.raw", i); // not in igpt
    models[0].layers[i].s_mlp_cfc_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].mlp_cfc_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].mlp_cfc_b[j] = models[0].layers[i].s_mlp_cfc_b[j];
    }

    sprintf(fn, "h%d_mlp_cproj_w.raw", i);
    models[0].layers[i].s_mlp_cproj_w = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].mlp_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].mlp_cproj_w[j] = models[0].layers[i].s_mlp_cproj_w[j];
    }

    sprintf(fn, "h%d_mlp_cproj_b.raw", i); // not in igpt
    models[0].layers[i].s_mlp_cproj_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].mlp_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].mlp_cproj_b[j] = models[0].layers[i].s_mlp_cproj_b[j];
    }

#ifdef QUANTIZE
    models[0].layers[i].mlp_cfc_w_q = 1.0;
    models[0].layers[i].mlp_cproj_w_q = 1.0;
#endif
    sprintf(fn, "h%d_attn_cproj_w.raw", i);
    models[0].layers[i].s_attn_cproj_w = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].attn_cproj_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].attn_cproj_w[j] = models[0].layers[i].s_attn_cproj_w[j];
    }

    if (!models[0].layers[i].attn_cproj_w)
    {
      // igpt uses different name. size 8x64x512
      sprintf(fn, "h%d_attn_cproj.raw", i);
      models[0].layers[i].attn_cproj_w = (bloom_precision *)readfile(fn, &sz, path);
    }
    if (i == 0)
    {
      int numheads = sz / (WVSIZE * HEADSIZE * sizeof(float));
#ifdef CONSTS_AS_VARS
      models[0].NUMHEADS = numheads;
      fprintf(stderr, "numheads=%d\n", numheads);
#else
      if (NUMHEADS != numheads)
      {
        fprintf(stderr, "number of heads (%d) doesn't match hardcoded %d!\n",
                numheads, NUMHEADS);
      }
#endif
    }
    sprintf(fn, "h%d_attn_cproj_b.raw", i); // not in igpt
    models[0].layers[i].s_attn_cproj_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].attn_cproj_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].attn_cproj_b[j] = models[0].layers[i].s_attn_cproj_b[j];
    }

    sprintf(fn, "h%d_attn_cattn_w.raw", i);
    models[0].layers[i].s_attn_cattn_w = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].attn_cattn_w = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].attn_cattn_w[j] = models[0].layers[i].s_attn_cattn_w[j];
    }

    sprintf(fn, "h%d_attn_cattn_b.raw", i);
    models[0].layers[i].s_attn_cattn_b = readfile(fn, &sz, path);
    dsz = sz / sizeof(float);
    models[0].layers[i].attn_cattn_b = (bloom_precision *)malloc(sizeof(bloom_precision) * dsz);
    // copy values
    for (int j = 0; j < dsz; j++)
    {
      models[0].layers[i].attn_cattn_b[j] = models[0].layers[i].s_attn_cattn_b[j];
    }

    if (!models[0].layers[i].attn_cattn_w)
    {
      models[0].layers[i].attn_cattn_w =
          malloc(sizeof(bloom_precision) * 3 * models[0].NUMHEADS * HEADSIZE * WVSIZE);
      bloom_precision *tmp;
      /* igpt separates cattn_w into 3 files */
      int j;
      for (j = 0; j < 3; j++)
      {
        sprintf(fn, "h%d_attn_%cproj.raw", i, "qkv"[j]);
        tmp = (bloom_precision *)readfile(fn, &sz, path);
        memcpy(models[0].layers[i].attn_cattn_w + j * models[0].NUMHEADS * HEADSIZE * WVSIZE, tmp,
               sizeof(bloom_precision) * models[0].NUMHEADS * HEADSIZE * WVSIZE);
        free(tmp);
      }
      is_igpt = 1;
    }

    /* transpose some of the bigger matrices for speedup. */

    // if(!is_igpt) /* igpt has already transposed this one */
    // {
    //   models[0].layers[i].attn_cattn_w =
    //     transpose(models[0].layers[i].attn_cattn_w,WVSIZE*3,WVSIZE);
    // }
    //   models[0].layers[i].attn_cproj_w =
    //     transpose(models[0].layers[i].attn_cproj_w,WVSIZE,WVSIZE);

    // models[0].layers[i].mlp_cfc_w =
    //   transpose(models[0].layers[i].mlp_cfc_w,WVSIZE*4,WVSIZE);
    // models[0].layers[i].mlp_cproj_w =
    //   transpose(models[0].layers[i].mlp_cproj_w,WVSIZE,WVSIZE*4);

#ifdef USE_PKDFLT
    models[0].layers[i].attn_cattn_w = packtensor((bloom_precision *)models[0].layers[i].attn_cattn_w, WVSIZE * 3 * WVSIZE);
    models[0].layers[i].attn_cproj_w = packtensor((bloom_precision *)models[0].layers[i].attn_cproj_w, WVSIZE * WVSIZE);
    models[0].layers[i].mlp_cfc_w = packtensor((bloom_precision *)models[0].layers[i].mlp_cfc_w, WVSIZE * WVSIZE * 4);
    models[0].layers[i].mlp_cproj_w = packtensor((bloom_precision *)models[0].layers[i].mlp_cproj_w, WVSIZE * WVSIZE * 4);
#endif
    models[0].layers[i].k = malloc(models[0].CTXSIZE * WVSIZE * sizeof(bloom_precision));
    models[0].layers[i].v = malloc(models[0].CTXSIZE * WVSIZE * sizeof(bloom_precision));
  }

  // build alabi tensor

  //     batch_size, seq_length = attention_mask.shape
  // closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
  // base = torch.tensor(
  //     2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
  // )
  // powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
  // slopes = torch.pow(base, powers)

  // if closest_power_of_2 != num_heads:
  //     extra_base = torch.tensor(
  //         2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
  //     )
  //     num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
  //     extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
  //     slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

  queries[0].attentions = malloc(models[0].CTXSIZE * models[0].NUMLAYERS * models[0].NUMHEADS * sizeof(bloom_precision));
  queries[0].attentions_presoftmax = malloc(models[0].CTXSIZE * models[0].NUMLAYERS * models[0].NUMHEADS * sizeof(bloom_precision));
  queries[0].attention_arrange_tensor = malloc(sizeof(bloom_precision) * models[0].CTXSIZE);
  queries[0].attention_mask = malloc(sizeof(bloom_precision) * models[0].CTXSIZE);
  models[0].closest_power_of_2 = pow(2, floor(log2((bloom_precision)models[0].NUMHEADS)));
  models[0].base = pow(2, (-(pow(2, -(log2(models[0].closest_power_of_2) - 3)))));
  models[0].alibi = malloc(sizeof(bloom_precision) * models[0].closest_power_of_2 * models[0].CTXSIZE);
  queries[0].att = malloc(sizeof(bloom_precision) * models[0].closest_power_of_2 * models[0].CTXSIZE);

#ifdef CONSTS_AS_VARS
  models[0].NUMLAYERS = i;
  fprintf(stderr, "number of layers = %d\n", models[0].NUMLAYERS);
#else
  if (i != NUMLAYERS)
  {
    fprintf(stderr, "number of layers (%d) doesn't match hardcoded NUMLAYERS=%d!\n",
            i, NUMLAYERS);
    exit(1);
  }
#endif
}

#if (0)
int isvalidutf8(char *s0)
{
  unsigned char *s = (unsigned char *)s0;
  char pt = 0;
  while (*s)
  {
    char t = 0;
    if (*s < 0x80)
      t = 0;
    else if (*s < 0xc0)
      t = 1;
    else
      t = 2;
    //    printf("- %02x %d %d\n",*s,pt,t);
    if (*s < 0x20 || *s == 0x7f)
      return 0;
    if (pt == 0 && t == 1)
      return 0;
    if (pt == 2 && t != 1)
      return 0;
    pt = t;
    s++;
  }
  if (pt == 2)
    return 0;
  //  printf("OK!\n");
  return 1;
}

void printtoken(char *s)
{
  if (isvalidutf8(s))
  {
    printf("%s", s);
  }
  else
  {
    for (; *s; s++)
    {
      if (*s < 0x20 || *s >= 0x7F)
        printf("<%02X>", (unsigned char)*s);
      else
        putchar(*s);
    }
  }
}
#endif

int isregularfile(char *fn)
{
  int rc;
  struct stat sb;
  rc = lstat(fn, &sb);
  if (rc < 0)
    return 0;
  if (S_ISREG(sb.st_mode))
    return 1;
  else
    return 0;
}

void loadmodel(char *modelpath)
{
  if (isregularfile(modelpath))
  {
    fprintf(stderr, "loading packed model...\n");
    int rc = loadpackedmodel(modelpath);
    if (rc)
    {
      fprintf(stderr, "failed to load packed model\n");
      exit(1);
    }
    models[0].emptytoken = tokenize("<|endoftext|>",0);
    return;
  }

  fprintf(stderr, "load tokens...\n");
  int rc = loadtokens(modelpath);
  if (rc)
  {
    fprintf(stderr, "load palette...\n");
    rc = loadpalette(modelpath);
    models[0].emptytoken = -1;
  }
  else
  {
    models[0].emptytoken = tokenize("<|endoftext|>",0);
  }
  /*
  if(rc)
  {
    rc=loadpackedmodel(modelpath);
    if(rc)
    {
      fprintf(stderr,"check if the model path (`%s') is valid!\n",modelpath);
      exit(1);
    }
    return;
  }
  */

  importlayerdata(modelpath);
  /*
    int i;
    for(i=0;i<numtokens;i++)
    {
      printf("%d. \"",i); printtoken(tokenstrings[i]);
      printf("\"\n");
    }
    exit(0);
  */
}

/* experimental: quantize some matrices into 8-bit integer format */

#ifdef QUANTIZE
bloom_precision statistics(bloom_precision *m, int sz)
{
  int i;
  bloom_precision sum = 0, sumabs = 0, maxabs = 0;
  for (i = 0; i < sz; i++)
  {
    bloom_precision a = m[i];
    sum += a;
    sumabs += fabs(a);
    if (fabs(a) > maxabs)
      maxabs = fabs(a);
  }
  sum /= sz;
  sumabs /= sz;
  bloom_precision muller = 128.0 / maxabs;
  fprintf(stderr, "avg %f avg(abs) %f max(abs) %f. mul by %f -> %f & %f\n\n",
          sum, sumabs, maxabs, muller, sumabs * muller, maxabs * muller);
  return muller;
}

void quantize_matrix_fake(int8_t *m8, bloom_precision *m, bloom_precision muller, int sz)
{
  int i;
  for (i = 0; i < sz; i++)
    m8[i] = m[i] = m[i] * muller;
}

void quantize_matrix(int8_t *m8, bloom_precision *m, bloom_precision muller, int sz)
{
  int i;
  for (i = 0; i < sz; i++)
    m8[i] = m[i] = floor(m[i] * muller);
}

void quantize()
{
  // safe
  bloom_precision muller, muller1;
  fprintf(stderr, "wte: ");
  models[0].quanter_wte = muller = statistics(wte, numtokens * WVSIZE);
  fprintf(stderr, "wpe: ");
  quanter_wpe = muller1 = statistics(wpe, CTXSIZE * WVSIZE);
  if (muller1 < muller)
    muller = muller1;

  wte8 = malloc(numtokens * WVSIZE * sizeof(int8_t));
  wpe8 = malloc(CTXSIZE * WVSIZE * sizeof(int8_t));

  // safe
  int i;
  for (i = 0; i < numtokens * WVSIZE; i++)
    wte8[i] = wte[i] = wte[i] * models[0].quanter_wte;
  for (i = 0; i < CTXSIZE * WVSIZE; i++)
    wpe8[i] = wpe[i] = wpe[i] * quanter_wpe;

  for (i = 0; i < NUMLAYERS; i++)
  {
    layers[i].mlp_cfc8w = malloc(WVSIZE * WVSIZE * 4);
    layers[i].mlp_cproj8w = malloc(WVSIZE * WVSIZE * 4);
    layers[i].mlp_cfc8b = malloc(WVSIZE * 4);
    layers[i].mlp_cproj8b = malloc(WVSIZE);

    // safe
    fprintf(stderr, "%d cfc: ", i);
    muller = statistics(layers[i].mlp_cfc_w, WVSIZE * WVSIZE * 4);
    // fprintf(stderr,"%d cfc_b: ",i);
    // muller1=statistics(layers[i].mlp_cfc8b,WVSIZE*4);
    layers[i].mlp_cfc_w_q = muller; //>muller1?muller:muller1;

    // safe
    fprintf(stderr, "%d cproj: ", i);
    muller = statistics(layers[i].mlp_cproj_w, WVSIZE * WVSIZE * 4);
    // fprintf(stderr,"%d cproj_b: ",i);
    // muller1=statistics(layers[i].mlp_cproj8b,WVSIZE);
    layers[i].mlp_cproj_w_q = muller; //>muller1?muller:muller1;

    // safe
#ifdef Q8MODE_MLP
    quantize_matrix(layers[i].mlp_cfc8w, layers[i].mlp_cfc_w,
                    layers[i].mlp_cfc_w_q, WVSIZE * WVSIZE * 4);
    quantize_matrix(layers[i].mlp_cproj8w, layers[i].mlp_cproj_w,
                    layers[i].mlp_cproj_w_q, WVSIZE * WVSIZE * 4);
#else
    quantize_matrix_fake(layers[i].mlp_cfc8w, layers[i].mlp_cfc_w,
                         layers[i].mlp_cfc_w_q, WVSIZE * WVSIZE * 4);
    quantize_matrix_fake(layers[i].mlp_cproj8w, layers[i].mlp_cproj_w,
                         layers[i].mlp_cproj_w_q, WVSIZE * WVSIZE * 4);
#endif

    // UNSAFE
    //    quantize_matrix(layers[i].mlp_cfc8b,layers[i].mlp_cfc_b,
    //      layers[i].mlp_cfc_w_q,WVSIZE*4);
    //    quantize_matrix(layers[i].mlp_cproj8b,layers[i].mlp_cproj_b,
    //      layers[i].mlp_cproj_w_q,WVSIZE);
  }

  // models[0].quanter_wte=1.0/muller;
}
#endif
