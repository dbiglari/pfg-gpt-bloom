#include "common.h"
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define COLOR0 "\033[0m"
#define COLOR1 "\033[1m"

int getch()
{
  struct termios old,curr;
  int c;
  tcgetattr(STDIN_FILENO,&old);
  memcpy(&curr,&old,sizeof(struct termios));
  curr.c_lflag&=~(ECHO|ICANON);
  tcsetattr(STDIN_FILENO,TCSANOW,&curr);
  c = getchar();
  tcsetattr(STDIN_FILENO,TCSANOW,&old);
  return c;
}

int hascontrolchars(char*s)
{
  while(*s)
  {
    if(*s<32) return 1;
    s++;
  }
  return 0;
}

void ttyui_printtail(int endslot)
{

}

void ttyui_edit()
{
 
}

void ttyui_matchlist()
{

}

void ttyui_settings()
{
 
}

void glitchvar(bloom_precision*v)
{
//  *v*=(4+((rand()&65535)/32768.0))/5.0;
  *v*=((rand()&65535)/32768.0);
//  *v*=(((rand()&65535)/32768.0)-0.1); // sittenkin toisenlainen tulos tällä
}

void glitchvar2(bloom_precision*v)
{
  *v=0;//((rand()&65535)/32768.0)-1.0;
//  *v=0-*v;
}

void glitchkv(int l)
{
 
}

void ttyui()
{
 
}
