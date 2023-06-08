#include <stdlib.h>
#include <signal.h>
void gdb(int sig) {
  system("exec xterm -e gdb -p \"$PPID\"");
  abort();
}

void _init() {
  signal(SIGSEGV, gdb);
}
