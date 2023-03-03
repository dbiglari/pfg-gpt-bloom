CC=gcc
# linenoise-ng requires stdc++, classic linenoise does not
#LUALIBS=-llua5.4 -lreadline
# -lncurses
#-llinenoise -lstdc++
#SDLLIBS=`sdl-config --libs --cflags` -lSDL_image
LIBS=-lm $(SDLLIBS) $(LUALIBS) -ljson-c -Ljson-c -lstdc++ -luuid -Llibb64-1.2/src -lb64

ifeq ($(DEBUG_ENABLE),1)
DEBUG = -g -DDEBUG
OPT2 = -O0
OPT3 = -O0
else
DEBUG = -g
OPT2 = -O3
OPT3 = -O3
endif


#OPT=$(DEBUG) -march=znver1 -mtune=znver1 -mfma -mavx2 -m3dnow -fomit-frame-pointer -fpermissive -fno-math-errno -Izip
#OPT=$(DEBUG) -mfpmath=sse -fpermissive -fno-math-errno -Izip -Ilibb64-1.2/include
OPT=$(DEBUG) -mfpmath=sse -fno-math-errno -Izip -Ilibb64-1.2/include -Werror

all: libb64_build json-c_build pfg-gpt-bloom fp32_to_fp16

pfg-gpt-bloom: main.o loader.o model.o tokens.o glyphgen.o ui_sdl.o ui_tty.o lua.o inlines.o  zip.o raw_loader.o bf16.o fp16.o unpickle.o fastbarrier.o simd.o server.o base64.o fp32to8bit.o client.o
	$(CC) $(OPT) $(OPT2) inlines.o main.o loader.o tokens.o glyphgen.o model.o ui_sdl.o ui_tty.o zip.o raw_loader.o bf16.o fp16.o fastbarrier.o simd.o lua.o unpickle.o server.o base64.o fp32to8bit.o client.o -o pfg-gpt-bloom $(LIBS) -pthread

unpickle.o: unpickle.cpp
	g++ $(OPT) $(OPT2) -c unpickle.cpp

fp16.o: fp16.c
	$(CC) $(OPT) $(OPT2) -c fp16.c

base64.o: base64.c
	$(CC) $(OPT) $(OPT2) -c base64.c

bf16.o: bf16.c
	$(CC) $(OPT) $(OPT2) -c bf16.c

client.o: client.c
	$(CC) $(OPT) $(OPT2) -c client.c



fp32to8bit.o: fp32to8bit.c
	$(CC) $(OPT) $(OPT2) -c fp32to8bit.c


server.o: server.c
	$(CC) $(OPT) $(OPT2) -c server.c

fastbarrier.o: fastbarrier.c
	$(CC) $(OPT) $(OPT2) -c fastbarrier.c

simd.o: simd.c
	$(CC) $(OPT) $(OPT2) -c simd.c

json-c_build:
	cd json-c;cmake . -DBUILD_STATIC_LIBS=on -DBUILD_SHARED_LIBS=off; make clean; make -j 12

libb64_build:
	cd libb64-1.2; make

zip.o: zip/zip.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c zip/zip.c

raw_loader.o: raw_loader.c
	$(CC) $(OPT) $(OPT2) -c raw_loader.c

main.o: main.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c main.c -D__MAIN__

tokens.o: tokens.c common.h config.h
	$(CC) $(OPT) $(OPT3) -funsafe-math-optimizations -c tokens.c

model.o: model.c common.h config.h
	$(CC) $(OPT) $(OPT3) -funsafe-math-optimizations -c model.c

loader.o: loader.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c loader.c

inlines.o: loader.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c inlines.c

ui_sdl.o: ui_sdl.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c ui_sdl.c

ui_tty.o: ui_tty.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c ui_tty.c

glyphgen.o: glyphgen.c common.h config.h
	$(CC) $(OPT) $(OPT3) -c glyphgen.c

lua.o: lua.c common.h config.h
	$(CC) $(OPT) $(OPT2) -c lua.c -I/usr/include/lua5.4

clean:
	rm -f *~ *.o pfg-gpt-bloom DEADJOE
	-@rm fp32_to_fp16
	-@rm client
	-@cd libb64-1.2; make clean


fp32_to_fp16:
	gcc -g -o fp32_to_fp16 fp16.c fp32_to_fp16.c

