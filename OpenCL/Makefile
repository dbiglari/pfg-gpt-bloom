all:
	gcc -g -o normalize_cl ./normalize_cl.c -lOpenCL
	gcc -g -o dot_cl ./dot_cl.c -lOpenCL
	gcc -g -o layer_cl ./layer_cl.c -lOpenCL -lm -DBUILD_TEST
	gcc -g -o hello ./hello.c -lOpenCL
clean:
	rm normalize_cl hello dot_cl layer_cl
