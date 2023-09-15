# pfg-gpt-bloom (Pig Fly Games GPT Bloom implementation)

required packages (Ubuntu 20.04):

sudo apt-get install git cmake uuid-dev

To build:

From the top level, run "make"

To download the models, go to (note that huggingface will require a login to be created and their git repositories use lfs):

bloom 560m model (smallest)
https://huggingface.co/bigscience/bloom-560m/tree/main

bloom 1b7 model
https://huggingface.co/bigscience/bloom-1b7/tree/13d30c3a831d5f88c99a9dd210521a0ae3ac9f7b

bloom 3b model 
https://huggingface.co/bigscience/bloom-3b/tree/fdeb52a91349a7e2a78781ebf93a75c976b3414e

bloom 7b1 model
https://huggingface.co/bigscience/bloom-7b1/tree/main

bloom 175b model (largest model, use pre-safetensor commit for files)
https://huggingface.co/bigscience/bloom/tree/4ab0472e70b36c99c1845b4eb8fb04317d5ac782

Place the files in a directory and update the pfg_gpt_config file to reference that directory.  The format for
 each row in the pfg_gpt_config file is:

[model name] [model location to config.json file]


To run and get quick output once you've downloaded the bloom-560m model:

./pfg-gpt-bloom -C 0 -f [prompt file] -t 1.0 -z 0.0 -m sampling -T 12 -F /etc/pfg_gpt_config -M [model name]

For example:

./pfg-gpt-bloom -C 0 -f prompts/prompt1.txt -t 1.0 -z 0.0 -m sampling -T 12 -F /etc/pfg_gpt_config -M bloom-560m



To see a list of command line options:

$ ./pfg-gpt-bloom --help
Usage: ./pfg-gpt-bloom <options> [modelpath]
-h          show this help
-d port     start server on port
-f file.txt read prompt from file
-F file.txt pfg_gpt_config file location
-l 512      set maximum length of text to output (in tokens)
-t 1.0      set noise temperature for match randomization
-a 1.0      set alternative temperature used at sentence boundaries etc
-T 4        set number of threads
-m mode     set sampling mode, (greedy/sampling)
-M model    use specified modelname
-s 123456   set random number seed (0 = use timer)
-c server   connect to a server as a client
-p port     use specified port when connecting to server (default 8081 if unspecified)
-n n        number of ngram repeats (-1 = disable)
-N n        stop after ngram repeats (1 = stop, -1 = don't stop)
-S n        stop after end of sequence token (1 = stop, -1 = dont' stop)
-g n        force generate at least n tokens
-x n        hard maximum n tokens
-y n        generate at least n sentences
-Y n        generate at least n paragraphs
-v          verbose/debug output
-z m        set minp value to m (float)
-X n        context size
-C          OpenCL mode (0 = no OpenCL, 1 = hybrid CPU/GPU mode, 2 = full GPU mode, default = 0)
-b <0|1|2>  Display information about OpenCL devices on the system (0=false, 1=true, 2=true, exit after)




Client/Server mode:

The program also supports a client server mode that can be wrapped with a web gui using node js or python.

