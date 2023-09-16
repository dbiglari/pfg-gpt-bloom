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

Prompts:

The prompts folder contains a few sample prompts that are useful for testing that the code is working. 
For example, the file prompts/promt1.txt contains the following sample text from OpenAI's GPT2 release page
(https://openai.com/research/better-language-models):

In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

Note that bloom was trained on a different data set and has a slightly different algorithm from GPT2, so it will complete the text differently.
To use the prompts/prompt1.txt file with pfg-gpt-bloom, 


To run and get quick output once you've downloaded the bloom-560m model:

./pfg-gpt-bloom -C [CPU/GPU mode] -f [prompt file] -t [temperature] -z [minp] -m [sampling/greedy] -T [Number of CPU threads to use] -F /etc/pfg_gpt_config -M [model name]

For example:

./pfg-gpt-bloom -C 0 -f prompts/prompt1.txt -t 1.0 -z 0.0 -m greedy -T 12 -F /etc/pfg_gpt_config -M bloom-560m

You should see the following output:

```
$ ./pfg-gpt-bloom -C 0 -f prompts/prompt1.txt -t 1.0 -z 0.0 -m greedy -T 12 -F /etc/pfg_gpt_config -M bloom-560m
loading model from /mnt/8c30b056-7182-460b-aa36-e534c6890c4f/data/bloom/models--bigscience--bloom-560m/snapshots/afe2e6f33eb135d254df849c74bb83322b53641c into RAM
initial load complete.
layer data size: 4096
model load complete
initializing query #0
query initialization complete
seed from time(): 1694846169
Prompt length: 46 tokens
----------------query #0 parameters---------------
temperature: 1.000000
temperature_alt: 1.000000
minp: 0.000000
mode: 0
hardmax_gen: -1
grammarmax_gen: -1
paragrammarmax_gen: 0
force_gen_tokens: -2
seed: 1694846169
---------------------------------------------------
----------------model #0 parameters---------------
modelname: bloom-560m
numthreads: 12
WVSIZE: 1024
CTXSIZE: 4096
NUMLAYERS: 24
NUMHEADS: 16
HEADSIZE: 64.000000
RSQRT_HEADSIZE: 0.125000
---------------------------------------------------
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. The researchers found that the unicorns were able to communicate with each other in a way that was impossible to do in the wild. The unicorns were able to communicate with one another in a way that was impossible to accomplish in the wild.
Total Generate Time          - Time measured: 6.237479557 seconds.
Tokens/sec = 7.695416
```












To see a list of command line options:

```
$ ./pfg-gpt-bloom --help

Usage: ./pfg-gpt-bloom [options] [modelpath]
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
```


GPU Mode:

When running in GPU mode, ensure you have a GPU with adequate GPU RAM to support the models you are wanting to run.
If you have a system with multiple GPUs, you can assign specific layers of a model to specific GPUs in the model_layer_device_map file.


Client/Server mode:

The program also supports a client server mode that can be wrapped with a web gui using node js or python.

