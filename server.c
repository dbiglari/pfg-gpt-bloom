#include <stdio.h>
#include <string.h> //strlen
#include <stdlib.h> //strlen
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <unistd.h>    //write
#include <pthread.h>   //for threading , link with lpthread
#include <semaphore.h>
#include <uuid/uuid.h>

#include "common.h"
#include "server.h"
#include "json-c/json.h"
#include "raw_loader.h"
#include "base64.h"

sem_t query_mutex;
extern model_path_t model_definitions[9];

#define MAX_THREADIDS 65535
pthread_t threadIds[MAX_THREADIDS];
int running = 0;

typedef struct bloom_thread_t
{
    int client_sock;
    int threadId_index;

} bloom_thread_t;

#define RECV_BUFFER_SIZE 1048560
#define SEND_BUFFER_SIZE 1048560 * 2

// allocates jsonstring_output, needs to be deallocated by caller
/**
 * @brief  Process prompt through model and send back response from model
 * @note   
 * @param  *jsonstring_input: 
 * @param  **jsonstring_output: 
 * @param  inputsize: 
 * @param  *outputsize: 
 * @retval 
 */
int process(char *jsonstring_input, char **jsonstring_output, int inputsize, int *outputsize)
{
    char *prompt = NULL;
    char *modelname = NULL;
    char *prompt_new = NULL;
    int promptlgt;
    int modelnum = 0;
    int lengthtogen;
    struct json_object *parsed_json = NULL;
    struct json_object *prompt_json = NULL;
    struct json_object *promptb64_json = NULL;
    struct json_object *currwv_json = NULL;
    struct json_object *model_json = NULL;
    struct json_object *temperature_json = NULL;
    struct json_object *temperature_alt_json = NULL;
    struct json_object *nummatches_json = NULL;
    struct json_object *minp_json = NULL;
    struct json_object *seed_json = NULL;
    struct json_object *guid_json = NULL;
    struct json_object *context_json = NULL;
    struct json_object *mode_json = NULL;
    struct json_object *force_gen_tokens_json = NULL;
    struct json_object *hardmax_gen_json = NULL;
    struct json_object *grammarmax_gen_json = NULL;
    struct json_object *paragrammarmax_gen_json = NULL;
    struct json_object *currslot_json = NULL;
    struct json_object *start_json = NULL;
    char *promptguidstr = NULL;
    char *context_string = NULL;
    char *currwv_string = NULL;
    char *promptb64_string = NULL;
    char *currwv = NULL;
    uuid_t response_uuid;
    uuid_t prompt_uuid;
    int start;
    int currslot;
    int genstart;
    int genend;
    int hardmax_gen;
    int grammarmax_gen;
    int paragrammarmax_gen;
    int force_gen_tokens;
    int seed;
    char emptystr[1] = "";

    // run jsonstring_input through json c
    parsed_json = json_tokener_parse(jsonstring_input);

    // grab the list of token
    json_object_object_get_ex(parsed_json, "prompt", &prompt_json);
    json_bool promptb64_ret = json_object_object_get_ex(parsed_json, "promptb64", &promptb64_json);
    json_bool model_ret = json_object_object_get_ex(parsed_json, "model", &model_json);
    json_bool guid_ret = json_object_object_get_ex(parsed_json, "guid", &guid_json);
    json_bool context_ret = json_object_object_get_ex(parsed_json, "context", &context_json);
    json_bool currwv_ret = json_object_object_get_ex(parsed_json, "currwv", &currwv_json);
    json_bool mode_ret = json_object_object_get_ex(parsed_json, "mode", &mode_json);
    json_bool temperature_ret = json_object_object_get_ex(parsed_json, "temperature", &temperature_json);
    json_bool temperature_alt_ret = json_object_object_get_ex(parsed_json, "temperature_alt", &temperature_alt_json);
    json_bool minp_ret = json_object_object_get_ex(parsed_json, "minp", &minp_json);
    json_bool nummatches_ret = json_object_object_get_ex(parsed_json, "nummatches", &nummatches_json);
    json_bool force_gen_tokens_json_ret = json_object_object_get_ex(parsed_json, "force_gen_tokens", &force_gen_tokens_json);
    json_bool hardmax_gen_json_ret = json_object_object_get_ex(parsed_json, "hardmax_gen", &hardmax_gen_json);
    json_bool grammarmax_gen_json_ret = json_object_object_get_ex(parsed_json, "grammarmax_gen", &grammarmax_gen_json);
    json_bool paragrammarmax_gen_json_ret = json_object_object_get_ex(parsed_json, "paragrammarmax_gen", &paragrammarmax_gen_json);
    json_bool seed_json_ret = json_object_object_get_ex(parsed_json, "seed", &seed_json);
    json_bool currslot_json_ret = json_object_object_get_ex(parsed_json, "currslot", &currslot_json);
    json_bool start_json_ret = json_object_object_get_ex(parsed_json, "start", &start_json);

    int querynum = -1;
    while (querynum == -1)
    {

        // find an available query make access to this array atomic
        sem_wait(&query_mutex);
        for (int i = 0; i < MAXNUMQUERIES; i++)
        {
            if (querynum == -1 && queries[i].in_use == false)
            {
                queries[i].in_use = true;
                querynum = i;
                break;
            }
        }
        sem_post(&query_mutex);
        if (querynum == -1)
        {
            // try again in 1 second
            sleep(1);
        }
    }

    if (querynum == -1)
    {
        fprintf(stderr, "unable to find available query\n");
        return -2;
    }

    //initQuery(modelnum, querynum);

    if (guid_ret != 0)
    {
        promptguidstr = (char *)json_object_get_string(guid_json);
        if (strcmp(promptguidstr, emptystr) != 0)
        {
            uuid_parse(promptguidstr, prompt_uuid);
            // look up guid, pull context from disk

            char infilename[4096];
            char guidpath[2048] = GUIDPATH;

            sprintf(infilename, "%s/%s", guidpath, promptguidstr);
            char *infilestr = read_file(infilename);
            if (infilestr == NULL)
            {
                fprintf(stderr, "No bytes read for %s\n", infilename);
                return -4;
            }

            parsed_json = json_tokener_parse(infilestr);
            // pull data

            json_object_object_get_ex(parsed_json, "model", &model_json);

            if (model_ret == 0)
                model_ret = json_object_object_get_ex(parsed_json, "model", &model_json);
            if (context_ret == 0)
                context_ret = json_object_object_get_ex(parsed_json, "context", &context_json);
            if (currwv_ret == 0)
                currwv_ret = json_object_object_get_ex(parsed_json, "currwv", &currwv_json);
            if (mode_ret == 0)
                mode_ret = json_object_object_get_ex(parsed_json, "mode", &mode_json);
            if (temperature_ret == 0)
                temperature_ret = json_object_object_get_ex(parsed_json, "temperature", &temperature_json);
            if (temperature_alt_ret == 0)
                temperature_alt_ret = json_object_object_get_ex(parsed_json, "temperature_alt", &temperature_alt_json);
            if (minp_ret == 0)
                minp_ret = json_object_object_get_ex(parsed_json, "minp", &minp_json);
            if (nummatches_ret == 0)
                nummatches_ret = json_object_object_get_ex(parsed_json, "nummatches", &nummatches_json);
            if (force_gen_tokens_json_ret == 0)
                force_gen_tokens_json_ret = json_object_object_get_ex(parsed_json, "force_gen_tokens", &force_gen_tokens_json);
            if (hardmax_gen_json_ret == 0)
                hardmax_gen_json_ret = json_object_object_get_ex(parsed_json, "hardmax_gen", &hardmax_gen_json);
            if (grammarmax_gen_json_ret == 0)
                grammarmax_gen_json_ret = json_object_object_get_ex(parsed_json, "grammarmax_gen", &grammarmax_gen_json);
            if (paragrammarmax_gen_json_ret == 0)
                paragrammarmax_gen_json_ret = json_object_object_get_ex(parsed_json, "paragrammarmax_gen", &paragrammarmax_gen_json);
            if (seed_json_ret == 0)
                seed_json_ret = json_object_object_get_ex(parsed_json, "seed", &seed_json);
            if (currslot_json_ret == 0)
                currslot_json_ret = json_object_object_get_ex(parsed_json, "currslot", &currslot_json);
            if (start_json_ret == 0)
                start_json_ret = json_object_object_get_ex(parsed_json, "start", &start_json);

            free(infilestr);
        }
    }
    else
    {
        promptguidstr = emptystr;
    }

    if (model_ret == 0)
    {
        fprintf(stderr, "model not specified\n");
        return -5;
    }
    modelname = (char *)json_object_get_string(model_json);

    int modelindex = -1;
    // find model in list of loaded models
    for (int i = 0; i < MAXNUMMODELS; i++)
    {
        if (strcmp(models[i].modelname, modelname) == 0)
        {
            modelindex = i;
            modelnum=modelindex;
            break;
        }
    }

    if (modelindex == -1)
    {
        fprintf(stderr, "requested model not loaded, attempting to load from disk\n");
        // attempt to load model, find first available slot
        int availableslot = -1;
        for (int i = 0; i < MAXNUMMODELS; i++)
        {
            if (models[i].inUse == false)
            {
                availableslot = i;
                break;
            }
        }

        if (availableslot == -1)
        {
            // unable to load model
            fprintf(stderr, "unable to load model\n");
            return -1;
        }
        else
        {
            char *modelpath = lookup_model_path(modelname);
            modelindex = initModel(modelpath, availableslot);
            modelnum = modelindex;
            strcpy(models[modelnum].modelname, modelname);
        }
    }

    initQuery(modelnum, querynum);

    if (promptb64_ret != 0)
    {
        promptb64_string = (char *)json_object_get_string(promptb64_json);
        // it will be in base64 hex, so convert it
        int len = strlen(promptb64_string);
        if (len > 0)
        {
            prompt = b64_decode(promptb64_string, &len);
        }
    }
    else
    {
        // tokenize prompt
        prompt = (char *)json_object_get_string(prompt_json);
    }

    // no prompt guid supplied, pull model state from json request
    if (context_ret != 0)
    {
        context_string = (char *)json_object_get_string(context_json);
        // it will be in base64 hex, so convert it
        int len = strlen(context_string);
        if (len > 0)
        {
            int size_context = ((len / 4) * 3) + 4;
            char *context = NULL;
            context = b64_decode(context_string, &len);
            memcpy(queries[querynum].context, context, len);
            free(context);
        }
    }

    if (currwv_ret != 0)
    {
        currwv_string = (char *)json_object_get_string(currwv_json);
        // it will be in base64 hex, so convert it
        int len = strlen(currwv_string);
        if (len > 0)
        {
            int size_currwv = ((len / 4) * 3) + 4;
            uint8_t *currwv_temp = malloc(size_currwv);

            currwv_temp = b64_decode(currwv_string, &len);
            memcpy(queries[querynum].currwv, currwv_temp, len);
            free(currwv_temp);
        }
    }

    // generate the response guid
    uuid_generate(response_uuid);

    // extract these values from the query
    if (temperature_json != 0)
    {
        queries[querynum].temperature = json_object_get_double(temperature_json);
    }
    else
    {
        queries[querynum].temperature = 1;
    }
    if (temperature_alt_json != 0)
    {
        queries[querynum].temperature_alt = json_object_get_double(temperature_alt_json);
    }
    else
    {
        queries[querynum].temperature_alt = 1;
    }
    if (nummatches_json != 0)
    {
        queries[querynum].nummatches = json_object_get_double(nummatches_json);
    }
    else
    {
        queries[querynum].nummatches = 80;
    }
    if (minp_json != 0)
    {
        queries[querynum].minp = json_object_get_double(minp_json);
    }
    else
    {
        queries[querynum].minp = 0;
    }
    if (seed_json != 0)
    {
        queries[querynum].seed = json_object_get_int(seed_json);
    }
    else
    {
        queries[querynum].seed = time(NULL);
    }
    if (force_gen_tokens_json != 0)
    {
        queries[querynum].force_gen_tokens = json_object_get_int(force_gen_tokens_json);
    }
    else
    {
        queries[querynum].force_gen_tokens = 30;
    }
    if (grammarmax_gen_json != 0)
    {
        queries[querynum].grammarmax_gen = json_object_get_int(grammarmax_gen_json);
    }
    else
    {
        queries[querynum].grammarmax_gen = 30;
    }
    if (paragrammarmax_gen_json != 0)
    {
        queries[querynum].paragrammarmax_gen = json_object_get_int(paragrammarmax_gen_json);
    }
    else
    {
        queries[querynum].paragrammarmax_gen = 3;
    }
    if (hardmax_gen_json != 0)
    {
        queries[querynum].hardmax_gen = json_object_get_int(hardmax_gen_json);
    }
    else
    {
        queries[querynum].hardmax_gen = 30;
    }
    if (currslot_json != 0)
    {
        queries[querynum].currslot = json_object_get_int(currslot_json);
    }
    else
    {
        queries[querynum].currslot = 0;
    }
    if (start_json != 0)
    {
        queries[querynum].start = json_object_get_int(start_json);
    }
    else
    {
        queries[querynum].start = 0;
    }
    if (mode_json != 0)
    {
        char *mode = (char *)json_object_get_string(mode_json);
        if (strcmp(mode, "greedy") == 0)
        {
            queries[querynum].mode = 0;
        }
        else if (strcmp(mode, "sampling") == 0)
        {
            queries[querynum].mode = 1;
        }
    }
    else
    {
        queries[querynum].start = 0;
    }

    // if (queries[querynum].isInitialized == false)
    // {
    //     initQuery(modelnum, querynum);
    // }

    if (modelindex != -1)
    {
        prompt_new = str_replace(prompt, " ", "Ġ");
        char *prompt_new2 = str_replace(prompt_new, "\n", "Ċ");
        promptlgt = tokenize_to_context(prompt_new2, queries[querynum].currslot, modelindex, querynum);
        start = queries[querynum].start;
        genstart = promptlgt;
        genend = genstart + queries[querynum].hardmax_gen-1;
        // send tokens through model
        if (queries[querynum].response != NULL)
            strcpy(queries[querynum].response, "");
        generate(start, genstart, genend, modelnum, querynum, false);

        queries[querynum].currslot++;
        queries[querynum].start = queries[querynum].currslot - 1;

        free(prompt_new);
        free(prompt_new2);
    }

    // write out the results on the from the model
    int currwv_size = models[modelnum].WVSIZE * sizeof(bloom_precision);
    int currwv_b64_size_test = ((((currwv_size * 4) / 3)) + 3 & ~0x03) + 1;
    int currwv_b64_size = B64_ENCODE_STRING_SIZE(currwv_size);
    uint8_t *currwv_b64 = NULL;

    currwv_b64 = b64_encode((const char *)queries[querynum].currwv, currwv_size);
    int context_size = (models[modelnum].WVSIZE * sizeof(bloom_precision));
    int context_b64_size = B64_ENCODE_STRING_SIZE(context_size);
    uint8_t *context_b64 = NULL;
    context_b64 = b64_encode((const char *)queries[querynum].context, context_size);
    token_t *check_context = malloc(context_size);
    check_context = (token_t *)b64_decode(context_b64, &context_b64_size);

    char *response = malloc(sizeof(char) * SEND_BUFFER_SIZE);

    if (promptb64_string == NULL)
    {
        // a b64 prompt wasn't sent, lets go ahead and build one from prompt
        int prompt_size = strlen(prompt);
        int prompt_b64_size = B64_ENCODE_STRING_SIZE(prompt_size);
        promptb64_string = NULL;
        promptb64_string = b64_encode(prompt, prompt_size);
    }

    int response_size = strlen(queries[querynum].response);
    int response_b64_size = B64_ENCODE_STRING_SIZE(response_size);
    uint8_t *response_b64 = NULL;
    response_b64 = b64_encode(queries[querynum].response, response_size);
    char responseguidstr[2048];
    uuid_unparse(response_uuid, responseguidstr);

    sprintf(response, "{  \"model\"   : \"%s\",  \"guid_prompt\"    : \"%s\",  \"guid_response\"    : \"%s\",  \"responseb64\" : \"%s\",  \"currwv\" : \"%s\",  \"context\" : \"%s\", \"start\" : \"%d\", \"currslot\" : \"%d\",  \"genstart\" : \"%d\",  \"genend\" : \"%d\" ,   \"mode\"    : \"%d\",    \"temperature\" : %lf,    \"temperature_alt\" : %lf,    \"minp\"        : %lf,    \"nummatches\"  : %d,   \"force_gen_tokens\"   : %d,  \"hardmax_gen\"         : %d,  \"grammarmax_gen\"      :  %d, \"paragrammarmax_gen\"      :  %d, \"seed\"                : %d }",
            models[modelindex].modelname,
            promptguidstr, responseguidstr,
            response_b64,
            currwv_b64,
            context_b64,
            queries[querynum].start,
            queries[querynum].currslot,
            queries[querynum].genstart,
            queries[querynum].genend,
            queries[querynum].mode,
            queries[querynum].temperature,
            queries[querynum].temperature_alt,
            queries[querynum].minp,
            queries[querynum].nummatches,
            queries[querynum].force_gen_tokens,
            queries[querynum].hardmax_gen,
            queries[querynum].grammarmax_gen,
            queries[querynum].paragrammarmax_gen,
            queries[querynum].seed);

    // save the guid for the input and guid for the output
    FILE *outfile;
    char outfilename[4096];
    char guidpath[2048] = GUIDPATH;

    sprintf(outfilename, "%s/%s", guidpath, responseguidstr);

    if ((outfile = fopen(outfilename, "w")) == NULL)
    {
        fprintf (stderr, "unable to open outputfile %s\n", outfilename);
        fflush(stderr);
        return -3;
    }

    fprintf(outfile, "%s", response);
    fclose(outfile);

    // just send the guid and response in base64 out to the client
    sprintf(response, "{   \"guid_response\"    : \"%s\",  \"responseb64\" : \"%s\" }",
            responseguidstr,
            response_b64);

    *outputsize = strlen(response);
    *jsonstring_output = response;

    free(currwv_b64);

    freeQuery(querynum);
    queries[querynum].isInitialized = false;

    sem_wait(&query_mutex);
    queries[querynum].in_use = false;
    sem_post(&query_mutex);

    return 0;
}


/**
 * @brief  Connection handler
 * @note   
 * @param  *thread_struct: 
 * @retval None
 */
void *connection_handler(bloom_thread_t *thread_struct)
{
    // Get the socket descriptor
    int sock = (int)((bloom_thread_t *)thread_struct)->client_sock;
    int read_size;
    int messagesize;
    char *message, *client_message;
    char *response=NULL;
    int response_size;

    // Receive a message from client
    //  create a 1mb buffer to accept the data
    client_message = malloc(sizeof(char) * RECV_BUFFER_SIZE);
    while ((read_size = recv(sock, client_message, RECV_BUFFER_SIZE, 0)) > 0)
    {
        // process the input
        int ret = process(client_message, &response, read_size, &response_size);
        // write the response
        if (response != NULL)
        {
            ssize_t byteswritten = write(sock, response, strlen(response));
            if (byteswritten != strlen(response))
            {
                fprintf(stderr, "byteswritten != response\n");
            }
            free(response);
        }
        else
        {
            //something wrong
            int q=0;
            q++;
        }
    }
    free(client_message);

    if (read_size == 0)
    {
        puts("Client disconnected");
        fflush(stdout);
    }
    else if (read_size == -1)
    {
        perror("recv failed");
    }

    if (thread_struct->threadId_index != -1 && threadIds[thread_struct->threadId_index] != 0)
    {
    }
    else
    {
        sleep(1);
    }
    threadIds[thread_struct->threadId_index] = 0;

    return 0;
}

/**
 * @brief  Server main entry point
 * @note   
 * @param  serverPort: 
 * @retval None
 */
void ServerStart(int serverPort)
{
    // loop through all queries
    // for (int i=0;i<MAXNUMQUERIES;i++)
    // {

    //     if (queries[i].isInitialized == true)
    //     {
    //         freeQuery(i);
    //         queries[i].isInitialized = false;
    //     }
    //     queries[i].in_use = false;
    // }

    int status;
    // start server
    sem_init(&query_mutex, 0, 1);

    int socket_desc, client_sock, c;
    struct sockaddr_in server, client;
    memset(threadIds, 0, sizeof(threadIds));

    // Create socket
    socket_desc = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_desc == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");

    /*
     * turn off bind address checking, and allow
     * port numbers to be reused - otherwise
     * the TIME_WAIT phenomenon will prevent
     * binding to these address.port combinations
     * for (2 * MSL) seconds.
     */

    int on = 1;

    status = setsockopt(socket_desc, SOL_SOCKET,
                        SO_REUSEADDR,
                        (const char *)&on, sizeof(on));

    if (-1 == status)
    {
        perror("setsockopt(...,SO_REUSEADDR,...)");
    }

    /*
     * when connection is closed, there is a need
     * to linger to ensure all data is
     * transmitted, so turn this on also
     */
    {
        struct linger linger = {0};

        linger.l_onoff = 1;
        linger.l_linger = 30;
        status = setsockopt(socket_desc,
                            SOL_SOCKET, SO_LINGER,
                            (const char *)&linger,
                            sizeof(linger));

        if (-1 == status)
        {
            perror("setsockopt(...,SO_LINGER,...)");
        }
    }

    // Prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(serverPort);

    // Bind
    if (bind(socket_desc, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        // print the error message
        perror("bind failed. Error");
        exit(1);
    }
    puts("bind done");

    // Listen
    listen(socket_desc, 3);

    // Accept and incoming connection
    puts("Waiting for incoming connections...");
    c = sizeof(struct sockaddr_in);
    pthread_t thread_id;
    int current_threadId_index;

    while ((client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t *)&c)))
    {
        puts("Connection accepted");

        // do a sweep through threadids and grab first available threadid slot
        current_threadId_index = -1;
        while (current_threadId_index == -1)
        {
            for (int i = 0; i < MAX_THREADIDS; i++)
            {
                if (threadIds[i] == 0)
                {
                    current_threadId_index = i;
                    break;
                }
            }
            if (current_threadId_index == -1)
                sleep(1);
        }

        bloom_thread_t *thread_struct = malloc(sizeof(bloom_thread_t));
        thread_struct->client_sock = client_sock;
        thread_struct->threadId_index = current_threadId_index;

        if (pthread_create(&thread_id, NULL, (void *(*)(void *))connection_handler, (void *)thread_struct) < 0)
        {
            perror("could not create thread");
            exit(1);
        }
        threadIds[current_threadId_index] = thread_id;

        // Now join the thread , so that we dont terminate before the thread
        puts("Handler assigned");
    }

    if (client_sock < 0)
    {
        perror("accept failed");
        exit(1);
    }
    close(socket_desc);

    exit(1);
}