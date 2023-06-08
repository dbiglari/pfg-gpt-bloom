#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h> //strlen
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <unistd.h>    //write
#include <pthread.h>   //for threading , link with lpthread
#include <semaphore.h>
#include "common.h"
#include "uuid/uuid.h"
#include "client.h"
// #include "base64.h"
#include "json-c/json.h"
#include "base64.h"


#define SA struct sockaddr
#define RECV_BUFFER_SIZE 1048576 * 2
#define SEND_BUFFER_SIZE 1048576 * 2

/**
 * @brief  Perform DNS lookup
 * @note   
 * @param  *host: host name
 * @retval 
 */
int lookup_host(const char *host)
{
    struct addrinfo hints, *res, *result;
    int errcode;
    char addrstr[100];
    void *ptr;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags |= AI_CANONNAME;

    errcode = getaddrinfo(host, NULL, &hints, &result);
    if (errcode != 0)
    {
        perror("getaddrinfo");
        return -1;
    }

    res = result;

    printf("Host: %s\n", host);
    while (res)
    {
        inet_ntop(res->ai_family, res->ai_addr->sa_data, addrstr, 100);

        switch (res->ai_family)
        {
        case AF_INET:
            ptr = &((struct sockaddr_in *)res->ai_addr)->sin_addr;
            break;
        case AF_INET6:
            ptr = &((struct sockaddr_in6 *)res->ai_addr)->sin6_addr;
            break;
        }
        inet_ntop(res->ai_family, ptr, addrstr, 100);
        printf("IPv%d address: %s (%s)\n", res->ai_family == PF_INET6 ? 6 : 4,
               addrstr, res->ai_canonname);
        res = res->ai_next;
    }

    freeaddrinfo(result);

    return 0;
}

/**
 * @brief  Client main entry point
 * @note   
 * @param  *clientServerAddress: NULL terminated character string containing server address
 * @param  clientServerPort: server port
 * @retval 
 */
int client_main(char *clientServerAddress, int clientServerPort)
{
    char emptystr[1] = "";
    char *guid_response = emptystr;
    char server_message[RECV_BUFFER_SIZE + 1];
    char senddata[SEND_BUFFER_SIZE];
    char modelname[1024];
    struct json_object *parsed_json;
    struct json_object *response_json;
    struct json_object *guid_response_json;
    char *response = NULL;

    bool run = true;
    char server[1024];
    int port = 8081;
    strcpy(server, clientServerAddress);

    int sockfd, connfd;
    struct sockaddr_in servaddr, cli;

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(server);
    servaddr.sin_port = htons(port);

    // connect the client socket to server socket
    if (connect(sockfd, (SA *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connected to the server..\n");

    strcpy(modelname, "bloom-alpaca-560m");
    double temperature = 1.0;
    double temperature_alt = 1.0;
    double min_p = 0.0;
    int nummatches = 80;
    char *prompt = NULL;
    size_t promptsize = 0;

    while (run == true)
    {
        // open a connection to the server
        printf("prompt: ");
        ssize_t retsize = getline(&prompt, &promptsize, stdin);
        if (retsize == 0)
        {
            printf("\n");
        }

        // remove carrage return at end of string
        prompt[strlen(prompt) - 1] = 0;

        int prompt_size = strlen(prompt);
        int prompt_b64_size = B64_ENCODE_STRING_SIZE(prompt_size);
        uint8_t *prompt_b64 = (uint8_t *)malloc(prompt_b64_size);
        prompt_b64 = b64_encode(prompt, prompt_size);

        // send prompt to server
        if (strcmp(guid_response, emptystr) == 0)
        {
            sprintf(senddata, "{   \"model\"   : \"%s\",   \"guid\"    : \"%s\",   \"promptb64\"  : \"%s\", \"currslot\" : 0,  \"context\" : \"\",   \"mode\"    : \"%s\",    \"temperature\" : %lf,     \"temperature_alt\" : %lf,     \"minp\"        : %lf,    \"nummatches\"  : 80,    \"force_gen_tokens\"   : %d,   \"hardmax_gen\"         : %d,   \"grammarmax_gen\"      :  %d, \"paragrammarmax_gen\"      :  %d, \"seed\" : %d }", models[0].modelname, guid_response, prompt_b64, queries[0].mode == 0 ? "greedy" : "sampling", queries[0].temperature, queries[0].temperature_alt, queries[0].minp, queries[0].force_gen_tokens, queries[0].hardmax_gen, queries[0].grammarmax_gen, queries[0].paragrammarmax_gen, queries[0].seed);
        }
        else
        {
            sprintf(senddata, "{  \"guid\"    : \"%s\",   \"promptb64\"  : \"%s\"}", guid_response, prompt_b64);
        }

        ssize_t byteswritten = write(sockfd, senddata, strlen(senddata));
        if (byteswritten != strlen(senddata))
        {
            fprintf(stderr, "byteswritten != senddata\n");
        }

        ssize_t read_size = recv(sockfd, server_message, RECV_BUFFER_SIZE, 0);
        if (read_size == 0)
        {
            printf("server disconnected\n");
            run = false;
            continue;
        }

        // extract response string
        parsed_json = json_tokener_parse(server_message);

        json_object_object_get_ex(parsed_json, "responseb64", &response_json);
        json_object_object_get_ex(parsed_json, "guid_response", &guid_response_json);

        char *responseb64_string = (char *)json_object_get_string(response_json);
        guid_response = (char *)json_object_get_string(guid_response_json);

        int len = strlen(responseb64_string);
        if (len > 0)
        {
            int size_responseb64 = ((len / 4) * 3) + 4;
            response = b64_decode(responseb64_string, &len);
        }

        // get response
        if (response != NULL)
        {
            printf("response: %s\n", response);
        }

        if (response != NULL)
        {
            free(response);
            response = NULL;
        }
        if (prompt != NULL)
        {
            free(prompt);
            prompt = NULL;
        }
    }
    exit(0);
}
