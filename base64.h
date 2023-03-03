
#define ROUND_UP_TO_NEAREST_4(i) (i + 3) & ~0x03
#define B64_ENCODE_SIZE(x) ROUND_UP_TO_NEAREST_4(((((x * 4) / 3))))
#define B64_ENCODE_STRING_SIZE(x) ((B64_ENCODE_SIZE(x)) + 1) * 2

#define B64_DECODE_STRING_SIZE(x) ((x / 4) * 3) + 4;

char *b64_encode(const char *input, int size);
char *b64_decode(const char *input, int *size);