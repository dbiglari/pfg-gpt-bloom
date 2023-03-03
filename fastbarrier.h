#ifndef __FASTBARRIER_H__
#define __FASTBARRIER_H__

typedef struct fast_barrier_t fast_barrier_t;
struct fast_barrier_t
{
	union
	{
		struct
		{
			unsigned seq;
			unsigned count;
		};
		unsigned long long reset;
	};
	unsigned refcount;
	unsigned total;
	int spins;
	unsigned flags;
};

void fast_barrier_init(fast_barrier_t *b, pthread_barrierattr_t *a, unsigned count);
void fast_barrier_destroy(fast_barrier_t *b);
int fast_barrier_wait(fast_barrier_t *b);


#endif