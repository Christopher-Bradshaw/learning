#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void breaking_locality(void);

int main() {
    breaking_locality();

    return 0;
}

void breaking_locality() {
    // Processes expect that when you use memory location X if is likely you will
    // use the memory near to it. So, when they load X into cache they pull a full
    // cache line. Find the size of this line with:
    // cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
	// There is a ton of other interesting stuff there, take a look
    printf("Successive memory accesses should ideally be near to each other\n");
    time_t start;
    int size = 10000;
	double **arr = calloc(size, sizeof(double*));
    for (int i = 0; i < size; i++) {
        arr[i] = calloc(size, sizeof(double));
    }


    // Good, the inner loop runs over contiguous memory. Cache hits
    start = clock();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            arr[i][j] = i*j;
        }
    }
    printf("Good: %lfs\n", (clock() - start)/1e6);
    /*
        $ valgrind --tool=cachegrind ./bad_cache
        LL - last level of cache. Here we are completely missing the cache 1/32 of the time
        ==30418== D   refs:      1,799,997,018  (900,699,991 rd   + 899,297,027 wr)
        ==30418== D1  misses:       25,218,072  (    187,297 rd   +  25,030,775 wr)
        ==30418== LLd misses:       24,991,607  (      4,012 rd   +  24,987,595 wr)
        ==30418== D1  miss rate:           1.4% (        0.0%     +         2.8%  )
        ==30418== LLd miss rate:           1.4% (        0.0%     +         2.8%  )
        ==30418==
        ==30418== LL refs:          25,219,412  (    188,637 rd   +  25,030,775 wr)
        ==30418== LL misses:        24,992,942  (      5,347 rd   +  24,987,595 wr)
        ==30418== LL miss rate:            0.6% (        0.0%     +         2.8%  )
    */

    // Bad, successive loops access memory that is far away. Cache misses
    start = clock();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            arr[j][i] = i*j;
        }
    }
    printf("Bad: %lfs\n", (clock() - start)/1e6);
    /*
        LL - last level of cache. Here we are completely missing the cache 1/8 of the time
        ==30382== D   refs:      1,799,997,069  (900,700,034 rd   + 899,297,035 wr)
        ==30382== D1  misses:      125,223,753  ( 12,696,053 rd   + 112,527,700 wr)
        ==30382== LLd misses:      111,492,448  (     14,087 rd   + 111,478,361 wr)
        ==30382== D1  miss rate:           7.0% (        1.4%     +        12.5%  )
        ==30382== LLd miss rate:           6.2% (        0.0%     +        12.4%  )
        ==30382==
        ==30382== LL refs:         125,225,101  ( 12,697,401 rd   + 112,527,700 wr)
        ==30382== LL misses:       111,493,791  (     15,430 rd   + 111,478,361 wr)
        ==30382== LL miss rate:            2.5% (        0.0%     +        12.4%  )
    */

    printf("Not sure why we get 1/8 and 1/32 cache miss rate...\n");
}
