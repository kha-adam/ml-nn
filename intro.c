#include <time.h>
#include <stdio.h>

#define NEURNET_IMPLEMENTATION
#include "neurnet.h"


int main(void) {

    srand(time(0));
    Mat a = mat_alloc(2, 3);
    Mat b = mat_alloc(3, 1);
    Mat dest = mat_alloc(2, 1);

    mat_fill(a, 2);
    mat_fill(b, 7);

    mat_print(a);
    printf("-----------------\n");
    mat_print(b);
    printf("-----------------\n");
    mat_dot(dest, a, b);
    mat_print(dest);

    return 0;
}
