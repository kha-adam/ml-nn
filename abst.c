#include <time.h>
#include <stdio.h>

#define NEURNET_IMPLEMENTATION
#include "neurnet.h"

float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,
};

int main(void) {

    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    Mat ti = {
	.rows = n,
	.cols = 2,
	.stride = stride,
	.es = td,
    };
    Mat to = {
	.rows = n,
	.cols = 1,
	.stride = stride,
	.es = td + 2,
    };

    size_t arch[] = {2, 2, 1};
    neurnet nn = neurnet_alloc(arch, ARRAY_LEN(arch));
    neurnet gg = neurnet_alloc(arch, ARRAY_LEN(arch));
    neurnet_rand(nn, 0, 1);
    
    float eps = 1e-1;
    float rate = 1e-1;
    float tol = 1e-1;

    printf("cost = %f\n", neurnet_cost(nn, ti, to));
    for (size_t i = 0; i < 100 * 1000; i++){
	neurnet_finite_diff(nn, gg, eps, ti, to);
	neurnet_learn(nn, gg, rate);
    }
    printf("cost = %f\n", neurnet_cost(nn, ti, to));

    for (size_t i = 0; i < 2; i++){
	for (size_t j = 0; j < 2; j++){
	    MAT_AT(NEURNET_INPUT(nn), 0, 0) = i;
	    MAT_AT(NEURNET_INPUT(nn), 0, 1) = j;
	    neurnet_forward(nn);
	    printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NEURNET_OUTPUT(nn), 0, 0));
	}
    }
    return 0;
    
}
