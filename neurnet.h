// #define NEURNET_IMPLEMENTATION

#ifndef NEURNET_H_
#define NEURNET_H_

#include <stddef.h>
#include <stdio.h>

#ifndef NEURNET_MALLOC
#include <stdlib.h>
#define NEURNET_MALLOC malloc
#endif// NEURNET_MALLOC

#ifndef NEURNET_ASSERT
#include <assert.h>
#define NEURNET_ASSERT assert
#endif // NEURNET_ASSERT
typedef struct {
    size_t rows;
    size_t cols;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

float rand_float(void);

void mat_fill(Mat a, float num);
Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a, Mat b);
void mat_print(Mat m);

#endif // NEURNET_H_

#ifdef NEURNET_IMPLEMENTATION

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

void mat_fill(Mat a, float num){
    for (size_t i = 0; i < a.rows; i++){
	for (size_t j = 0; j < a.cols; j++){
	    MAT_AT(a, i, j) = num;
	    }
    }
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = NEURNET_MALLOC(sizeof(*m.es)*rows*cols);
    NEURNET_ASSERT(m.es != NULL);
    return m;
}
void mat_dot(Mat dest, Mat a, Mat b){
    NEURNET_ASSERT(a.cols == b.rows);
    NEURNET_ASSERT(dest.rows == a.rows && dest.cols == b.cols);
    for (size_t i = 0; i < dest.rows; i++){
	for (size_t j = 0; j < dest.cols; j++){
	    MAT_AT(dest, i, j) = 0;
	    for (size_t k = 0; k < a.cols; k++){
		MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
	    }
	}
    }
}
void mat_sum(Mat dest, Mat a, Mat b){
    NEURNET_ASSERT(dest.rows == a.rows && b.rows == a.rows);
    NEURNET_ASSERT(dest.cols == a.cols && b.cols == a.cols);
    for (size_t i = 0; i < dest.rows; i++){
	for (size_t j = 0; j < dest.cols; j++){
	    MAT_AT(dest, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
	}
    }
}
void mat_print(Mat m){
    for (size_t i = 0; i < m.rows; i++){
	for (size_t j = 0; j < m.cols; j++){
	    printf("%f ", MAT_AT(m, i, j));
	}
	printf("\n");
    }
}
void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++){
	for(size_t j = 0; j < m.cols; j++){
	    MAT_AT(m, i, j) = rand_float()*(high - low) + low;

	}
    }
}
#endif //NEURNET_IMPLEMENTATION
