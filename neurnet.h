#define NEURNET_IMPLEMENTATION

#ifndef NEURNET_H_
#define NEURNET_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NEURNET_MALLOC
#include <stdlib.h>
#define NEURNET_MALLOC malloc
#endif// NEURNET_MALLOC

#ifndef NEURNET_ASSERT
#include <assert.h>
#define NEURNET_ASSERT assert
#endif // NEURNET_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

void mat_fill(Mat a, float num);
Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
Mat mat_col(Mat m, size_t col);
void mat_copy(Mat dest, Mat src);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a, Mat b);
void mat_sig(Mat a);
void mat_print(Mat m, char * name);
#define MAT_PRINT(m) mat_print(m, #m);

typedef struct{
    
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // count + 1 (don't forget initial)

} neurnet;

#define NEURNET_INPUT(nn) (nn).as[0]
#define NEURNET_OUTPUT(nn) (nn).as[(nn).count]

neurnet neurnet_alloc(size_t *arch, size_t arch_count);
void neurnet_rand(neurnet nn, float low, float high);
void neurnet_forward(neurnet nn);
float neurnet_cost(neurnet nn, Mat ti, Mat to);
void neurnet_finite_diff(neurnet nn, neurnet gg, float eps, Mat ti, Mat to);
void neurnet_learn(neurnet nn, neurnet gg, float rate);
void neurnet_print(neurnet nn, const char *name);
#define NEURNET_PRINT(nn) neurnet_print(nn, #nn);

#endif // NEURNET_H_

#ifdef NEURNET_IMPLEMENTATION

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
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
    m.stride = m.cols;
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
Mat mat_row(Mat m, size_t row){
    return (Mat) {
	.rows = 1,
	.cols = m.cols,
	.stride = m.stride,
	.es = &MAT_AT(m, row, 0),
    };
}
void mat_copy(Mat dest, Mat src){
    NEURNET_ASSERT(dest.rows == src.rows);
    NEURNET_ASSERT(dest.cols == src.cols);
    for (size_t i = 0; i < dest.rows; i++){
	for(size_t j = 0; j < dest.cols; j++){
	    MAT_AT(dest, i, j) = MAT_AT(src, i, j);
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
void mat_sig(Mat a)
{
    for (size_t i = 0; i < a.rows; i++){
	for(size_t j = 0; j < a.cols; j++){
	    MAT_AT(a, i, j) = sigmoidf(MAT_AT(a, i, j));
	}
    }
}
void mat_print(Mat m, char *name){
    printf("%s: [\n", name);
    for (size_t i = 0; i < m.rows; i++){
	for (size_t j = 0; j < m.cols; j++){
	    printf("	%f ", MAT_AT(m, i, j));
	}
	printf("\n");
    }
    printf("]\n");
}
void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++){
	for(size_t j = 0; j < m.cols; j++){
	    MAT_AT(m, i, j) = rand_float()*(high - low) + low;

	}
    }
}


neurnet neurnet_alloc(size_t *arch, size_t arch_count)
{
    NEURNET_ASSERT(arch_count > 0);
    neurnet nn;
    nn.count = arch_count - 1;

    nn.ws = NEURNET_MALLOC(sizeof(*nn.ws)*nn.count);
    NEURNET_ASSERT(nn.ws != NULL);
    nn.bs = NEURNET_MALLOC(sizeof(*nn.bs)*nn.count);
    NEURNET_ASSERT(nn.bs != NULL);
    nn.as = NEURNET_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NEURNET_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; i++){
	nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
	nn.bs[i-1] = mat_alloc(1, arch[i]);
	nn.as[i] = mat_alloc(1, arch[i]);
    }
    return nn;
};

void neurnet_print(neurnet nn, const char *name){
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++) {
	snprintf(buf, sizeof(buf), "ws%zu", i);
	mat_print(nn.ws[i], buf);
	snprintf(buf, sizeof(buf), "bs%zu", i);
	mat_print(nn.bs[i], buf);

    }
    printf("]\n");
}
void neurnet_rand(neurnet nn, float low, float high){
    for (size_t i = 0; i < nn.count; i++){
	mat_rand(nn.ws[i], low, high);
	mat_rand(nn.bs[i], low, high);
    }
}
void neurnet_forward(neurnet nn){
    for (size_t i = 0; i < nn.count; i++){
	mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
	mat_sum(nn.as[i+1], nn.as[i+1], nn.bs[i]);
	mat_sig(nn.as[i+1]);
    }
}
float neurnet_cost(neurnet nn, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == NEURNET_OUTPUT(nn).cols);
    float c = 0;
    for (size_t i = 0; i < ti.rows; i++){
	Mat x = mat_row(ti, i);
	Mat y = mat_row(to, i);
	mat_copy(NEURNET_INPUT(nn), x);
	neurnet_forward(nn);

	for (size_t j = 0; j < to.cols; j++) {
	    float d = MAT_AT(NEURNET_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
	    c+= d*d;
    }
}
    return c/ti.rows;
}


void neurnet_finite_diff(neurnet nn, neurnet gg, float eps, Mat ti, Mat to){
    
    float saved;
    float c = neurnet_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; i++){
	for (size_t j = 0; j < nn.ws[i].rows; j++) {
	    for (size_t k = 0; k < nn.ws[i].cols; k++){
		saved = MAT_AT(nn.ws[i], j, k);
		MAT_AT(nn.ws[i], j, k) += eps;
		MAT_AT(gg.ws[i], j, k) = (neurnet_cost(nn, ti, to) - c)/eps;
		MAT_AT(nn.ws[i], j, k) = saved;
	    }
	}
	for (size_t j = 0; j < nn.bs[i].rows; j++) {
	    for (size_t k = 0; k < nn.bs[i].cols; k++){
		saved = MAT_AT(nn.bs[i], j, k);
		MAT_AT(nn.bs[i], j, k) += eps;
		MAT_AT(gg.bs[i], j, k) = (neurnet_cost(nn, ti, to) - c)/eps;
		MAT_AT(nn.bs[i], j, k) = saved;
	    }
	}
    }
}

void neurnet_learn(neurnet nn, neurnet gg, float rate){

    for (size_t i = 0; i < nn.count; i++){
	for (size_t j = 0; j < nn.ws[i].rows; j++) {
	    for (size_t k = 0; k < nn.ws[i].cols; k++){
		MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(gg.ws[i], j, k);
	    }
	}
	for (size_t j = 0; j < nn.bs[i].rows; j++) {
	    for (size_t k = 0; k < nn.bs[i].cols; k++){
		MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(gg.bs[i], j, k);
	    }
	}
    }
    
}

#endif //NEURNET_IMPLEMENTATION
