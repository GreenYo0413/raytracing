#ifndef __RAY_MATH_TOOLKIT_H
#define __RAY_MATH_TOOLKIT_H

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include "openmp.h"

static inline
void normalize(double *v)
{
    double d = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    assert(d != 0.0 && "Error calculating normal");

    v[0] /= d;
    v[1] /= d;
    v[2] /= d;
}

static inline
double length(const double *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static inline
void add_vector(const double *a, const double *b, double *out)
{
    __m256d a_reg = _mm256_set_pd ((double)0, a[2], a[1], a[0]);
    __m256d b_reg = _mm256_set_pd ((double)0, b[2], b[1], b[0]);
    __m256d a_add_b = _mm256_add_pd( a_reg, b_reg );
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, a_add_b);
    out[0] = (double)tmp[0];
    out[1] = (double)tmp[1];
    out[2] = (double)tmp[2];
}

static inline
void subtract_vector(const double *a, const double *b, double *out)
{
    __m256d a_reg = _mm256_set_pd ((double)0, a[2], a[1], a[0]);
    __m256d b_reg = _mm256_set_pd ((double)0, b[2], b[1], b[0]);
    __m256d a_add_b = _mm256_sub_pd( a_reg, b_reg );
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, a_add_b);
    out[0] = (double)tmp[0];
    out[1] = (double)tmp[1];
    out[2] = (double)tmp[2];
}

static inline
void multiply_vectors(const double *a, const double *b, double *out)
{
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
}

static inline
void multiply_vector(const double *a, double b, double *out)
{
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
}

static inline
void cross_product(const double *v1, const double *v2, double *out)
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

static inline
double dot_product(const double *v1, const double *v2)
{
    __m256d v1_reg = _mm256_set_pd ((double)0, v1[2], v1[1], v1[0]);
    __m256d v2_reg = _mm256_set_pd ((double)0, v2[2], v2[1], v2[0]);
    __m256d v1_add_v2 = _mm256_mul_pd( v1_reg, v2_reg );
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, v1_add_v2);
    return (double)(tmp[0]+tmp[1]+tmp[2]);

}

static inline
void scalar_triple_product(const double *u, const double *v, const double *w,
                           double *out)
{
    cross_product(v, w, out);
    multiply_vectors(u, out, out);
}

static inline
double scalar_triple(const double *u, const double *v, const double *w)
{
    double tmp[3];
    cross_product(w, u, tmp);
    return dot_product(v, tmp);
}

#endif
