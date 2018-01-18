/* Copyright 2017 The MathWorks, Inc. */

#ifndef __LEAKY_RELU_IMPL_HPP
#define __LEAKY_RELU_IMPL_HPP

void leakyReLUForwardImpl(int YgcpEBUCwCLaPhyntIio,
                          int vIWQzNvYZSuxmOTVDFhU,
                          int gzSTokDHvkXefhiGDcWL,
                          int LtEgcYoEYjkrWuohutgw,                          
                          const double qWwjVYwfnvEnFKlgpqwA,
                          float* output);

/*
 * Device code
 */
void __global__ leakyReLUImpl(float  * AFQBkxwYGKLsACiDKwRM,
                              const double AwZQzUhuWVLGrWgLHRuM,
                              const int CpMjJjtGOeWOzwxpAAQP);

#endif
