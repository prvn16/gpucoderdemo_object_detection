#include "MWLeakyReLULayerImpl.hpp"
 void __global__ leakyReLUImpl(float * AFQBkxwYGKLsACiDKwRM, const double AwZQzUhuWVLGrWgLHRuM, 
const int CpMjJjtGOeWOzwxpAAQP) { int const i = blockDim.x * blockIdx.x + threadIdx.x; 
if (i < CpMjJjtGOeWOzwxpAAQP) { float tf = float(AFQBkxwYGKLsACiDKwRM[i]<0); AFQBkxwYGKLsACiDKwRM[i] = 
AFQBkxwYGKLsACiDKwRM[i] - tf*AwZQzUhuWVLGrWgLHRuM*AFQBkxwYGKLsACiDKwRM[i]; } } void 
leakyReLUForwardImpl(int YgcpEBUCwCLaPhyntIio, int vIWQzNvYZSuxmOTVDFhU, int 
gzSTokDHvkXefhiGDcWL, int LtEgcYoEYjkrWuohutgw,  const double 
kkqTyvjYvRFtTOyQUwrF, float* output) { int fjfzkUfcCOqjrkAVGfuc = 
YgcpEBUCwCLaPhyntIio*vIWQzNvYZSuxmOTVDFhU* 
gzSTokDHvkXefhiGDcWL*LtEgcYoEYjkrWuohutgw; int 
pckLLTEdVPoCZLRwyDnM = (fjfzkUfcCOqjrkAVGfuc < 1024) ? fjfzkUfcCOqjrkAVGfuc : 
1024; int MEmIeGILUZNEWEagSzRk = (fjfzkUfcCOqjrkAVGfuc + 
pckLLTEdVPoCZLRwyDnM - 1)/pckLLTEdVPoCZLRwyDnM; 
leakyReLUImpl<<<MEmIeGILUZNEWEagSzRk, pckLLTEdVPoCZLRwyDnM>>>( 
output, (1 - kkqTyvjYvRFtTOyQUwrF), fjfzkUfcCOqjrkAVGfuc); }