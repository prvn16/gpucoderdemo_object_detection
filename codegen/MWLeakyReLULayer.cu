#include "MWLeakyReLULayer.hpp"
#include "MWLeakyReLULayerImpl.hpp"
 MWLeakyReLULayer::MWLeakyReLULayer() { 
CUDNN_CALL(cudnnCreateTensorDescriptor(getOutputDescriptor())); } 
MWLeakyReLULayer::~MWLeakyReLULayer() { 
CUDNN_CALL(cudnnDestroyTensorDescriptor(*getOutputDescriptor())); } void 
MWLeakyReLULayer::createLeakyReLULayer(MWCNNLayer* jaqKGCwoANNDMHgAsehk, 
double JwxFdqOKggeawILBfGgg) {  setTop(jaqKGCwoANNDMHgAsehk); 
setBatchSize(getTop()->getBatchSize()); setHeight(getTop()->getHeight()); 
setWidth(getTop()->getWidth()); 
setNumInputFeatures(getTop()->getNumOutputFeatures()); 
setNumOutputFeatures(getNumInputFeatures()); 
setWorkSpaceSize(*getTop()->getWorkSpaceSize()); kkqTyvjYvRFtTOyQUwrF = 
JwxFdqOKggeawILBfGgg; 
CUDNN_CALL(cudnnSetTensor4dDescriptor(*getOutputDescriptor(), 
CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, getBatchSize(), getNumOutputFeatures(), 
getHeight(), getWidth())); setData(getTop()->getData()); return; } void 
MWLeakyReLULayer::predict() { leakyReLUForwardImpl(getHeight(), getWidth(), 
getNumInputFeatures(), getBatchSize(), kkqTyvjYvRFtTOyQUwrF, getData());  return; } 
void MWLeakyReLULayer::cleanup() { return; }