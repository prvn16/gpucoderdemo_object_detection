
#include "cnn_exec.hpp"
#include "MWLeakyReLULayer.hpp"
#include "MWLeakyReLULayerImpl.hpp"
CnnMain::CnnMain()
{
    this->numLayers = 58;
    this->cublasHandle = 0;
    this->cudnnHandle = 0;
    this->workSpace = 0;
    this->layers[0] = new MWInputLayer;
    this->layers[1] = new MWConvLayer;
    this->layers[2] = new MWLeakyReLULayer;
    this->layers[3] = new MWMaxPoolingLayer;
    this->layers[4] = new MWConvLayer;
    this->layers[5] = new MWLeakyReLULayer;
    this->layers[6] = new MWMaxPoolingLayer;
    this->layers[7] = new MWConvLayer;
    this->layers[8] = new MWLeakyReLULayer;
    this->layers[9] = new MWConvLayer;
    this->layers[10] = new MWLeakyReLULayer;
    this->layers[11] = new MWConvLayer;
    this->layers[12] = new MWLeakyReLULayer;
    this->layers[13] = new MWConvLayer;
    this->layers[14] = new MWLeakyReLULayer;
    this->layers[15] = new MWMaxPoolingLayer;
    this->layers[16] = new MWConvLayer;
    this->layers[17] = new MWLeakyReLULayer;
    this->layers[18] = new MWConvLayer;
    this->layers[19] = new MWLeakyReLULayer;
    this->layers[20] = new MWConvLayer;
    this->layers[21] = new MWLeakyReLULayer;
    this->layers[22] = new MWConvLayer;
    this->layers[23] = new MWLeakyReLULayer;
    this->layers[24] = new MWConvLayer;
    this->layers[25] = new MWLeakyReLULayer;
    this->layers[26] = new MWConvLayer;
    this->layers[27] = new MWLeakyReLULayer;
    this->layers[28] = new MWConvLayer;
    this->layers[29] = new MWLeakyReLULayer;
    this->layers[30] = new MWConvLayer;
    this->layers[31] = new MWLeakyReLULayer;
    this->layers[32] = new MWConvLayer;
    this->layers[33] = new MWLeakyReLULayer;
    this->layers[34] = new MWConvLayer;
    this->layers[35] = new MWLeakyReLULayer;
    this->layers[36] = new MWMaxPoolingLayer;
    this->layers[37] = new MWConvLayer;
    this->layers[38] = new MWLeakyReLULayer;
    this->layers[39] = new MWConvLayer;
    this->layers[40] = new MWLeakyReLULayer;
    this->layers[41] = new MWConvLayer;
    this->layers[42] = new MWLeakyReLULayer;
    this->layers[43] = new MWConvLayer;
    this->layers[44] = new MWLeakyReLULayer;
    this->layers[45] = new MWConvLayer;
    this->layers[46] = new MWLeakyReLULayer;
    this->layers[47] = new MWConvLayer;
    this->layers[48] = new MWLeakyReLULayer;
    this->layers[49] = new MWConvLayer;
    this->layers[50] = new MWLeakyReLULayer;
    this->layers[51] = new MWConvLayer;
    this->layers[52] = new MWLeakyReLULayer;
    this->layers[53] = new MWFCLayer;
    this->layers[54] = new MWLeakyReLULayer;
    this->layers[55] = new MWFCLayer;
    this->layers[56] = new MWSoftmaxLayer;
    this->layers[57] = new MWOutputLayer;
}
void CnnMain::setup()
{
    int32_T idx_handles;
    int32_T idx_ws;
    this->cublasHandle = new cublasHandle_t;
    cublasCreate(this->cublasHandle);
    this->cudnnHandle = new cudnnHandle_t;
    cudnnCreate(this->cudnnHandle);
    for (idx_handles = 0; idx_handles < 58; idx_handles++) {
        this->layers[idx_handles]->setCublasHandle(this->cublasHandle);
        this->layers[idx_handles]->setCudnnHandle(this->cudnnHandle);
    }
    this->layers[0]->createInputLayer(1, 448, 448, 3, 0);
    this->layers[1]->createConvLayer(this->layers[0], 7, 7, 3, 64, 2, 2, 3, 3, 1);
    this->layers[1]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer_w");
    this->layers[1]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer_b");
    this->layers[2]->createLeakyReLULayer(this->layers[1], 0.1);
    this->layers[3]->createMaxPoolingLayer(this->layers[2], 2, 2, 2, 2, 0, 0);
    this->layers[4]->createConvLayer(this->layers[3], 3, 3, 64, 192, 1, 1, 1, 1, 1);
    this->layers[4]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer0_w");
    this->layers[4]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer0_b");
    this->layers[5]->createLeakyReLULayer(this->layers[4], 0.1);
    this->layers[6]->createMaxPoolingLayer(this->layers[5], 2, 2, 2, 2, 0, 0);
    this->layers[7]->createConvLayer(this->layers[6], 1, 1, 192, 128, 1, 1, 0, 0, 1);
    this->layers[7]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer1_w");
    this->layers[7]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer1_b");
    this->layers[8]->createLeakyReLULayer(this->layers[7], 0.1);
    this->layers[9]->createConvLayer(this->layers[8], 3, 3, 128, 256, 1, 1, 1, 1, 1);
    this->layers[9]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer2_w");
    this->layers[9]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer2_b");
    this->layers[10]->createLeakyReLULayer(this->layers[9], 0.1);
    this->layers[11]->createConvLayer(this->layers[10], 1, 1, 256, 256, 1, 1, 0, 0, 1);
    this->layers[11]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer3_w");
    this->layers[11]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer3_b");
    this->layers[12]->createLeakyReLULayer(this->layers[11], 0.1);
    this->layers[13]->createConvLayer(this->layers[12], 3, 3, 256, 512, 1, 1, 1, 1, 1);
    this->layers[13]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer4_w");
    this->layers[13]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer4_b");
    this->layers[14]->createLeakyReLULayer(this->layers[13], 0.1);
    this->layers[15]->createMaxPoolingLayer(this->layers[14], 2, 2, 2, 2, 0, 0);
    this->layers[16]->createConvLayer(this->layers[15], 1, 1, 512, 256, 1, 1, 0, 0, 1);
    this->layers[16]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer5_w");
    this->layers[16]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer5_b");
    this->layers[17]->createLeakyReLULayer(this->layers[16], 0.1);
    this->layers[18]->createConvLayer(this->layers[17], 3, 3, 256, 512, 1, 1, 1, 1, 1);
    this->layers[18]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer6_w");
    this->layers[18]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer6_b");
    this->layers[19]->createLeakyReLULayer(this->layers[18], 0.1);
    this->layers[20]->createConvLayer(this->layers[19], 1, 1, 512, 256, 1, 1, 0, 0, 1);
    this->layers[20]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer7_w");
    this->layers[20]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer7_b");
    this->layers[21]->createLeakyReLULayer(this->layers[20], 0.1);
    this->layers[22]->createConvLayer(this->layers[21], 3, 3, 256, 512, 1, 1, 1, 1, 1);
    this->layers[22]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer8_w");
    this->layers[22]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer8_b");
    this->layers[23]->createLeakyReLULayer(this->layers[22], 0.1);
    this->layers[24]->createConvLayer(this->layers[23], 1, 1, 512, 256, 1, 1, 0, 0, 1);
    this->layers[24]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer9_w");
    this->layers[24]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer9_b");
    this->layers[25]->createLeakyReLULayer(this->layers[24], 0.1);
    this->layers[26]->createConvLayer(this->layers[25], 3, 3, 256, 512, 1, 1, 1, 1, 1);
    this->layers[26]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer10_w");
    this->layers[26]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer10_b");
    this->layers[27]->createLeakyReLULayer(this->layers[26], 0.1);
    this->layers[28]->createConvLayer(this->layers[27], 1, 1, 512, 256, 1, 1, 0, 0, 1);
    this->layers[28]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer11_w");
    this->layers[28]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer11_b");
    this->layers[29]->createLeakyReLULayer(this->layers[28], 0.1);
    this->layers[30]->createConvLayer(this->layers[29], 3, 3, 256, 512, 1, 1, 1, 1, 1);
    this->layers[30]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer12_w");
    this->layers[30]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer12_b");
    this->layers[31]->createLeakyReLULayer(this->layers[30], 0.1);
    this->layers[32]->createConvLayer(this->layers[31], 1, 1, 512, 512, 1, 1, 0, 0, 1);
    this->layers[32]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer13_w");
    this->layers[32]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer13_b");
    this->layers[33]->createLeakyReLULayer(this->layers[32], 0.1);
    this->layers[34]->createConvLayer(this->layers[33], 3, 3, 512, 1024, 1, 1, 1, 1, 1);
    this->layers[34]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer14_w");
    this->layers[34]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer14_b");
    this->layers[35]->createLeakyReLULayer(this->layers[34], 0.1);
    this->layers[36]->createMaxPoolingLayer(this->layers[35], 2, 2, 2, 2, 0, 0);
    this->layers[37]->createConvLayer(this->layers[36], 1, 1, 1024, 512, 1, 1, 0, 0, 1);
    this->layers[37]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer15_w");
    this->layers[37]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer15_b");
    this->layers[38]->createLeakyReLULayer(this->layers[37], 0.1);
    this->layers[39]->createConvLayer(this->layers[38], 3, 3, 512, 1024, 1, 1, 1, 1, 1);
    this->layers[39]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer16_w");
    this->layers[39]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer16_b");
    this->layers[40]->createLeakyReLULayer(this->layers[39], 0.1);
    this->layers[41]->createConvLayer(this->layers[40], 1, 1, 1024, 512, 1, 1, 0, 0, 1);
    this->layers[41]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer17_w");
    this->layers[41]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer17_b");
    this->layers[42]->createLeakyReLULayer(this->layers[41], 0.1);
    this->layers[43]->createConvLayer(this->layers[42], 3, 3, 512, 1024, 1, 1, 1, 1, 1);
    this->layers[43]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer18_w");
    this->layers[43]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer18_b");
    this->layers[44]->createLeakyReLULayer(this->layers[43], 0.1);
    this->layers[45]->createConvLayer(this->layers[44], 3, 3, 1024, 1024, 1, 1, 1, 1, 1);
    this->layers[45]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer19_w");
    this->layers[45]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer19_b");
    this->layers[46]->createLeakyReLULayer(this->layers[45], 0.1);
    this->layers[47]->createConvLayer(this->layers[46], 3, 3, 1024, 1024, 2, 2, 1, 1, 1);
    this->layers[47]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer20_w");
    this->layers[47]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer20_b");
    this->layers[48]->createLeakyReLULayer(this->layers[47], 0.1);
    this->layers[49]->createConvLayer(this->layers[48], 3, 3, 1024, 1024, 1, 1, 1, 1, 1);
    this->layers[49]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer21_w");
    this->layers[49]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer21_b");
    this->layers[50]->createLeakyReLULayer(this->layers[49], 0.1);
    this->layers[51]->createConvLayer(this->layers[50], 3, 3, 1024, 1024, 1, 1, 1, 1, 1);
    this->layers[51]->loadWeights(".\\codegen\\cnn_CnnMain_Convolution2DLayer22_w");
    this->layers[51]->loadBias(".\\codegen\\cnn_CnnMain_Convolution2DLayer22_b");
    this->layers[52]->createLeakyReLULayer(this->layers[51], 0.1);
    this->layers[53]->createFCLayer(this->layers[52], 50176, 4096);
    this->layers[53]->loadWeights(".\\codegen\\cnn_CnnMain_FullyConnectedLayer_w");
    this->layers[53]->loadBias(".\\codegen\\cnn_CnnMain_FullyConnectedLayer_b");
    this->layers[54]->createLeakyReLULayer(this->layers[53], 0.1);
    this->layers[55]->createFCLayer(this->layers[54], 4096, 1470);
    this->layers[55]->loadWeights(".\\codegen\\cnn_CnnMain_FullyConnectedLayer1_w");
    this->layers[55]->loadBias(".\\codegen\\cnn_CnnMain_FullyConnectedLayer1_b");
    this->layers[56]->createSoftmaxLayer(this->layers[55]);
    this->layers[57]->createOutputLayer(this->layers[56]);
    this->layers[57]->createWorkSpace((&this->workSpace));
    for (idx_ws = 0; idx_ws < 58; idx_ws++) {
        this->layers[idx_ws]->setWorkSpace(this->workSpace);
    }
    this->inputData = this->layers[0]->getData();
    this->outputData = this->layers[57]->getData();
}
void CnnMain::predict()
{
    int32_T idx;
    for (idx = 0; idx < 58; idx++) {
        this->layers[idx]->predict();
    }
}
void CnnMain::cleanup()
{
    int32_T idx;
    for (idx = 0; idx < 58; idx++) {
        this->layers[idx]->cleanup();
    }
    if (this->workSpace) {
        cudaFree(this->workSpace);
    }
    if (this->cublasHandle) {
        cublasDestroy(*this->cublasHandle);
    }
    if (this->cudnnHandle) {
        cudnnDestroy(*this->cudnnHandle);
    }
}
CnnMain::~CnnMain()
{
    int32_T idx;
    this->cleanup();
    for (idx = 0; idx < 58; idx++) {
        delete this->layers[idx];    }
}
