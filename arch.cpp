#include "arch.hpp"
#include "conv1d.hpp"
#include "doubleConv1d.hpp"
#include "relu.hpp"
#include "pool.hpp"
#include "dense.hpp"
#include "softmax.hpp"
#include "sigmoid.hpp"
#include "dropout.hpp"

namespace Architecture{

    void initNTPLayers(Model& m, size_t vSize, size_t batchSize, size_t seqLen, size_t embDim, size_t hidden, size_t kernelSize){
        auto conv1 = std::make_shared<Conv1d>(embDim, hidden, kernelSize, batchSize, seqLen); 
        auto relu1 = std::make_shared<ReLU>(batchSize*seqLen*hidden);
        auto conv2 = std::make_shared<Conv1d>(hidden, hidden, kernelSize, batchSize, seqLen); 
        auto relu2 = std::make_shared<ReLU>(batchSize*seqLen*hidden);
        auto dense1 = std::make_shared<Dense>(hidden, hidden);
        auto relu3 = std::make_shared<ReLU>(batchSize*seqLen*hidden); 
        auto dense2 = std::make_shared<Dense>(hidden, vSize);
        auto softmax = std::make_shared<Softmax>();

        m.add(conv1); 
        m.add(relu1); 
        m.add(conv2);
        m.add(relu2);
        m.add(dense1); 
        m.add(relu3); 
        m.add(dense2);
        m.add(softmax);

    }

    void initClassifierLayers(Model& m, size_t batchSize, size_t seqLen, size_t embDim, size_t hidden, size_t kernelSize, bool isTraining){
        //auto conv1 = std::make_shared<Conv1d>(embDim, hidden, kernelSize, batchSize, seqLen);
        auto doubleConv = std::make_shared<DoubleConv1d>(embDim, hidden, kernelSize, kernelSize+2, batchSize, seqLen);
        auto relu1 = std::make_shared<ReLU>(batchSize*seqLen);
        //auto conv2 = std::make_shared<Conv1d>(hidden, hidden, kernelSize, batchSize, seqLen);
        //auto relu2 = std::make_shared<ReLU>(batchSize*hidden);
        //auto conv3 = std::make_shared<Conv1d>(hidden, hidden, kernelSize+4, batchSize, seqLen);
        //relu3 = std::make_shared<ReLU>(batchSize * seqLen); 
        auto pooling = std::make_shared<Pooling>(batchSize, seqLen, hidden*2);
        //auto dropout1 = std::make_shared<Dropout>(0.5f, isTraining);
        auto dense1 = std::make_shared<Dense>(hidden*4, hidden);
        auto relu4 = std::make_shared<ReLU>(batchSize*(hidden));
        //auto dropout2 = std::make_shared<Dropout>(0.3f, isTraining);
        auto dense2 = std::make_shared<Dense>(hidden, 1); 
        auto sigmoid = std::make_shared<Sigmoid>();

        m.add(doubleConv);
        //m.add(conv1);
        m.add(relu1);
        //m.add(conv2);
        //m.add(relu2);
        //m.add(conv3);
        //m.add(relu3);
        m.add(pooling);
        //m.add(dropout1);
        m.add(dense1);
        m.add(relu4); 
        //m.add(dropout2);
        m.add(dense2);
        m.add(sigmoid);
    }
}
