#ifndef MODEL_HPP
#define MODEL_HPP

#include "embedding.hpp"
#include "layer.hpp"

#include <vector> 
#include <memory> 

class Model {
private: 
    std::shared_ptr<Embedding> embeddingLayer;
    std::vector<std::shared_ptr<Layer>> layers; 
    //std::vector<std::shared_ptr<Layer>> baseLayers;
    Matrix buffA; 
    Matrix buffB; 
    Matrix gradBuffA; 
    Matrix gradBuffB;
    Matrix finalGrad; 

public: 

    Model(std::shared_ptr<Embedding> embLayer, size_t batchSize, size_t seqLen, size_t hiddenDim, size_t vSize, size_t embDim)
        : embeddingLayer(embLayer), 
        buffA(batchSize*seqLen, hiddenDim), 
        buffB(batchSize*seqLen, hiddenDim), 
        gradBuffA(batchSize*seqLen, hiddenDim), 
        gradBuffB(batchSize*seqLen, hiddenDim), 
        finalGrad(vSize, embDim){}

    void add(std::shared_ptr<Layer> layer) { layers.push_back(layer); }

    void setLayers(std::vector<std::shared_ptr<Layer>> layerlist) { layers = layerlist; }

    std::vector<std::shared_ptr<Layer>> getLayers() { return layers; }

    void forward(const Matrix& tokens, Matrix& output);


    void backward(const Matrix& gradOut);

    void saveModel(const std::string& filename, Embedding& emb, std::vector<std::shared_ptr<Layer>>& layers, size_t vocabSize, size_t embDim, size_t hiddenDim, size_t seqLen, size_t kernelSize); 
    void loadModel(const std::string& fileame, Embedding& emb, std::vector<std::shared_ptr<Layer>>& layers);

    void saveEmbeddings(const std::string& filename, Embedding& emb); 
    std::shared_ptr<Embedding> loadEmbeddings(const std::string& filename); 

    float getL2(Matrix& m); 
    float getSTD(Matrix& m); 



}; 

#endif
