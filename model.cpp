#include "model.hpp"
#include <algorithm>
#include <iostream>

void Model::forward(const Matrix& tokens, Matrix& output){ 
    //std::cout << "Input shape: " << tokens.getRows() << "x" << tokens.getCols() << std::endl;
    embeddingLayer->forward(tokens, buffA); 

    Matrix* curIn = &buffA; 
    Matrix* curOut = &buffB;

    for (size_t i{0}; i < layers.size(); ++i) { 
        
        if (i == layers.size() - 1){
            layers[i]->forward(*curIn, output);
            if (output.detectNaN()){
                std::cerr << "ERROR : Nan Detected in forward layer " << i << std::endl;
                exit(1);
            }
        }else{
            layers[i]->forward(*curIn, *curOut);

            if ((*curOut).detectNaN()){
                std::cerr << "ERROR : Nan Detected in forward layer " << i << std::endl;
                exit(1);
            }

            std::swap(curIn, curOut);
        }

    }
    
    //output.copyFrom(*curIn); 

}

void Model::backward(const Matrix& gradOut){

    Matrix* curGrad = const_cast<Matrix*>(&gradOut);
    Matrix* nextGrad = &gradBuffA; 
    
    for (int i = layers.size()-1; i>=0; --i){
        layers[i]->backward(*curGrad, *nextGrad, 0.0f); 


        if ((*nextGrad).detectNaN()){
            std::cerr << "ERROR : Nan Detected in backward layer " << i << std::endl;
            exit(1);
        }
        
        std::swap(curGrad, nextGrad);

        //nextGrad = (nextGrad == &gradBuffA) ? &gradBuffB : &gradBuffA;

    }
    embeddingLayer->backward(*curGrad, finalGrad, 0.0f);
}

void Model::saveModel(const std::string& filename, Embedding& emb, std::vector<std::shared_ptr<Layer>>& layers, size_t vocabSize, size_t embDim, size_t hiddenDim, size_t seqLen, size_t kernelSize) {
    std::ofstream ofs(filename, std::ios::binary); 
    uint32_t magic = 0x4D444C31;  //"MDL1"

    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize)); 
    ofs.write(reinterpret_cast<const char*>(&embDim), sizeof(embDim));
    ofs.write(reinterpret_cast<const char*>(&hiddenDim), sizeof(hiddenDim));
    ofs.write(reinterpret_cast<const char*>(&seqLen), sizeof(seqLen));
    ofs.write(reinterpret_cast<const char*>(&kernelSize), sizeof(kernelSize));

    emb.save(ofs); 
    for (auto& layer : layers) layer->save(ofs); 
}

void Model::loadModel(const std::string& filename, Embedding& emb, std::vector<std::shared_ptr<Layer>>& layers) {
    std::ifstream ifs(filename, std::ios::binary); 
    uint32_t magic; 
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic)); 
    if (magic != 0x4D444C31) throw std::runtime_error("Invalid model file"); 

    size_t vocabSize, embDim, hiddenDim, seqLen, kernelSize; 
    ifs.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    ifs.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
    ifs.read(reinterpret_cast<char*>(&hiddenDim), sizeof(hiddenDim));
    ifs.read(reinterpret_cast<char*>(&seqLen), sizeof(seqLen));
    ifs.read(reinterpret_cast<char*>(&kernelSize), sizeof(kernelSize));

    emb.load(ifs); 
    for (auto& layer: layers) layer->load(ifs); 
}

void Model::saveEmbeddings(const std::string& filename, Embedding& emb) {
    std::ofstream ofs(filename, std::ios::binary); 
    emb.getWeights().save(ofs);
}

std::shared_ptr<Embedding> Model::loadEmbeddings(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary); 
    if (!ifs) throw std::runtime_error("Failed to open embedding file: " + filename);
    auto emb = std::make_shared<Embedding>(); 
    emb->load(ifs); 
    return emb; 
}



float Model::getL2(Matrix& m) {
    return m.L2(); 
}

float Model::getSTD(Matrix& m) {
    return m.std(); 
}



