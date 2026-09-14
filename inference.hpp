#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <memory>
#include <cstdint>
#include <vector>
#include <fstream>

#include "embedding.hpp"
#include "model.hpp"
#include "matrix.hpp"
#include "tokenizer.hpp"


struct ModelConfig {
    size_t vocabSize, embDim, hiddenDim, seqLen, kernelSize; 
    
    ModelConfig():
        vocabSize(0),
        embDim(0),
        hiddenDim(0),
        seqLen(0),
        kernelSize(0) {}
    ModelConfig(size_t vSize, size_t eDim, size_t hDim, size_t sLen, size_t kSize):
        vocabSize(vSize),
        embDim(eDim), 
        hiddenDim(hDim),
        seqLen(sLen),
        kernelSize(kSize) {}
};

class Inference{
private:

    Tokenizer& tokenizer; 
    
    bool ready = false; 
    std::shared_ptr<Embedding> embedding; 
    std::unique_ptr<Model> model; 

    ModelConfig config; 

    std::vector<uint32_t> frame(std::vector<uint32_t>& tokens, size_t seqLen); 
    std::vector<uint32_t> getTail(std::vector<uint32_t>& tokens, size_t n); 
    Matrix fillRow(Matrix& m, size_t row, float val); 
    uint32_t argmaxRow(const Matrix& m, size_t row);
    ModelConfig loadConfig(std::ifstream& ifs); 
    void loadTokenizer(const std::string& filename);
    bool NTP = false;


public:
    Inference(Tokenizer& t) : tokenizer(t) {}

    void loadModel(const std::string& modelFile, const std::string& bpeFile, bool isNTP=false); 
    float classify(const std::string& text); 
    std::string generate(const std::string& seedText, size_t rounds); 



};

#endif

