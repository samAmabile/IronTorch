#ifndef TRAIN_HPP
#define TRAIN_HPP

#include "model.hpp"
#include "tokenizer.hpp"
#include "loss.hpp" 
#include "sgd.hpp" 
#include "adam.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>



class Trainer {
private:
    Tokenizer& tokenizer; 
    //SGD& optimizer; 
    Adam optimizer;
    Loss& lossFunction;
    std::vector<std::shared_ptr<ParamaterLayer>> layers;
    

    size_t vocabSize, batchSize, seqLen, embDim, hiddenDim, kSize;
    float learningRate; 
    std::unordered_map<std::string, size_t*> paramaterMap = {
        {"vocabSize", &vocabSize},
        {"batchSize", &batchSize},
        {"seqLen", &seqLen},
        {"embDim", &embDim},
        {"hiddenDim", &hiddenDim},
        {"kernelSize", &kSize}, 
    };



public:
    Trainer(Tokenizer& t, Adam& opt, Loss& lossFunc)
        : tokenizer(t), optimizer(opt), lossFunction(lossFunc) 
    {
        
    }
    
    std::vector<uint32_t> tokenize(const std::string& text); 
    std::vector<std::string> trainNTP(const std::string& text, size_t epochs);
    void printL2ratio(std::vector<std::shared_ptr<Layer>>& layers);
    std::vector<std::string> trainClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, size_t epochs, const std::string& classA, const std::string& classB, const std::string& title, const std::string& embFile, const std::string& bpeFile);
    std::vector<std::string> trainCodeClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, size_t epochs, const std::string& classA, const std::string& classB, const std::string& title);
    void setParamater(const std::string& name, size_t value, float lr);
    


};

#endif
