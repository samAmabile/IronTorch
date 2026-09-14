#ifndef IRONTORCH_HPP
#define IRONTORCH_HPP

#include "tokenizer.hpp"
#include "sgd.hpp"
#include "adam.hpp"
#include "loss.hpp" 
#include "train.hpp"
#include "inference.hpp"

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

class IronTorch {
private:
    Tokenizer tokenizer;
    std::unique_ptr<Adam> optimizer; 
    Loss lossFunction; 

    std::unique_ptr<Trainer> trainer; 
    std::unique_ptr<Inference> inference;

    bool isNTP=false;
    bool isClassifier=true;
    bool isFit=false;
    bool modelLoaded=false;

    float learningRate; 
    
    std::string bpeFilename; 
    std::string cnnFilename;

    std::unordered_map<std::string, float> classMap; 
    std::unordered_map<float, std::string> mapClass; 



public:

    IronTorch(const std::string& type="classifier", float lr=0.001f);

    std::vector<std::string> parseCSV(const std::string& filename, const std::string& column) { return tokenizer.readcsv(filename, column); }

    void fitClassifier(const std::string& csvFile, const std::string& dataCol, const std::string& classCol, const std::string& classA, const std::string& classB, size_t epochs, bool isCode, const std::string& embFile, const std::string& bpeFile);
    void fitNTP(const std::string& textFile, size_t epochs);
    std::vector<std::string> fitEmbeddings(const std::string& data, size_t epochs); 

    void load(const std::string& modelFile, const std::string& bpeFile, bool isNTP=false);
    std::vector<float> predict(const std::string& text);
    std::vector<std::vector<float>> predict(std::vector<std::string>& data);
    std::pair<std::string, float> classify(const std::string& text); 
    std::string generate(const std::string& seedText, size_t rounds);
    void setLearningRate(float lr); 

    float calcConf(float prediction);

};

#endif


