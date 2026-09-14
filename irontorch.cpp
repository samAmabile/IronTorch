#include "irontorch.hpp"
#include "utils.hpp"

#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <utility>


//text loading helper
std::string loadTXT(const std::string& path) { 
    std::ifstream ifs(path); 
    if (!ifs.is_open()) throw std::runtime_error("failed to open file: " + path); 

    std::stringstream buffer; 
    buffer << ifs.rdbuf(); 
    std::string fullText = buffer.str(); 

    std::transform(fullText.begin(), fullText.end(), fullText.begin(), [](unsigned char c) {
        if (c == '\n' || c == '\r' || c == '\t') {
            return ' '; 
        }
        return static_cast<char>(c); 
    }); 

    return fullText; 

}

IronTorch::IronTorch(const std::string& type, float lr): learningRate(lr) {
    if (type == "NTP") {
        isNTP = true;
        isClassifier = false; 
    }
    optimizer = std::make_unique<Adam>(learningRate);
    trainer = std::make_unique<Trainer>(tokenizer, *optimizer, lossFunction); 
}

void IronTorch::setLearningRate(float lr) {
    optimizer->updateLearningRate(lr); 
}

void IronTorch::fitClassifier(const std::string& csvFile, const std::string& dataCol, const std::string& classCol, const std::string& classA, const std::string& classB, size_t epochs, bool isCode=false, const std::string& embFile="", const std::string& bpeFile="") {
    classMap[classB] = 0.0f; 
    classMap[classA] = 1.0f; 
    mapClass[0.0f] = classB; 
    mapClass[1.0f] = classA;
    std::vector<std::string> modelFiles;
    if (isCode) {
        modelFiles = trainer->trainCodeClassifier(csvFile, dataCol, classCol, epochs, classA, classB, "code_classifier");

    } else {
        modelFiles = trainer->trainClassifier(csvFile, dataCol, classCol, epochs, classA, classB, "classifier", embFile, bpeFile); 

    }
    cnnFilename = modelFiles[0]; 
    bpeFilename = modelFiles[1];
    
    isNTP = false;
    isFit = true;
}

void IronTorch::fitNTP(const std::string& textFile, size_t epochs) {

    std::string text = loadTXT(textFile); 

    std::vector<std::string> modelFiles = trainer->trainNTP(text, epochs);
    cnnFilename = modelFiles[0];
    bpeFilename = modelFiles[1]; 


    isNTP = true;
    isFit = true;

}

std::vector<std::string> IronTorch::fitEmbeddings(const std::string& data, size_t epochs){

    std::vector<std::string> modelFiles = trainer->trainNTP(data, epochs); 
    return {modelFiles[1], modelFiles[2]}; //{bpeFile, embeddingFile}

}

void IronTorch::load(const std::string& modelFile, const std::string& bpeFile, bool ntp) {

    isNTP = ntp; 

    cnnFilename = modelFile; 
    bpeFilename = bpeFile;

    inference = std::make_unique<Inference>(tokenizer); 
    
    inference->loadModel(cnnFilename, bpeFilename, isNTP); 
    modelLoaded = true;

    isFit = true; 
}

std::vector<float> IronTorch::predict(const std::string& text) {
    if (!isFit) throw std::runtime_error("No model trained/loaded"); 
    
    if (!modelLoaded){
        inference = std::make_unique<Inference>(tokenizer);
        inference->loadModel(cnnFilename, bpeFilename, false); 
        modelLoaded = true; 
    }
    float prediction = inference->classify(text); 
    float class_ = std::round(prediction); 

    return {class_, prediction}; 
}

std::vector<std::vector<float>> IronTorch::predict(std::vector<std::string>& data) {
    std::vector<std::vector<float>> predictions; 
    predictions.reserve(data.size()); 
    for (const std::string& sample : data){ 
        predictions.emplace_back(predict(sample)); 
    }
    return predictions;
}

float IronTorch::calcConf(float prediction) {

    float confidence = std::abs(prediction - 0.5) / 0.5; 

    return confidence;

}

std::pair<std::string, float> IronTorch::classify(const std::string& text){
    std::vector<float> stats = predict(text); 
    std::string label = mapClass[stats[0]]; 
    float confidence = calcConf(stats[1]); 
    std::pair<std::string, float> ret = {label, confidence}; 

    return ret; 
}

std::string IronTorch::generate(const std::string& seedText, size_t rounds) {
    if (!isFit) throw std::runtime_error("No model trained/loaded"); 
    
    if (!modelLoaded){
        inference = std::make_unique<Inference>(tokenizer); 
        inference->loadModel(cnnFilename, bpeFilename, true);
        modelLoaded = true; 
    }

    std::string output = inference->generate(seedText, rounds); 

    return output; 
}


























        






