#include "tokenizer.hpp"
#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>

namespace Tools = Utils; 

std::vector<std::string> Tokenizer::readcsv(const std::string& filename, const std::string& column){
    CSVTable table = Tools::parse_csv(filename);

    if (table.empty()) return {""}; 

    std::vector<std::string>& header = table[0]; 
    int colNum = -1; 
    for (size_t i{0}; i < header.size(); i++){
        if (header[i] == column){
            colNum = i;
            break; 
        }
    }
    if (colNum == -1){ 
        throw std::runtime_error("Column '" + column + "' not found in CSV.");
    }

    std::vector<std::string> columnContents; 
    for (size_t i{1}; i < table.size(); i++){
        if (colNum < table[i].size()){
            columnContents.push_back(table[i][colNum]); 
        }
    }

    return columnContents; 

}

std::string Tokenizer::readtxt(const std::string& filename){ 
    std::ifstream infile(filename); 
    if (!infile.is_open()) throw std::runtime_error("failed to open file: " + filename);
    
    std::stringstream buffer; 
    buffer << infile.rdbuf();

    std::string content = buffer.str(); 

    return content; 
}

std::vector<Script> Tokenizer::fitClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, const std::string& classA="LLM", const std::string& classB="HUMAN", const std::string& title="Classifier", const std::string& bpeFile=""){
    std::vector<std::string> data = readcsv(filename, dataColumn); 
    std::vector<std::string> labels = readcsv(filename, classColumn);
    //std::string zeroLabel = labels[150];
    std::vector<Script> labeledScripts;
    assert(data.size() == labels.size() && "data size not matched to labels size");
    
    
    /*for (int i = 0; i < data.size(); ++i){
        std::string line; 
        {
            std::istringstream ss(data[i]); 
            std::getline(ss, line); 
        }
        std::cout << labels[i] << ": " << line << std::endl; 
    }*/


    std::vector<std::string> dataClassA; 
    std::vector<std::string> dataClassB; 
    size_t sampleLimit = Tools::floorPowerOfTwo(static_cast<size_t>(data.size()) / 2);
    size_t countA = 0; 
    size_t countB = 0; 
    for (size_t i{0}; i < data.size(); ++i){
        if (labels[i] == classA && countA < sampleLimit){
            dataClassA.push_back(data[i]); 
            countA++;
        }else if (labels[i] == classB && countB < sampleLimit){
            dataClassB.push_back(data[i]);
            countB++;
        }
    }
    
    std::string timestamp = Tools::makeTimestamp(); 
    std::string saveAs = "bpe_models/" + title + timestamp + ".bin";

    if (bpeFile != ""){
        bpe.load(bpeFile); 
        bpeFilename = bpeFile;
        bpe.save(saveAs);
    } else {
        std::string masterData = "";

        size_t bpeFitLimit = sampleLimit < 20000 ? sampleLimit : sampleLimit / 4; 
        for (size_t i{0}; i < bpeFitLimit; i++){
            masterData += dataClassA[i]; 
            masterData += dataClassB[i];
        }


        //bpe.initVocab();
        //bpe.fitText(masterData);
        bpe.fitWords(masterData);
        bpe.save(saveAs);
        bpeFilename = saveAs;
    }
    
    size_t tokenLimit = 1024;
    labeledScripts.reserve(2*sampleLimit);
    for (size_t i{0}; i < sampleLimit; i++){
        std::vector<uint32_t> fullTokenSetA = bpe.tokenize(dataClassA[i]);
        std::vector<uint32_t> fullTokenSetB = bpe.tokenize(dataClassB[i]);
        std::vector<uint32_t> tokenSetA(tokenLimit, 0); 
        std::vector<uint32_t> tokenSetB(tokenLimit, 0); 

        for (size_t j = 0; j < tokenLimit; j++){
            if (j < fullTokenSetA.size()){
                tokenSetA[j] = fullTokenSetA[j];
            }
            if (j < fullTokenSetB.size()){
                tokenSetB[j] = fullTokenSetB[j];
            }

        }
        
        Script labeledScriptA(tokenSetA, 1.0f);
        Script labeledScriptB(tokenSetB, 0.0f);
        labeledScripts.push_back(labeledScriptA); 
        labeledScripts.push_back(labeledScriptB);
    }

    return labeledScripts;
}


std::vector<Script> Tokenizer::fitCodeClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, const std::string& classA="LLM", const std::string& classB="HUMAN", const std::string& title="Classifier"){
    std::vector<std::string> data = readcsv(filename, dataColumn); 
    std::vector<std::string> labels = readcsv(filename, classColumn);
    //std::string zeroLabel = labels[150];
    std::vector<Script> labeledScripts;
    assert(data.size() == labels.size() && "data size not matched to labels size");

    std::vector<std::string> dataClassA; 
    std::vector<std::string> dataClassB; 
    size_t sampleLimit = 2048;
    size_t countA = 0; 
    size_t countB = 0; 
    for (size_t i{0}; i < data.size(); ++i){
        if (labels[i] == classA && countA < sampleLimit){
            dataClassA.push_back(data[i]); 
            countA++;
        }else if (labels[i] == classB && countB < sampleLimit){
            dataClassB.push_back(data[i]);
            countB++;
        }
    }

    std::string masterData = "";
    for (size_t i{0}; i < sampleLimit/4; i++){
        masterData += dataClassA[i]; 
        masterData += dataClassB[i];
    }

    std::string timestamp = Tools::makeTimestamp(); 
    std::string saveAs = "bpe_models/code_" + title + timestamp + ".bin";

    bpe.fitCode(masterData);
    bpe.save(saveAs);
    bpeFilename = saveAs; 
    
    size_t tokenLimit = 256;
    labeledScripts.reserve(2*sampleLimit);
    for (size_t i{0}; i < sampleLimit; i++){
        std::vector<uint32_t> fullTokenSetA = bpe.tokenize(dataClassA[i]);
        std::vector<uint32_t> fullTokenSetB = bpe.tokenize(dataClassB[i]);
        std::vector<uint32_t> tokenSetA(tokenLimit, 0); 
        std::vector<uint32_t> tokenSetB(tokenLimit, 0); 

        for (size_t j = 0; j < tokenLimit; j++){
            if (j < fullTokenSetA.size()){
                tokenSetA[j] = fullTokenSetA[j];
            }
            if (j < fullTokenSetB.size()){
                tokenSetB[j] = fullTokenSetB[j];
            }

        }
        
        Script labeledScriptA(tokenSetA, 1.0f);
        Script labeledScriptB(tokenSetB, 0.0f);
        labeledScripts.push_back(labeledScriptA); 
        labeledScripts.push_back(labeledScriptB);
    }

    return labeledScripts;
}

void Tokenizer::fitFile(const std::string& corpusFile, const std::string& saveAs){
    std::string text = readtxt(corpusFile); 
    //bpe.initVocab(); 
    bpe.fitWords(text); 
    bpe.save(saveAs);
    bpeFilename = saveAs;
}

void Tokenizer::fit(const std::string& text, const std::string& saveAs){
    bpe.fitWords(text); 
    bpe.save(saveAs);
    bpeFilename = saveAs;
}

void Tokenizer::loadRules(const std::string& rulesFile){
    bpe.load(rulesFile); 
}

std::vector<uint32_t> Tokenizer::tokenize(const std::string& text){

    if (bpe.getVocabSize() < 256){
        throw std::runtime_error("Error: BPE has not been initialized");
    }
    return bpe.tokenize(text); 
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens){ 
    return bpe.decode(tokens); 
}

    







    


