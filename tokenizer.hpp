#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string> 
#include <vector> 
#include "bpe.hpp"
#include <cstdint>

struct Script {
    std::vector<std::uint32_t> tokens; 
    float label; 
    Script() : tokens({}), label(-1.0) {}
    Script(std::vector<uint32_t> t, float l) : tokens(t), label(l) {}
};


class Tokenizer { 
private: 
	BPE bpe; 
    std::vector<Script> dataset;
    std::string bpeFilename = ""; 

public: 
	Tokenizer() = default; 

	std::string readtxt(const std::string& filename); 
	std::vector<std::string> readcsv(const std::string& filename, const std::string& column);

    std::vector<Script> fitClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, const std::string& classA, const std::string& classB, const std::string& title, const std::string& bpeFile);
    std::vector<Script> fitCodeClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, const std::string& classA, const std::string& classB, const std::string& title);

    size_t getVocabSize() { return bpe.getVocabSize(); }
	
	void fitFile(const std::string& corpusFile, const std::string& saveAs);
    void fit(const std::string& text, const std::string& saveAs); 
	
	void loadRules(const std::string& rulesFile);
	std::vector<uint32_t> tokenize(const std::string& text); 
	std::string decode(const std::vector<uint32_t>& tokens); 

    std::string getFilename() { return bpeFilename; }
};

#endif



