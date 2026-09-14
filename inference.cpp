#include "inference.hpp" 
#include "train.hpp" 
#include "arch.hpp" 

//Helper functions: 
std::vector<uint32_t> Inference::frame(std::vector<uint32_t>& tokens, size_t seqLen) {
    std::vector<uint32_t> padded(seqLen, 0); 

    size_t n = tokens.size();

    for (size_t i{0}; i < seqLen; ++i) {
        if (i < n) {
            padded[i] = tokens[i]; 
        }
    }

    return padded; 
}

std::vector<uint32_t> Inference::getTail(std::vector<uint32_t>& tokens, size_t n) {
    size_t size = std::min(n, tokens.size()); 
    size_t source = tokens.size() - size;
    size_t dest = n - size;
    size_t startIndex = (size - n) > 0 ? startIndex : 0;  

    std::vector<uint32_t> padded(n, 0); 
 
    for (size_t i = 0; i < size; ++i) {
        padded[dest + i] = tokens[source + i]; 
    }

    return padded; 
}

Matrix Inference::fillRow(Matrix& m, size_t row, float val) {
    size_t cols = m.getCols(); 

    for (size_t i{0}; i < cols; ++i) {
        m(row, i) = val; 
    }

    return m; 
}

uint32_t Inference::argmaxRow(const Matrix& m, size_t row) {

    float max = -1e9f; 
    uint32_t maxID = 0; 
    size_t cols = m.getCols(); 

    for (size_t i{0}; i < cols; ++i) {
        if (m(row, i) > max) {
            max = m(row, i); 
            maxID = static_cast<uint32_t>(i); 
        }
    }

    return maxID; 
}

ModelConfig Inference::loadConfig(std::ifstream& ifs) {
    uint32_t magicVal = 0x4D444C31;  //"MDL1"
    uint32_t magic; 
    size_t vocabSize, embDim, hiddenDim, seqLen, kernelSize; 

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic != magicVal) {
        throw std::runtime_error("Invalid Model File: Mismatched or Missing hash"); 
    }

    ifs.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize)); 
    ifs.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
    ifs.read(reinterpret_cast<char*>(&hiddenDim), sizeof(hiddenDim));
    ifs.read(reinterpret_cast<char*>(&seqLen), sizeof(seqLen));
    ifs.read(reinterpret_cast<char*>(&kernelSize), sizeof(kernelSize));

    ModelConfig config(vocabSize, embDim, hiddenDim, seqLen, kernelSize); 

    return config;
    
}

void Inference::loadTokenizer(const std::string& filename) {

    tokenizer.loadRules(filename); 

}

//primary functions:

void Inference::loadModel(const std::string& modelFile, const std::string& bpeFile, bool isNTP) {

    std::ifstream ifs(modelFile, std::ios::binary);
    
    if (!ifs.is_open()) throw std::runtime_error("ifstream in inference::loadModel failed to open path: " + modelFile);
    config = loadConfig(ifs); 
    loadTokenizer(bpeFile); 
    NTP = isNTP; 
    
    embedding = std::make_shared<Embedding>(config.vocabSize, config.embDim); 
    model = std::make_unique<Model>(embedding, 1, config.seqLen, config.hiddenDim, config.vocabSize, config.embDim);

    if (NTP){ 
        Architecture::initNTPLayers(*model, config.vocabSize, 1, config.seqLen, config.embDim, config.hiddenDim, config.kernelSize);
    } else {
        bool isTraining = false;
        Architecture::initClassifierLayers(*model, 1, config.seqLen, config.embDim, config.hiddenDim, config.kernelSize, isTraining); 
    }

    embedding->load(ifs); 
    for (auto& layer : model->getLayers()) {
        layer->load(ifs); 
    }

    ready = true;

    ifs.close();

}

float Inference::classify(const std::string& text) {
    if (NTP) throw std::runtime_error("Model is NTP, cannot classify"); 
    if (!ready) throw std::runtime_error("No model weights have been loaded"); 

    std::vector<uint32_t> tokens = tokenizer.tokenize(text); 
    std::vector<uint32_t> windowed = frame(tokens, config.seqLen); 

    Matrix inputMat = Matrix(1, config.seqLen); 
    inputMat.convert(windowed, 1, config.seqLen);  
    Matrix prediction; 
    model->forward(inputMat, prediction); 

    return prediction(0, 0); 

}

std::string Inference::generate(const std::string& seedText, size_t rounds) {

    if (!NTP) throw std::runtime_error("Model is classifier, cannot run NTP"); 
    if (!ready) throw std::runtime_error("No model weights have been loaded");

    std::vector<uint32_t> context = tokenizer.tokenize(seedText);

    for (auto i = 0; i < rounds; ++i) {
        std::vector<uint32_t> windowed = getTail(context, config.seqLen); 

        Matrix inputMat = Matrix(1, config.seqLen); 
        inputMat.convert(windowed, 1, config.seqLen); 
        Matrix output; 
        model->forward(inputMat, output); 

        size_t lastRow = config.seqLen - 1; 

        uint32_t nextToken = argmaxRow(output, lastRow); 
        context.push_back(nextToken); 

    }

    return tokenizer.decode(context); 
}

