#include "train.hpp" 
//#include "conv1d.hpp"
//#include "relu.hpp"
//#include "dense.hpp"
#include "batcher.hpp"
//#include "pool.hpp"
//#include "sigmoid.hpp"
//#include "softmax.hpp"
#include "arch.hpp"
#include "paramaterlayer.hpp"

#include <chrono>
#include <memory>
#include <ctime> 
#include <iomanip>
#include <iostream>
#include <fstream> 
#include <sstream> 
#include <random> 
#include <algorithm>
#include <cassert>
#include <cmath>

//TODO: refactor training loops to eliminate redundancy
//helper functions to create: 
// - save
// - runBatch
// - validateBatch 



std::string makeTimestamp() {
    auto now = std::chrono::system_clock::now(); 

    std::time_t timeT = std::chrono::system_clock::to_time_t(now);

    std::tm localTime; 
#if defined(_WIN32)
    localtime_s(&localTime, &timeT); 
#else
    localtime_r(&timeT, &localTime); 
#endif 

    std::stringstream ss; 
    ss << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");

    return ss.str(); 
}

std::string save(Model model, std::shared_ptr<Embedding> emb, size_t vocabSize, size_t embDim, size_t hiddenDim, size_t seqLen, size_t kernelSize, const std::string& label){
    std::string timeStamp = makeTimestamp(); 
    std::string modelName = "cnn_models/" + label + "_" + timeStamp + ".bin"; 
    std::vector<std::shared_ptr<Layer>> modelLayers = model.getLayers();
    model.saveModel(modelName, *emb, modelLayers, vocabSize, embDim, hiddenDim, seqLen, kernelSize);

    return modelName;
}

std::string saveLoss(std::vector<float>& training, std::vector<float>& validation, size_t epochs, size_t numBatches, size_t numValBatches){
    std::string timeStamp = makeTimestamp(); 
    std::string filename = "data/training_validation_" + timeStamp + ".csv"; 
    std::ofstream ofs(filename); 

    //push header onto csv:
    ofs << "epoch,trainLosses,validationLosses,meanTrain,meanValidation\n";
    
    size_t TID = 0; 
    size_t VID = 0; 
    
    //for each row
    for (size_t e = 0; e < epochs; ++e){
        //column 0:
        if (TID >= training.size() && VID >= validation.size()) break;
        ofs << e << ","; 
        float batchSum = 0; 
        size_t trainCount = 0; 
        for (size_t b = 0; b < numBatches && TID < training.size(); ++b){
            //column 1:
            ofs << training[TID] << " "; 
            batchSum += training[TID]; 
            TID++;
            trainCount++;
        }
        ofs << ",";
        float valSum = 0; 
        size_t valCount = 0;
        for (size_t vb = 0; vb < numValBatches && VID < validation.size(); ++vb){
            //column 2:
            ofs << validation[VID] << " "; 
            valSum += validation[VID]; 
            VID++; 
            valCount++;
        }

        ofs << ","; 
        //column 3:
        ofs << (trainCount > 0 ? batchSum / static_cast<float>(trainCount) : 0.0f); 
        ofs << ","; 
        //column 4:
        ofs << (valCount > 0 ? valSum / static_cast<float>(valCount) : 0.0f); 
        ofs << "\n";
    }

    std::cout << "\nTraining/Validation losses saved to: " << filename << std::endl;

    ofs.close();

    return filename; 
}

void makeTarget(const std::vector<uint32_t>& targets, size_t vocabSize, size_t numRows, Matrix& targetMatrix){

    if (targets.size() != numRows){
        throw std::runtime_error("Target vector must match matrix rows");
    }

    targetMatrix.setDimensions(numRows, vocabSize);
    targetMatrix.fill(0.0f); 

    for (size_t i{0}; i < numRows; ++i) {
        uint32_t tokenID = targets[i]; 
        if (tokenID < vocabSize){
            targetMatrix(i, tokenID) = 1.0f;
        }
    }

}
void makeClassifierTarget(std::vector<float>& targetFloats, Matrix& target){
    target.fillData(targetFloats);
}
void progressBar(size_t current, size_t total, const std::string& label, float loss, float lr=0.01f) {
    const int barWidth = 40; 
    float progress = (float)current / total;
    int pos = (int)(barWidth * progress); 

    std::cout << "\r" << label << " LR: " << lr << " [";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "#"; 
        else if (i == pos) std::cout << ">"; 
        else std::cout << " "; 
    }
    std::cout << "] " << int(progress * 100.0) << "% " << std::to_string(loss) << std::flush; 
}

void Trainer::printL2ratio(std::vector<std::shared_ptr<Layer>>& layers) {

    std::vector<float> ratios; 

    for (auto& layer : layers) {
        auto pLayer = dynamic_cast<ParamaterLayer*>(layer.get()); 
        if (pLayer){ 
            float weight =  pLayer->getWeights().L2(); 
            float gradient = pLayer->getGradient().L2();
            float ratio = (weight > 0.0f) ? (gradient / weight) : 0.0f;
            ratios.push_back(ratio); 
        }
    }

    std::cout << "\nGrad : Wieght Ratio " << std::endl; 
    for (int i = 0; i < ratios.size(); ++i) {
        std::cout << "Ratio " << i << ": " << ratios[i] << std::endl; 
    }

}

std::vector<std::string> Trainer::trainClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, size_t epochs=300, const std::string& classA="LLM", const std::string& classB="HUMAN", const std::string& title="Classifier", const std::string& embFile="", const std::string& bpeFile=""){
    //load data:

    std::vector<Script> fullDataset = tokenizer.fitClassifier(filename, dataColumn, classColumn, classA, classB, title, bpeFile);

    std::string bpeFilename = tokenizer.getFilename();
    
    //random device:
    std::random_device rd; 
    std::mt19937 generator(rd());
    std::shuffle(fullDataset.begin(), fullDataset.end(), generator);

    //train/validation split:
    size_t datasetSize = fullDataset.size(); 
    size_t trainSize = static_cast<size_t>(datasetSize * 0.8); 
    
    std::vector<Script> dataset(fullDataset.begin(), fullDataset.begin() + trainSize); 
    std::vector<Script> testSet(fullDataset.begin() + trainSize, fullDataset.end());
    
    //set params:
    vocabSize = tokenizer.getVocabSize();//should be 5000
    seqLen = dataset[0].tokens.size(); //should be 1024
    embDim = 256; 
    batchSize = 32;
    hiddenDim = 128; 
    kSize = 5; //kernel size
    Batcher batcher(batchSize, seqLen); //32 x 128

    //make layers:
    auto emb = std::make_shared<Embedding>(vocabSize, embDim);//(5000, 128)
                                                              //
    std::cout << "SeqLen: " << seqLen << std::endl;
    std::cout << "vocabSize: " << vocabSize << std::endl;

                                                              
    //model utility:
    Model model(emb, batchSize, seqLen, hiddenDim, vocabSize, embDim);
    if (embFile != ""){
        emb = model.loadEmbeddings(embFile); 
    }
    std::string modelName = "";
    
    //config model:
    bool isTraining = true;
    Architecture::initClassifierLayers(model, batchSize, seqLen, embDim, hiddenDim, kSize, isTraining);
    Matrix batch(batchSize, seqLen); // should be 32 x 128
    Matrix targets(batchSize, 1);    // should be 32 x 1 
    Matrix prediction(batchSize, 1); // ''
    Matrix lossGrad(batchSize, 1);   // '' 
    std::vector<float> targetVec(batchSize, 0.0f); // 32 x 1
    size_t numBatches = dataset.size() / batchSize;

    Matrix testBatch(batchSize, seqLen); // should be 32 x 128
    Matrix testTargets(batchSize, 1);    // should be 32 x 1 
    Matrix verification(batchSize, 1); // ''
    std::vector<float> testTargetVec(batchSize, 0.0f); // 32 x 1
    size_t numTestBatches = testSet.size() / batchSize;

    float loss = 0.0f; 
    size_t startIndex = 0; 
    float bestValLoss = std::numeric_limits<float>::max();
    size_t epochsSinceImprovement = 0; 
    size_t patience = 4; 
    size_t cooldown = 4; 
    size_t cooldownCounter = 0;
    size_t updates = 0; 
    float lr = 0.001f;
    float minLR = 0.0001f; 
    float minDelta = 1e-4f;
    float lrFactor = 0.5f; 
    std::vector<float> meanLosses(epochs, 0.0f); 
    
    std::vector<float> trainingLosses; 
    trainingLosses.reserve(epochs * numBatches);
    std::vector<float> validationLosses; 
    validationLosses.reserve(epochs * numTestBatches);

    std::cout << "NumBatches: " << numBatches << std::endl;
    for (size_t e{0}; e < epochs; ++e){

        std::shuffle(dataset.begin(), dataset.end(), generator); 
        
        for (size_t b = 0; b < numBatches; b++){ 

            startIndex = b * batchSize; //should be 0, 32, 64, 96 ... 
            
            if (startIndex + batchSize > dataset.size()) break;

            batcher.fillBatchMatrix(dataset, startIndex, embDim, batch);
            for (size_t i{0}; i < batchSize; ++i){
                targetVec[i] = dataset[startIndex + i].label; 
            }
            makeClassifierTarget(targetVec, targets); 

            //Matrix targetsT = targets.transpose(); 
            
            model.forward(batch, prediction);
            lossFunction.calcCrossEntropyGrad(prediction, targets, lossGrad);

            model.backward(lossGrad);

            //printL2ratio();
        
            optimizer.step(model.getLayers(), emb);
            //optimizer.step({emb});
            if (prediction.detectNaN()){
                std::cout << "\n ** Nan detected in prediction at epoch ** " << e << ", batch " << b << std::endl;
            }
            if (targets.detectNaN()){
                std::cout << "\n ** Nan detected in targets at epoch **" << e << ", batch " << b << std::endl;
            }

            loss = lossFunction.crossEntropy(prediction, targets);
            trainingLosses.emplace_back(loss);
            
            progressBar(b, numBatches, "Epoch " + std::to_string(e), loss, lr); 
            
        } 

        std::vector<float> valLosses;
        valLosses.reserve(numTestBatches);
        for (size_t b = 0; b < numTestBatches; ++b){
            size_t testStartIndex = b * batchSize; 
            if (testStartIndex + batchSize > testSet.size()) break; 

            batcher.fillBatchMatrix(testSet, testStartIndex, embDim, testBatch); 
            for (size_t i = 0; i < batchSize; ++i){
                testTargetVec[i] = testSet[testStartIndex + i].label; 
            }
            makeClassifierTarget(testTargetVec, testTargets); 

            model.forward(testBatch, verification); 
            float vLoss = lossFunction.crossEntropy(verification, testTargets); 
            valLosses.push_back(vLoss);

            progressBar(b, numTestBatches, "*VLoss " + std::to_string(e), vLoss, lr);
            validationLosses.emplace_back(vLoss); 
        }
        float meanLoss = std::accumulate(valLosses.begin(), valLosses.end(), 0.0f) / valLosses.size(); 
        meanLosses[e] = meanLoss; 
        
        if (meanLoss < bestValLoss - minDelta) { 
            bestValLoss = meanLoss; 
            modelName = save(model, emb, vocabSize, embDim, hiddenDim, seqLen, kSize, "classifier");
            epochsSinceImprovement = 0; 
        } else {
            if (cooldownCounter > 0){
                cooldownCounter--;
            } else {
                epochsSinceImprovement++; 
                if (epochsSinceImprovement >= patience) {
                    if ((lr * lrFactor) >= minLR) {
                        lr *= lrFactor; 
                        setParamater("learningRate", 1, lr); 
                        epochsSinceImprovement = 0; 
                        cooldownCounter = cooldown;
                    }
                    else {
                        std::cout << "\nEarly stopping at epoch " << e << ", model saved to: " << modelName << ", Tokenizer saved to: " << bpeFilename << std::endl; 
                        std::string statsFile = saveLoss(trainingLosses, validationLosses, e, numBatches, numTestBatches);

                        return {modelName, bpeFilename}; 
                    }
                }

            }
        }
    }


    std::string statsFile = saveLoss(trainingLosses, validationLosses, epochs, numBatches, numTestBatches);

    std::cout << "\nTraining complete. Model saved to: " << modelName << "Tokenizer saved to: " << bpeFilename << std::endl; 
    return {modelName, bpeFilename}; 
 
}


std::vector<std::string> Trainer::trainCodeClassifier(const std::string& filename, const std::string& dataColumn, const std::string& classColumn, size_t epochs=300, const std::string& classA="LLM", const std::string& classB="HUMAN", const std::string& title="Classifier"){
    std::vector<Script> fullDataset = tokenizer.fitCodeClassifier(filename, dataColumn, classColumn, classA, classB, title);

    std::string bpeFilename = tokenizer.getFilename();
    
    std::random_device rd; 
    std::mt19937 generator(rd());
    std::shuffle(fullDataset.begin(), fullDataset.end(), generator);

    size_t datasetSize = fullDataset.size(); 
    size_t trainSize = static_cast<size_t>(datasetSize * 0.8); 

    std::vector<Script> dataset(fullDataset.begin(), fullDataset.begin() + trainSize); 
    std::vector<Script> testSet(fullDataset.begin() + trainSize, fullDataset.end());

    vocabSize = tokenizer.getVocabSize();//should be 1000 
    seqLen = dataset[0].tokens.size(); //should be 128
    embDim = 256; 
    batchSize = 32;
    hiddenDim = 128; 
    kSize = 3; //kernel size
    Batcher batcher(batchSize, seqLen); //32 x 128
    auto emb = std::make_shared<Embedding>(vocabSize, embDim);//(5000, 128)
                                                              //
    std::cout << "SeqLen: " << seqLen << std::endl;
    std::cout << "vocabSize: " << vocabSize << std::endl;

                                                              

    Model model(emb, batchSize, seqLen, hiddenDim, vocabSize, embDim); 
    std::string modelName = "";
   
    bool isTraining = true;
    Architecture::initClassifierLayers(model, batchSize, seqLen, embDim, hiddenDim, kSize, isTraining);
    Matrix batch(batchSize, seqLen); // should be 32 x 128
    Matrix targets(batchSize, 1);    // should be 32 x 1 
    Matrix prediction(batchSize, 1); // ''
    Matrix lossGrad(batchSize, 1);   // '' 
    std::vector<float> targetVec(batchSize, 0.0f); // 32 x 1
    size_t numBatches = dataset.size() / batchSize;

    Matrix testBatch(batchSize, seqLen); // should be 32 x 128
    Matrix testTargets(batchSize, 1);    // should be 32 x 1 
    Matrix verification(batchSize, 1); // ''
    std::vector<float> testTargetVec(batchSize, 0.0f); // 32 x 1
    size_t numTestBatches = testSet.size() / batchSize;

    float loss = 0.0f; 
    size_t startIndex = 0; 
    float bestValLoss = std::numeric_limits<float>::max();
    size_t epochsSinceImprovement = 0; 
    size_t patience = 10; 
    float minDelta = 1e-4f;
    std::vector<float> meanLosses(epochs, 0.0f); 
 

    std::cout << "NumBatches: " << numBatches << std::endl;
    for (size_t e{0}; e < epochs; ++e){

        std::shuffle(dataset.begin(), dataset.end(), generator); 
        

        for (size_t b = 0; b < numBatches; b++){ 

            startIndex = b * batchSize; //should be 0, 32, 64, 96 ... 
            
            if (startIndex + batchSize > dataset.size()) break;

            batcher.fillBatchMatrix(dataset, startIndex, embDim, batch);
            for (size_t i{0}; i < batchSize; ++i){
                targetVec[i] = dataset[startIndex + i].label; 
            }
            makeClassifierTarget(targetVec, targets); 

            //Matrix targetsT = targets.transpose(); 
            
            model.forward(batch, prediction);
            lossFunction.calcCrossEntropyGrad(prediction, targets, lossGrad);

            model.backward(lossGrad);

            //printL2ratio();
        
            optimizer.step(model.getLayers(), emb);
            //optimizer.step({emb});
            
            loss = lossFunction.crossEntropy(prediction, targets);
            
            
            progressBar(b, numBatches, "Epoch " + std::to_string(e), loss); 
            
        } 

        std::vector<float> valLosses;
        valLosses.reserve(numTestBatches);
        for (size_t b = 0; b < numTestBatches; ++b){
            size_t testStartIndex = b * batchSize; 
            if (testStartIndex + batchSize > testSet.size()) break; 

            batcher.fillBatchMatrix(testSet, testStartIndex, embDim, testBatch); 
            for (size_t i = 0; i < batchSize; ++i){
                testTargetVec[i] = testSet[testStartIndex + i].label; 
            }
            makeClassifierTarget(testTargetVec, testTargets); 

            model.forward(testBatch, verification); 
            float vLoss = lossFunction.crossEntropy(verification, testTargets); 
            valLosses.push_back(vLoss);
        }
        float meanLoss = std::accumulate(valLosses.begin(), valLosses.end(), 0.0f) / valLosses.size(); 
        meanLosses[e] = meanLoss; 
        
        if (meanLoss < bestValLoss - minDelta) { 
            bestValLoss = meanLoss; 
            modelName = save(model, emb, vocabSize, embDim, hiddenDim, seqLen, kSize, "class");
            epochsSinceImprovement = 0; 
        } else {
            epochsSinceImprovement++; 
            if (epochsSinceImprovement >= patience) {
                std::cout << "Early stopping at epoch " << e << ", model saved to: " << modelName << ", Tokenizer saved to: " << bpeFilename << std::endl; 
                return {modelName, bpeFilename}; 
            }
        }
    }

    std::cout << "Training complete. Model saved to: " << modelName << "Tokenizer saved to: " << bpeFilename << std::endl; 
    return {modelName, bpeFilename}; 
 
}

std::vector<std::string> Trainer::trainNTP(const std::string& text, size_t epochs) {
    batchSize = 16;
    seqLen = 64; 

    std::string timestamp = makeTimestamp(); 
    std::string bpeFilename = "bpe_models/bpe_"+timestamp+".bin"; 
    std::string embeddingName = "embeddings/embedding_"+timestamp+".bin";

    tokenizer.fit(text, bpeFilename); 

    std::vector<uint32_t> allTokens = tokenizer.tokenize(text); 

    size_t splitIndex = static_cast<size_t>(allTokens.size() * 0.9); 
    std::vector<uint32_t> tokens(allTokens.begin(), allTokens.begin() + splitIndex); 
    std::vector<uint32_t> testTokens(allTokens.begin() + splitIndex, allTokens.end()); 

    Batcher batcher(batchSize, seqLen); 
    vocabSize = tokenizer.getVocabSize(); 
    embDim = 256; 
    hiddenDim = 128; 
    kSize = 3; 

    Matrix prediction(batchSize, seqLen);
    Matrix targets; 
    Matrix validation(batchSize, seqLen); 
    Matrix valTargets; 
    Matrix lossGrad(batchSize, 1); 

    auto emb = std::make_shared<Embedding>(vocabSize, embDim); 
    std::vector<std::vector<uint32_t>> batches = batcher.batchData(tokens); 
    std::vector<std::vector<uint32_t>> valBatches = batcher.batchData(testTokens); 

    float bestValLoss = std::numeric_limits<float>::max();
    size_t epochsSinceImprovement = 0; 
    size_t patience = 10;
    //size_t cooldown = 5; 
    //float minLR = 0.00001; 
    float minDelta = 1e-4f;


    std::random_device rd; 
    std::mt19937 generator(rd()); 
    
    Model model(emb, batchSize, seqLen, hiddenDim, vocabSize, embDim); 
    std::string modelName = "";

    Architecture::initNTPLayers(model, vocabSize, batchSize, seqLen, embDim, hiddenDim, kSize); 
    Matrix batchMat(batchSize, seqLen);
    Matrix valBatchMat(batchSize, seqLen); 
    
    std::vector<float> meanLosses;
    meanLosses.reserve(epochs);

    std::vector<float> meanValLosses; 
    meanValLosses.reserve(epochs);
    
    std::cout << "log(N) = " << std::log(vocabSize) << std::endl;
    for (int e{0}; e < epochs; ++e){
        std::shuffle(batches.begin(), batches.end(), generator); 
        //std::shuffle(valBatches.begin(), valBatches.end(), generator);
        
        std::vector<float> losses; 
        losses.reserve(batches.size());

        std::vector<float> valLosses;
        valLosses.reserve(valBatches.size());
        
        int b = 0; 
        int numBatches = batches.size();
        for (auto& batch : batches) { 

            float loss = 0.0f;
            batchMat.convert(batch, batchSize, seqLen);
            model.forward(batchMat, prediction); 

            size_t expectedRows = prediction.getRows(); 
            size_t expectedCols = prediction.getCols(); 

            std::vector<uint32_t> targetVec = batch; 
            targetVec.erase(targetVec.begin()); 
            targetVec.push_back(0); 

            makeTarget(targetVec, expectedCols, expectedRows, targets); 
            lossFunction.calcCrossEntropyGrad(prediction, targets, lossGrad); 

            model.backward(lossGrad); 
            optimizer.step(model.getLayers(), emb); 
            //optimizer.step({emb}); 

            loss += lossFunction.categoricalCrossEntropy(prediction, targets); 
            losses.emplace_back(loss); 

            progressBar(b, numBatches, "Epoch " + std::to_string(e), loss);
            b++;
        }

        for (auto& valBatch : valBatches){

            float valLoss = 0.0f; 
            valBatchMat.convert(valBatch, batchSize, seqLen);

            model.forward(valBatchMat, validation); 
            
            size_t valRows = validation.getRows(); 
            size_t valCols = validation.getCols(); 

            std::vector<uint32_t> valVec = valBatch; 
            valVec.erase(valVec.begin()); 
            valVec.push_back(0); 

            makeTarget(valVec, valCols, valRows, valTargets); 

            valLoss += lossFunction.categoricalCrossEntropy(validation, valTargets); 
            valLosses.emplace_back(valLoss); 
        }

        float meanLoss = std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
        meanLosses.emplace_back(meanLoss); 

        float meanValLoss = std::accumulate(valLosses.begin(), valLosses.end(), 0.0f) / valLosses.size();
        meanValLosses.emplace_back(meanValLoss); 
        
        //std::string embeddingName = "data/embedding_"+timestamp+".bin";
        if (meanValLoss < bestValLoss - minDelta) { 
            bestValLoss = meanValLoss; 
            modelName = save(model, emb, vocabSize, embDim, hiddenDim, seqLen, kSize, "ntp");
            model.saveEmbeddings(embeddingName, *emb);
            epochsSinceImprovement = 0; 
        } else {
            epochsSinceImprovement++; 
            if (epochsSinceImprovement >= patience) {
                std::cout << "Early stopping at epoch " << e << ", model saved to: " << modelName << ", Tokenizer saved to: " << bpeFilename << ", Embedding saved to: " << embeddingName << std::endl; 
                return {modelName, bpeFilename, embeddingName}; 
            }
        }

    }

    std::cout << "Training complete model saved to: " << modelName << ", Tokenizer saved to: " << bpeFilename << ", Embedding saved to: " << embeddingName << std::endl;
    return {modelName, bpeFilename, embeddingName};
}

void Trainer::setParamater(const std::string& name, size_t value, float lr=0.001) {
    if (name == "learningRate") {
        learningRate = lr; 
        optimizer.updateLearningRate(lr); 
        return;
    }
    auto it = paramaterMap.find(name);
    if (it == paramaterMap.end()) {
        throw std::runtime_error("Paramater: " + name + " not a valid paramater name");
    }

    size_t prev = *paramaterMap[name];
    *paramaterMap[name] = value; 

    std::cout << "updated " << name << " from: " << prev << " to: " << value << std::endl; 

}


    




















