#include "irontorch.hpp"
#include "embedding.hpp"

#include <iostream> 
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono> 
#include <ctime>
#include <iomanip>
#include <sstream>

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now(); 
    std::time_t time = std::chrono::system_clock::to_time_t(now); 
    
    std::tm timeInfo; 
#if defined(_MS_VER)
    localtime_s(&timeInfo, &time); 
#else
    localtime_r(&time, &timeInfo);
#endif

    std::ostringstream oss; 
    oss << std::put_time(&timeInfo, "%Y-%m-%d_%H-%M-%S");

    return oss.str();
}

bool endswith(const std::string& word, const std::string& suffix) {
    if (word.length() < suffix.length()){ 
        return false; 
    }

    return word.compare(word.length() - suffix.length(), suffix.length(), suffix) == 0;

}

void printDocumentation(){
    std::ifstream ifs("txt/documentation.txt"); 
    if (!ifs.is_open()) throw std::runtime_error("Failed to open documentation file");

    std::string line; 
    while (getline(ifs, line)) {
        std::cout << line << std::endl; 
    }
}

std::vector<std::string> trainTestSplit(const std::string& filename) {
    assert(endswith(filename, ".csv") && "data file must be a CSV for train-test-split");
    std::ifstream ifs(filename); 

    std::string trainFilename = ""; 
    std::string testFilename = ""; 

    for (char c: filename){
        if (c == '.') {
            trainFilename += "_train.csv"; 
            testFilename += "_test.csv"; 
            break; 
        }
        trainFilename += c; 
        testFilename += c; 
    }

    if (!ifs.is_open()) throw std::runtime_error("Unable to open file: " + filename); 

    std::string line; 
    std::vector<std::string> rows; 
    std::string header; 

    std::getline(ifs, header); 

    while (std::getline(ifs, line)) {
        rows.push_back(line); 
    }

    std::shuffle(rows.begin(), rows.end(), std::mt19937{std::random_device{}()});

    size_t trainSize = static_cast<size_t>(rows.size() * 0.9); 

    std::ofstream trainF(trainFilename); 
    
    trainF << header << "\n"; 
    for (size_t i = 0; i < trainSize; ++i){
        trainF << rows[i] << "\n";
    }
    trainF.close(); 

    std::ofstream testF(testFilename); 
    testF << header << "\n"; 
    for (size_t i = trainSize; i < rows.size(); ++i) {
        testF << rows[i] << "\n";
    }
    testF.close(); 

    return {trainFilename, testFilename}; 

}

void cli(std::string& masterFile, std::string& trainingFile, std::string& testingFile, std::string& dataColName, std::string& classColName, std::string& type) {

    std::cout << "~~~~Data Input System~~~~" << std::endl; 
    std::cout << "to avoid this CLI, compile with <filename.csv> [data_column_name] [class_column_name] [type(text/code)]" << std::endl; 
    std::cout << "compile with [help] to see full documentation" << std::endl;

    if (masterFile == ""){
        std::cout << "Enter csv file path for data: " << std::endl; 
        std::cin >> masterFile; 

        std::vector<std::string> tempFiles = trainTestSplit(masterFile); 
        trainingFile = tempFiles[0]; 
        testingFile = tempFiles[1]; 

    }
    if (dataColName == ""){
        std::cout << "Enter column name for DATA (ie 'src', 'data', 'text'): " << std::endl;
        std::cin >> dataColName; 
    }
    if (classColName == ""){
        std::cout << "Enter column name for CLASS (ie 'type', 'class', 'label'): " << std::endl;
        std::cin >> classColName; 
    }
    if (type == ""){
        std::cout << "Choose data type: [1] text, [2] code"; 
        int choice; 
        std::cin >> choice; 
        type = choice == 1 ? "text" : "code";
    }
}

void saveStats(std::vector<bool>& results, std::vector<std::string>& predictions, std::vector<float>& confidences, const std::string& saveAs){
    int totalResults = results.size(); 
    int successes = 0; 

    for (bool r : results){
        if (r) successes++; 
    }

    float successProportion = successes > 0 ? (static_cast<float>(successes) / totalResults) : 0.0f;
    float meanConfidence = 0.0f; 

    if (!confidences.empty()) {
        meanConfidence = std::accumulate(confidences.begin(), confidences.end(), 0.0f) / confidences.size();
    }

    std::ofstream ofs(saveAs); 
    
    ofs << saveAs << std::endl;
    ofs << "Results: " << "\n"; 
    ofs << "Successful tests: (" << successes << "/" << totalResults << ")" << (successProportion * 100) << "%" << std::endl; 
    ofs << "Mean Confidence: " << (meanConfidence * 100) << "%" << std::endl; 
    ofs << "Breakdown of Predictions: " << std::endl;
    
    size_t truePos = 0; 
    size_t actualPos = 0; 
    size_t falsePos = 0; 
    size_t falseNeg = 0; 
    for (size_t i = 0; i < predictions.size(); ++i) {
        ofs << "Prediction: " << predictions[i] << " | Confidence: " << confidences[i] << " | Result: " << results[i] << std::endl; 
        if (predictions[i] == "1" && results[i] == 1){
            truePos++;  
        } else if (predictions[i] == "0" && results[i] == 0){
            falseNeg++; 
        } else if (predictions[i] == "1" && results[i] == 0){
            falsePos++;
        }     
    }
    actualPos = truePos + falseNeg; 
    float precision = static_cast<float>(truePos) / static_cast<float>(truePos + falsePos); 
    float recall = static_cast<float>(truePos) / static_cast<float>(truePos + falseNeg);

    float f1 = 2 * ((precision * recall) / (precision + recall));
    
    ofs << "==========================================================" << std::endl;
    ofs << "Precision: " << precision << std::endl; 
    ofs << "Recall: " << recall << std::endl; 
    ofs << "F1: " << f1 << std::endl; 
    ofs << "==========================================================" << std::endl;

    ofs.close(); 
}

std::string concat(const std::vector<std::string>& docs) {
    std::string allDocs = ""; 
    for (const std::string& doc : docs) {
        allDocs += "\n" + doc; 
    }

    return allDocs; 
}

std::vector<std::string> makeEmbeddings(const std::vector<std::string>& docs) {
    std::string trainingData = concat(docs); 
    IronTorch model("NTP", 1e-3f); 
    
    std::vector<std::string> embFiles = model.fitEmbeddings(trainingData, 50); 

    return embFiles; 

}





int main(int argc, char* argv[]) {

    IronTorch model; 

    std::string masterFile = "";
    std::string trainingFile = ""; 
    std::string testingFile = ""; 
    std::string dataColName = ""; 
    std::string classColName = ""; 

    std::string type = ""; 
    bool preTrainEmbeddings = false;
    
    switch (argc) {
        case 1: 
            printDocumentation();
            cli(masterFile, trainingFile, testingFile, dataColName, classColName, type);
            break;
        case 2: {
            if (std::string(argv[1]) == "help"){
                printDocumentation();
                return 0;
            } else {
                masterFile = argv[1]; 
                std::vector<std::string> tempFiles = trainTestSplit(masterFile);
                trainingFile = tempFiles[0]; 
                testingFile = tempFiles[1];
                cli(masterFile, trainingFile, testingFile, dataColName, classColName, type); 
                break;
            }
        }
        case 3:
            masterFile = argv[1]; 
            dataColName = argv[2]; 
            cli(masterFile, trainingFile, testingFile, dataColName, classColName, type); 
            break;
        case 4:
            masterFile = argv[1]; 
            dataColName = argv[2]; 
            classColName = argv[3]; 
            cli(masterFile, trainingFile, testingFile, dataColName, classColName, type); 
            break;
        case 5:{
            masterFile = argv[1]; 
            dataColName = argv[2]; 
            classColName = argv[3]; 
            type = argv[4];
            std::vector<std::string> tempFiles = trainTestSplit(masterFile); 
            trainingFile = tempFiles[0]; 
            testingFile = tempFiles[1];  
        }
        case 6:
            if (argc >= 6 && std::string(argv[5]) == "pretrain"){
                preTrainEmbeddings = true;
            }
    }

    
    bool isCode = type == "code" ? true : false; 

    std::vector<std::string> labels = model.parseCSV(trainingFile, classColName);
    std::vector<std::string> docs = model.parseCSV(trainingFile, dataColName); 
    std::string classA = labels[0]; 
    std::string classB = ""; 
    for (std::string label : labels) {
        if (label != classA){ 
            classB = label; 
            break;
        }
    }

    std::string embFile = ""; 
    std::string bpeFile = ""; 

    //setup if classifier classify(); if ntp ntp(); ?
    
    ///*
    if (preTrainEmbeddings){
        std::cout << "Pre-Training Embeddings..." << std::endl; 
        std::vector<std::string> embedFiles = makeEmbeddings(docs);
        embFile = embedFiles[1]; 
        bpeFile = embedFiles[0]; 
        std::cout << "Embeddings trained!" << std::endl;
    }
    //*/

    model.fitClassifier(trainingFile, dataColName, classColName, classA, classB, 100, isCode, embFile, bpeFile); 
    
    std::vector<std::string> testData = model.parseCSV(testingFile, dataColName); 
    std::vector<std::string> testLabels = model.parseCSV(testingFile, classColName); 

    std::vector<std::string> predictions = {}; 
    std::vector<float> confidences = {};
    std::vector<bool> results = {}; 


    for (size_t i=0; i < testData.size(); ++i) { 
        std::pair<std::string, float> prediction = model.classify(testData[i]); 
        std::cout << "Prediction: " << prediction.first << " | Confidence: " << prediction.second << std::endl; 
        if (prediction.first == testLabels[i]){
            std::cout << "Correct" << std::endl;
            results.push_back(true);
        } else {
            std::cout << "Incorrect" << std::endl; 
            results.push_back(false); 
        }

        predictions.push_back(prediction.first); 
        confidences.push_back(prediction.second); 
    }

    //call stat saving function that takes (confidences, results) and saves %correct and mean(confidences) to a file, then returns those to print here to terminal.
    std::string timestamp = getTimestamp(); 
    std::string resultsFile = "data/cnn_results_" + timestamp + ".txt";
    saveStats(results, predictions, confidences, resultsFile);
    return 0; 
}




        




    
















