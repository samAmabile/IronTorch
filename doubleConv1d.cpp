#include "doubleConv1d.hpp" 

DoubleConv1d::DoubleConv1d(size_t inChans, size_t outChans, size_t kSizeA, size_t kSizeB, size_t batchSize, size_t seqLen):
    smallConv(std::make_shared<Conv1d>(inChans, outChans, kSizeA, batchSize, seqLen)), 
    largeConv(std::make_shared<Conv1d>(inChans, outChans, kSizeB, batchSize, seqLen)), 
    smallWidth(outChans),
    A(batchSize*seqLen, outChans), B(batchSize*seqLen, outChans),
    gradA(batchSize*seqLen, outChans), gradB(batchSize*seqLen, outChans),
    gradInA(batchSize*seqLen, inChans), gradInB(batchSize*seqLen, inChans)
    {}

size_t DoubleConv1d::getOutDim() const {
    return smallConv->getOutDim() + largeConv->getOutDim(); 
}

void DoubleConv1d::forward(const Matrix& input, Matrix& output) {
    smallConv->forward(input, A); 
    largeConv->forward(input, B);

    Matrix::concatCols(A, B, output); 
}

void DoubleConv1d::backward(const Matrix& gradientOut, Matrix& gradIn, float lr) {
    Matrix::splitCols(gradientOut, smallWidth, gradA, gradB); 

    smallConv->backward(gradA, gradInA, lr); 
    largeConv->backward(gradB, gradInB, lr);

    gradIn.setDimensions(gradInA.getRows(), gradInA.getCols()); 
    gradIn.fill(0.0f); 

    gradIn += gradInA; 
    gradIn += gradInB; 
}

void DoubleConv1d::save(std::ofstream& ofs) const {
    smallConv->save(ofs);
    largeConv->save(ofs);
}

void DoubleConv1d::load(std::ifstream& ifs) {
    smallConv->load(ifs); 
    largeConv->load(ifs);
}




