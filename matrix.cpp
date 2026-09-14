#include "matrix.hpp" 
#include <random> 
#include <algorithm> 
#include <fstream>
#include <cassert>
#include <iostream>
#include <cstring>
#include <cmath>

Matrix::Matrix(){
    rows = 1; 
    cols = 1; 
    data.assign(rows * cols, 0.0f);
}

Matrix::Matrix(size_t r, size_t c, float initial) : rows(r), cols(c), data(r * c, initial) {

    if (r == 0 || c == 0){
        throw std::invalid_argument("Trying to create matrix with dimensions: (" 
                + std::to_string(r) + "," 
                + std::to_string(c) + ")"
        );
    }

}

size_t Matrix::size() const{
    return rows * cols; 
}

void Matrix::copyFrom(const Matrix& other){
    if (this->data.size() < other.rows * other.cols){
        this->data.resize(other.rows * other.cols); 
    }
    this->rows = other.rows; 
    this->cols = other.cols; 
    std::copy(other.data.begin(), other.data.begin() + other.rows * other.cols, this->data.begin()); 
    
}

Matrix Matrix::operator+(const Matrix& other) const { 
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "Adding (" << rows << "," << cols << ") matrix to (" << other.rows << "," << other.cols << ") matrix" << std::endl;
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix result(rows, cols); 

    const float* a = this->data.data();
    const float* b = other.data.data();
    float* c = result.data.data();

    #pragma omp simd
    for (size_t i = 0; i < data.size(); i++) {
        c[i] = a[i] + b[i]; 
    }

    return result; 
}


Matrix Matrix::operator-(const Matrix& other) const { 
    if (rows != other.rows || cols != other.cols) { 
        std::cerr << "Subtracting (" << rows << "," << cols << ") matrix minus (" << other.rows << "," << other.cols << ") matrix" << std::endl;
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows, cols); 

    const float* a = this->data.data();
    const float* b = other.data.data();
    float* c = result.data.data();
    
    #pragma omp simd
    for (size_t i = 0; i < rows * cols; i++) {
        c[i] = a[i] - b[i]; 
    }

    return result; 
}

void Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Dimensions must match for += operation");
    }
    float* a = this->data.data(); 
    const float* b = other.data.data(); 
    size_t active = rows * cols; 

    #pragma omp simd
    for (size_t i = 0; i < active; ++i){
        a[i] += b[i]; 
    }
}


void Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Dimensions must match for += operation");
    }
    float* a = this->data.data(); 
    const float* b = other.data.data(); 
    size_t active = rows * cols; 

    #pragma omp simd
    for (size_t i = 0; i < active; ++i){
        a[i] -= b[i]; 
    }
}

Matrix Matrix::operator*(const Matrix& other) const { 
    if (cols != other.rows) {
        std::cerr << "ERROR: Attempting to multiply (" << rows << "," << cols << ") Matrix by a (" << other.rows << "," << other.cols << ") Matrix" << std::endl;
        throw std::invalid_argument("Rows of matrix A must match columns of matrix B for multiplication");
    }

    Matrix result(rows, other.cols, 0.0); 
    
    const float* a = this->data.data(); 
    const float* b = other.data.data(); 
    float* c = result.data.data(); 
   
    const int TILE = 32; 

    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < rows; ii += TILE){
        for (size_t jj = 0; jj < other.cols; jj += TILE){
            for (size_t kk = 0; kk < cols; kk += TILE){
                for (size_t i = ii; i < std::min(ii + TILE, rows); ++i){
                        for (size_t k = kk; k < std::min(kk + TILE, cols); ++k){
                            float cur = a[i * cols + k]; 
                            #pragma omp simd
                            for (size_t j = jj; j < std::min(jj + TILE, other.cols); ++j){
                                c[i * other.cols + j] += cur * b[k * other.cols + j]; 
                            }
                        }
                }
            }
        }
    }

    return result; 

}

//this matrix multiplication expects other matrix to already be transposed:
void Matrix::matMul(const Matrix& other, Matrix& result) const{
    assert(cols == other.cols && "Inner dimensions must match Rows for MatMul!!"); 
    
    result.setDimensions(rows, other.rows); 
    result.fill(0.0f); 

    const float* a = this->data.data(); 
    //use other here as it will already be transposed:
    const float* bT = other.data.data(); 
    float* c = result.data.data(); 
    
    const int TILE = 32; 
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < rows; ii += TILE){
        for (size_t jj = 0; jj < other.rows; jj += TILE){
            for (size_t i = ii; i < std::min(ii + TILE, rows); ++i){
                const float* rowA = a + (i * cols); 
                for (size_t j = jj; j < std::min(jj + TILE, other.rows); ++j){
                    const float* rowBT = bT + (j * cols);

                    float sum = 0.0f; 
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < cols; ++k){
                        sum += rowA[k] * rowBT[k];
                    }
                    c[i * other.rows +j] = sum;
                }
            }
        }
    }
}


void Matrix::matSub(const Matrix& other, Matrix& result) const{
    assert(rows == other.rows && "rows mismatched for subtraction");
    assert(cols == other.cols && "cols mismatched for subtraction");

    result.setDimensions(rows, cols); 
    result.fill(0.0f); 

    const float* a = this->data.data(); 
    const float* b = other.data.data(); 
    float* c = result.data.data(); 
    size_t active = rows * cols; 

    #pragma omp simd
    for (size_t i = 0; i < active; ++i){
        c[i] = a[i] - b[i]; 
    }
}

float Matrix::getData(size_t index) const { 
    if (index > data.size()-1) { 
        throw std::invalid_argument("Index out of range"); 
    }
    return data[index]; 
}

void Matrix::setData(size_t index, float value) { 
    if (index >= data.size()) {
        data.push_back(value);
        return;
    }
    data[index] = value; 
}

//this has been replaced by the void version below:
Matrix Matrix::getRow(size_t r) const {
    assert(r < rows && "Row out of bounds");
    Matrix row(1, cols); 
    
    #pragma omp simd
    for (size_t i = 0; i < cols; ++i){
        row(0, i) = (*this)(r, i);
    }

    return row;
}

void Matrix::getRow(size_t r, Matrix& result) const{
    assert(r < rows && "Row out of bounds!"); 

    result.setDimensions(1, cols); 

    std::memcpy(result.data.data(), &this->data[r * cols], cols * sizeof(float));

}

void Matrix::setRow(size_t r, const Matrix& values){

    assert(r < rows && "row out of bounds");
    assert(values.rows == 1 && "row must be 1xn");
    assert(values.cols == this->cols && "columns must match");

    std::memcpy(&this->data[r * cols], values.data.data(), cols * sizeof(float)); 

}

void Matrix::addRow(size_t srcRow, size_t dstRow, const Matrix& other){

    assert(this->cols == other.cols && "Column mismatch");
    assert(srcRow < other.rows && "in Row Out of Bounds");
    assert(dstRow < this->rows && "out Row out of bounds");

    float* dstPtr = this->data.data() + (dstRow * this->cols);
    const float* srcPtr = other.data.data() + (srcRow * other.cols);

    
    #pragma omp simd
    for (size_t i = 0; i < cols; ++i){
        dstPtr[i] += srcPtr[i]; 
    }
}

//this has been replaced by broadcastAdd:
Matrix Matrix::addBias(const Matrix& bias) const {
    if (this->cols != bias.cols || bias.rows != 1) {
        throw std::invalid_argument("Bias dimensions must be (1, cols)");
    }

    Matrix result(*this); 
    #pragma omp parallel for
    for (size_t i = 0; i < result.rows; ++i){
        #pragma omp simd
        for (size_t j = 0; j < result.cols; ++j){
            result(i, j) += bias(0, j);
        }
    }

    return result;
}

void Matrix::setMask(std::vector<bool>& mask) { 

    #pragma omp simd
    for (size_t i = 0; i < rows * cols; ++i) {
        if (data[i] > 0) {
            mask[i] = true; 
        } else {
            mask[i] = false; 
            data[i] = 0.0; 
        }
    }
}

void Matrix::applyMask(const std::vector<bool>& mask) { 
    
    #pragma omp simd
    for (size_t i = 0; i < mask.size(); ++i) { 
        if (!mask[i]){ 
            data[i] = 0.0; 
        }
    }
}

void Matrix::setLeakyMask(std::vector<float>& mask, float alpha) {
    #pragma omp simd
    for (size_t i = 0; i < rows * cols; ++i){
        if (data[i] > 0.0f){
            mask[i] = 1.0f;
        } else {
            mask[i] = alpha; 
            data[i] = data[i] * alpha; 
        }
    }
}
void Matrix::applyLeakyMask(const std::vector<float>& mask){
    #pragma omp simd
    for (size_t i = 0; i < mask.size(); ++i){
        data[i] *= mask[i]; 
    }
}

//this is inefficient and we should always use randomize instead
Matrix Matrix::random(size_t r, size_t c){

    Matrix m(r, c); 
    
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-0.1, 0.1); 
    
    for (size_t i = 0; i < r*c; ++i){
        m.data[i] = distribution(generator); 
    }
    
    return m; 
}

void Matrix::randomize(float low, float high){
    std::random_device rd; 
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(low, high); 

    for (size_t i = 0; i < rows*cols; ++i){
        data[i] = dist(generator);
    }
}


Matrix Matrix::transpose() const { 

    Matrix result(cols, rows);

    const float* src = data.data(); 
    float* dst = result.data.data(); 
    
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j]; 
        }
    }

    return result; 
}

void Matrix::transposeTo(Matrix& result) const {
    result.setDimensions(cols, rows); 
    const float* srcPtr = data.data(); 
    float* dstPtr = result.data.data(); 

    const int TILE = 32; 
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < rows; ii += TILE){
        for (size_t jj = 0; jj < cols; jj += TILE){
            for (size_t i = ii; i < std::min(ii + TILE, rows); ++i){
                for (size_t j = jj; j < std::min(jj + TILE, cols); ++j){
                    dstPtr[j * rows + i] = srcPtr[i * cols + j];
                }
            }
        }
    }
}


void Matrix::subtractScaled(const Matrix& other, float scalar) { 
    if (data.size() != other.data.size()) { 
        throw std::invalid_argument("Matrix sizes must match for subtraction"); 
    }
    
    #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= scalar * other.data[i]; 
    }
}

void Matrix::subtractConstant(float scalar) {
    float* d = data.data(); 
    size_t n = rows * cols;
    
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        d[i] -= scalar; 
    }
}


void Matrix::fill(float val){

    if (val == 0.0f) {
        std::memset(data.data(), 0, rows * cols * sizeof(float)); 
    } else{
        std::fill(data.begin(), data.begin() + (rows*cols), val);
    }
}


//this one is inefficient, can probably remove:
Matrix Matrix::sumRows() const {
    Matrix ret(1, cols, 0.0); 
    
    for (size_t i = 0; i < rows; i++){
        #pragma omp simd
        for (size_t j = 0; j < cols; j++){
            ret(0, j) += (*this)(i, j); 
        }
    }
    return ret; 
}

//keep:
void Matrix::sumRows(Matrix& result) const{
    result.setDimensions(1, cols); 
    result.fill(0.0f); 
    
    float* resData = result.data.data();
    const float* myData  =this->data.data(); 

    for (size_t i = 0; i < rows; ++i){
        const float* rowPtr = myData + (i * cols);
        #pragma omp simd
        for (size_t j = 0; j < cols; ++j){
            resData[j] += rowPtr[j];
        }
    }
}

void Matrix::scale(float c){
    #pragma omp simd
    for (size_t i = 0; i < rows * cols; ++i){
        data[i] *= c; 
    }
}

//TODO: replace all overloaded matrix(r, c) calls with float* direct vector access 
//
void Matrix::broadcastAdd(const Matrix& bias){ 
    float* curPtr = this->data.data(); 
    const float* biasPtr = bias.data.data(); 

    #pragma omp parallel for
    for (size_t r = 0; r < rows; ++r){
        float* rowPtr = curPtr + (r * cols);
        #pragma omp simd
        for (size_t c = 0; c < cols; ++c){
            rowPtr[c] += biasPtr[c];
        }
    }
}

void Matrix::broadcastSub(const Matrix& bias) {

    float* curPtr = this->data.data(); 
    const float* biasPtr = bias.data.data(); 

    #pragma omp parallel for
    for (size_t r = 0; r < rows; ++r){
        float* rowPtr = curPtr + (r * cols);
        #pragma omp simd
        for (size_t c = 0; c < cols; ++c){
            rowPtr[c] -= biasPtr[c];
        }
    }
}

void Matrix::convert(std::vector<uint32_t>& data, size_t batchSize, size_t seqLen){

    setDimensions(batchSize, seqLen); 
    fill(0.0f); 
    for (size_t i = 0; i < batchSize; ++i){
        for (size_t j = 0; j < seqLen; ++j){
            (*this).data[i * seqLen + j] = static_cast<float>(data[i * seqLen + j]); 
        }
    }
}

bool Matrix::detectNaN() const {

    int size = rows * cols; 
    for (int i{0}; i < size; i++){

        if (std::isnan(data[i]) || std::isinf(data[i])) return true;

    }

    return false;
}

void Matrix::concatCols(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.getRows() == B.getRows() && "matrix rows must match");

    size_t ARows = A.getRows();
    size_t ACols = A.getCols();
    size_t BCols = B.getCols();
    size_t newRows = A.getRows(); 
    size_t newCols = ACols + BCols;

    C.setDimensions(newRows, newCols); 

    const float* Adata = A.dataPtr(); 
    const float* Bdata = B.dataPtr(); 
    float* Cdata = C.dataPtr(); 

    size_t index = 0; 
    
    for (size_t r{0}; r < ARows; ++r){
        std::memcpy(Cdata + (r * newCols), Adata + (r * ACols), ACols * sizeof(float)); 
        std::memcpy(Cdata + (r * newCols) + ACols, Bdata + (r * BCols), BCols * sizeof(float));
    }
}

void Matrix::splitCols(const Matrix& C, size_t ACols, Matrix& A, Matrix& B) {
    
    assert(A.getRows() == B.getRows() && "matrix rows must match");

    size_t totalCols = C.getCols(); 
    size_t totalRows = C.getRows();
    size_t BCols = totalCols - ACols; 

    A.setDimensions(totalRows, ACols); 
    B.setDimensions(totalRows, BCols); 

    const float* Cdata = C.dataPtr(); 
    float* Adata = A.dataPtr(); 
    float* Bdata = B.dataPtr(); 


    for (size_t r{0}; r < totalRows; ++r){
        std::memcpy(Adata + (r * ACols), Cdata + (r * totalCols), ACols * sizeof(float)); 
        std::memcpy(Bdata + (r * BCols), Cdata + (r * totalCols) + ACols, BCols * sizeof(float)); 
    }
}

float Matrix::mean() {
    float mean = 0.0f; 
    float sum = 0.0f; 
    size_t n = rows * cols;
    for (size_t i{0}; i < n; i++){
        sum += data[i]; 
    }
    mean = sum / n; 
    
    return mean; 
}

float Matrix::var() {
    float mean = (*this).mean(); 
    float sum = 0.0f; 
    size_t n = rows * cols; 
    for (size_t i{0}; i < n; ++i){
        float cur = data[i] - mean; 
        sum += cur * cur; 
    }

    float var = sum / n; 
    return var; 
}

float Matrix::std() {
    float var = (*this).var(); 
    float std = std::sqrt(var); 
    
    return std;
}

float Matrix::L2() {
    float sum = 0.0f; 
    size_t n = rows * cols; 

    for (size_t i{0}; i < n; ++i){
        sum += data[i] * data[i]; 
    }

    float L2 = std::sqrt(sum); 

    return L2; 
}



void Matrix::save(std::ofstream& ofs) const {
    size_t r = getRows(); 
    size_t c = getCols(); 
    ofs.write(reinterpret_cast<const char*>(&r), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&c), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

}

void Matrix::load(std::ifstream& ifs) {
    size_t r, c; 
    ifs.read(reinterpret_cast<char*>(&r), sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(&c), sizeof(size_t)); 
    setDimensions(r, c);

    ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
}
