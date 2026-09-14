#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector> 
#include <fstream>
#include <cstdint>


class Matrix {
private: 
    size_t rows; 
    size_t cols; 
    std::vector<float> data; 

public: 

    float* dataPtr() { return data.data(); }
    const float* dataPtr() const { return data.data(); }
    
    size_t size() const;

    Matrix(); 
    Matrix(size_t r, size_t c, float initial=0.0); 
    Matrix& operator=(const Matrix& other) = delete; 
    
    void copyFrom(const Matrix& other); 

    
    inline float& operator()(size_t r, size_t c) { 
        if (r >= rows || c >= cols){
            throw std::out_of_range("Attempting to access uninitialized data, (" 
                    + std::to_string(r) + "," + std::to_string(c) 
                    + "). Actual dimensions: (" 
                    + std::to_string(rows) + "," + std::to_string(cols) + ")"
            );
        }

        return data[r * cols + c]; 
    }
    inline const float& operator()(size_t r, size_t c) const { 
        if (r >= rows || c >= cols ){
            throw std::out_of_range("Attempting to access uninitialized data, (" 
                    + std::to_string(r) + "," + std::to_string(c) 
                    + "). Actual dimensions: (" 
                    + std::to_string(rows) + "," + std::to_string(cols) + ")"
            );
        }

        return data[r * cols + c]; 
    }
    
    size_t dataSize() const { return data.size(); }
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; } 
    float getData(size_t index) const;
    void fillData(std::vector<float>& dataVec){ data = dataVec; }
    void setData(size_t index, float value);
    void setMask(std::vector<bool>& mask); 
    void applyMask(const std::vector<bool>& mask); 

    void setLeakyMask(std::vector<float>& mask, float alpha);
    void applyLeakyMask(const std::vector<float>& mask);
    
    Matrix transpose() const;
    void transposeTo(Matrix& result) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    void operator+=(const Matrix& other); 
    void operator-=(const Matrix& other);

    void matMul(const Matrix& other, Matrix& result) const;
    void matSub(const Matrix& other, Matrix& result) const;

    Matrix getRow(size_t r) const;
    void getRow(size_t r, Matrix& result) const;
    void setRow(size_t r, const Matrix& values);
    void addRow(size_t srcRow, size_t dstRow, const Matrix& other); 
    Matrix addBias(const Matrix& bias) const;
    static void concatCols(const Matrix& A, const Matrix& B, Matrix& C);
    static void splitCols(const Matrix& C, size_t ACols, Matrix& A, Matrix& B); 
    
    void setMatrix(size_t r, size_t c, float v){ 
        if (r * c > data.size()) data.resize(r * c); 
        rows = r; 
        cols = c; 
        fill(v); 
    }
    void setDimensions(size_t r, size_t c){
        if (r * c > data.size()) data.resize(r * c); 
        rows = r; 
        cols = c; 
    }

    void convert(std::vector<uint32_t>& data, size_t, size_t);

    void fill(float val);
    Matrix sumRows() const;
    void sumRows(Matrix& result) const; 
    void subtractScaled(const Matrix& other, float scalar);
    void subtractConstant(float scalar);
    static Matrix random(size_t r, size_t c);
    void scale(float c);
    void randomize(float low, float high); 
    
    void broadcastAdd(const Matrix& bias); 
    void broadcastSub(const Matrix& bias); 

    bool detectNaN() const;

    float mean(); 
    float var(); 
    float std(); 
    float L2();

    void save(std::ofstream& ofs) const;
    void load(std::ifstream& ifs); 

}; 

#endif
