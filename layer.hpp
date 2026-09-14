#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp" 
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;
    
    virtual size_t getOutDim() const { return 0; }
    virtual void forward(const Matrix& input, Matrix& output) = 0; 
    virtual void backward(const Matrix& gradientOut, Matrix& gradIn, float learnRate) = 0; 

    virtual void save(std::ofstream& ofs) const = 0;
    virtual void load(std::ifstream& ifs) = 0; 
};

#endif



