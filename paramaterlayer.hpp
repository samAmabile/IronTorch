#ifndef PARAMATERLAYER_HPP
#define PARAMATERLAYER_HPP
#include "layer.hpp"

class ParamaterLayer : public Layer { 
public: 
    virtual Matrix& getWeights() = 0; 
    virtual Matrix& getGradient() = 0;
    virtual Matrix& getBias() = 0; 
    virtual Matrix& getBiasGradient() = 0; 

    virtual void update(float lr) = 0;

    virtual bool hasBias() const = 0; 

};

#endif

