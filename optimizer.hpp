#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "paramaterlayer.hpp"

class Optimizer {
public: 
    virtual ~Optimizer() = default; 
    virtual void step(ParamaterLayer& layer) = 0; 
};

#endif
