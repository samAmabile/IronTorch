#ifndef SGD_HPP
#define SGD_HPP

#include "optimizer.hpp"
#include <memory>

class SGD : public Optimizer {
private:
    float learningRate; 

public:
    SGD(float lr) : learningRate(lr) {}

    void step(ParamaterLayer& layer) override {
        layer.update(learningRate);
    }

    void step(const std::vector<std::shared_ptr<Layer>>& layers) {
        for (auto& layer : layers){
            auto pLayer = dynamic_cast<ParamaterLayer*>(layer.get()); 
            if (pLayer){
                step(*pLayer); 
            }
        }
    } 

};



#endif
