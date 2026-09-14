#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"
#include "paramaterlayer.hpp"
#include "doubleConv1d.hpp"
#include <unordered_map>
#include <memory>
#include <cmath>

struct AdamState {
    Matrix m, v;
    Matrix mBias, vBias; 
    bool initialized = false;
    bool biasInitialized = false; 
}; 

class Adam: public Optimizer {
private:
    float b1 = 0.9f; 
    float b2 = 0.999f; 
    float eps = 1e-8f; 
    float learnRate; 
    size_t t = 0; 

    std::unordered_map<ParamaterLayer*, AdamState> states; 

public:
    Adam(float lr) : learnRate(lr) {}
    
    void updateLearningRate(float lr){
        learnRate = lr; 
    }
    void updateMoment(float* weightsData, const float* gradWeightsData, float* mData, float* vData, size_t size){

        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            float g = gradWeightsData[i]; 

            mData[i]= b1 * mData[i] + (1 - b1) * g; 
            vData[i] = b2 * vData[i] + (1 - b2) * (g * g); 

            float mHat = mData[i] / (1 - std::pow(b1, t)); 
            float vHat = vData[i] / (1 - std::pow(b2, t)); 

            weightsData[i] -= learnRate * mHat / (std::sqrt(vHat) + eps); 
        }
    }


    void step(ParamaterLayer& layer) override {

        AdamState& state = states[&layer]; 

        Matrix& weights = layer.getWeights(); 
        Matrix& gradWeights = layer.getGradient(); 

        float* weightData = weights.dataPtr();
        const float* gradWeightData = gradWeights.dataPtr(); 


        if (!state.initialized) {
            state.m.setMatrix(weights.getRows(), weights.getCols(), 0.0f); 
            state.v.setMatrix(weights.getRows(), weights.getCols(), 0.0f); 
            state.initialized = true; 
        }

        float* mData = state.m.dataPtr(); 
        float* vData = state.v.dataPtr(); 


        updateMoment(weightData, gradWeightData, mData, vData, weights.size());

        gradWeights.fill(0.0f);

        if (layer.hasBias()) {

            Matrix& bias = layer.getBias(); 
            Matrix& gradBias = layer.getBiasGradient(); 


            float* biasData = bias.dataPtr(); 
            const float* gradBiasData = gradBias.dataPtr();


            if (!state.biasInitialized) {
                
                state.mBias.setMatrix(bias.getRows(), bias.getCols(), 0.0f); 
                state.vBias.setMatrix(bias.getRows(), bias.getCols(), 0.0f); 

                state.biasInitialized = true; 
            }


            float* mBiasData = state.mBias.dataPtr(); 
            float* vBiasData = state.vBias.dataPtr(); 
 
            updateMoment(biasData, gradBiasData, mBiasData, vBiasData, bias.size()); 
            gradBias.fill(0.0f);
        }

    }

    void step(const std::vector<std::shared_ptr<Layer>>& layers, const std::shared_ptr<ParamaterLayer>& emb) {
        ++t;
        for (auto& layer : layers){ 
            auto dcLayer = dynamic_cast<DoubleConv1d*>(layer.get());
            if (dcLayer){
                step(*dcLayer->getSmallConv()); 
                step(*dcLayer->getLargeConv());
                continue;
            }
            auto pLayer = dynamic_cast<ParamaterLayer*>(layer.get()); 

            if (pLayer){
                step(*pLayer); 
            }
        }
        step(*emb);
    }




};

#endif
