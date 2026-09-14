#include "dense.hpp" 
#include <iostream>
#include <cassert>
#include <cmath>

Dense::Dense(size_t in, size_t out) : 
    weights(in, out),
    weightsT(out, in),
    bias(1, out, 0.0), 
    gradWeights(in, out, 0.0), 
    gradWeightsT(out, in, 0.0),
    gradBias(1, out, 0.0),
    gradBiasT(out, 1, 0.0),
    prevIn(1, 1), 
    prevInT(1, 1)
{
    
    float init = std::sqrt(2.0f/static_cast<float>(in));
    weights.randomize(-init, init);
    weights.transposeTo(weightsT);
}

void Dense::forward(const Matrix& input, Matrix& output) {

    prevIn.copyFrom(input); 
    

    output.setDimensions(input.getRows(), weights.getCols()); 
    output.fill(0.0f);

    assert(input.getCols() == weightsT.getCols() && "mismatch in Dense::forward, input * weightsT -> output");   
    input.matMul(weightsT, output);
    output.broadcastAdd(bias); 
}

void Dense::backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/) {
    
    prevIn.transposeTo(prevInT); //transpose so math works, NOT to accomodate special matMul 
    gradientOut.transposeTo(gradWeightsT); // transpose to accomodate special matMul

    assert(prevInT.getCols() == gradWeightsT.getCols() && "mismatch in Dense::backward (first assert)");

    scratch.setDimensions(prevInT.getRows(), gradWeightsT.getRows());//prep scratchpad for matMul 
    scratch.fill(0.0f);

    prevInT.matMul(gradWeightsT, scratch); //prevInT * gradientOutT -> scratch

    gradWeights += scratch; //add scratch to gradWeights
    
    
    gradientOut.sumRows(scratch); //here scratch gets re-purposed to hold the summed rows of gradientOut

    assert((gradBias.getRows() == scratch.getRows() && gradBias.getCols() == scratch.getCols()) && "Dense Backward Scratch addition fail SECOND");
    gradBias += scratch; //add summed rows of gradientOut to gradBias
    
    //THIS WAS THE BUG:
    weights.transposeTo(weightsT); // since gradIn expects gradientOut * weightsT, but matMul will expect gradientOut * (weightsT)T to do the optimized operation, we leave weights alone for the multiplication 
    gradIn.setDimensions(gradientOut.getRows(), weights.getCols()); //IDK why not rows x rows, but it works ??? 
    gradIn.fill(0.0f);

    gradientOut.matMul(weights, gradIn);

}


Matrix& Dense::getWeights() { return weights; }
Matrix& Dense::getBias() { return bias; }
Matrix& Dense::getGradient() { return gradWeights; }
Matrix& Dense::getBiasGradient() { return gradBias; }

void Dense::update(float lr){
    weights.subtractScaled(gradWeights, lr); 
    bias.subtractScaled(gradBias, lr); 
    gradWeights.fill(0.0);
    gradBias.fill(0.0); 

    weights.transposeTo(weightsT);
}

void Dense::save(std::ofstream& ofs) const {
    weights.save(ofs);
    bias.save(ofs);
}

void Dense::load(std::ifstream& ifs) {
    weights.load(ifs); 
    bias.load(ifs); 
    weights.transposeTo(weightsT); 

}
