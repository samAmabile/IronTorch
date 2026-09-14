#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp" 

class Loss { 
public: 
    static float computeMSE(const Matrix& output, const Matrix& target);
    static Matrix calcGrad(const Matrix& output, const Matrix& target); 

    static float crossEntropy(const Matrix& output, const Matrix& target); 
    static void calcCrossEntropyGrad(const Matrix& output, const Matrix& target, Matrix& grad);


    static float crossEntropySM(const Matrix& output, const Matrix& target); 
    static void calcCrossEntropyGradSM(const Matrix& output, const Matrix& target, Matrix& grad);

    static float categoricalCrossEntropy(const Matrix& output, const Matrix& target); 


    static void softmax(Matrix& output); 


}; 

#endif
