//
//  MnistNet.cpp
//  MyCaffe
//
//  Created by quanhai on 2017/5/12.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#include "MnistNet.hpp"



#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream>
#include <vector>
using namespace std;

// http://www.cnblogs.com/yeahgis/archive/2012/07/13/2590485.html
// 高斯分布的随机数，均值为0，方差为1
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    
    return X;
}

// 不做内存释放，搞简单一点
typedef shared_ptr<double> DoublePtr;
inline DoublePtr newDoubleArray(int size)
{
    double *p = new double[size];
    return DoublePtr(p, default_delete<double[]>());
}

// 简单的矩阵(复制的时候只复制meta，不复制实际数据)
struct Matrix
{
    int row, col, size;
    DoublePtr data;
    
    Matrix(int _row=1, int _col=1) : row(_row), col(_col)
    {
        size = row * col;
        data = newDoubleArray(size);
        memset(data.get(), 0, sizeof(double) * size);
    }
    
    inline double* operator[](int i) {
        assert(i < row);
        return data.get() + i * col;
    }
};

// 打印矩阵内容
ostream& operator<<(ostream& out, Matrix w)
{
    out << "[ (" << w.row << " x " << w.col << ")" << endl;
    for(int i = 0;i < w.row;i++) {
        out << "\t[";
        for(int j = 0;j < w.col;j++) {
            if(j > 0) out << ",";
            out << w[i][j];
        }
        out << "]" << endl;
    }
    out << "]";
    return out;
}

// 简单的向量(复制的时候只复制meta，不复制实际数据)
struct Vector
{
    int size;
    DoublePtr data;
    
    Vector(int _size=1) : size(_size)
    {
        data = newDoubleArray(size);
        memset(data.get(), 0, sizeof(double) * size);
    }
    
    inline double &operator[](int x)
    {
        assert(x < size);
        return data.get()[x];
    }
};

// 打印向量内容
ostream& operator<<(ostream& out, Vector v)
{
    out << "[ (" << v.size << ") ";
    for(int i = 0;i < v.size;i++) {
        if(i > 0) out << ",";
        out << v[i];
    }
    out << "]";
    return out;
}

Vector operator*(Matrix w, Vector v)
{
    Vector ret(w.row);
    for(int i = 0;i < w.row;i++) {
        for(int j = 0;j < w.col;j++) {
            ret[i] += w[i][j] * v[j];
        }
    }
    return ret;
}

// 点乘
Vector operator*(Vector x, Vector y)
{
    Vector ret(x.size);
    for(int i = 0;i < x.size;i++) {
        ret[i] = x[i] * y[i];
    }
    return ret;
}

// w转置，然后和v相乘
Vector TandMul(Matrix w, Vector v)
{
    Vector ret(w.col);
    for(int i = 0;i < w.col;i++) {
        for(int j = 0;j < w.row;j++) {
            ret[i] += w[j][i] * v[j];
        }
    }
    return ret;
}

Vector operator+(Vector x, Vector y)
{
    Vector ret(x.size);
    for(int i = 0;i < x.size;i++) {
        ret[i] = x[i] + y[i];
    }
    return ret;
}

Vector operator-(Vector x, Vector y)
{
    Vector ret(x.size);
    for(int i = 0;i < x.size;i++) {
        ret[i] = x[i] - y[i];
    }
    return ret;
}

Vector operator*(double x, Vector y)
{
    Vector ret(y.size);
    for(int i = 0;i < y.size;i++) {
        ret[i] = x * y[i];
    }
    return ret;
}

Vector operator*(Vector x, double y)
{
    return y * x;
}

// Cost函数
struct CostFun
{
    virtual double calc(Vector x, Vector y)
    {
        return 0;
    }
    
    virtual double operator()(Vector x, Vector y)
    {
        return calc(x,y);
    }
    
    virtual Vector propagateDelta(Vector output, Vector y)
    {
        return Vector(output.size);
    }
};

// 方差Cost函数
struct SqrCostFun: CostFun
{
    // \sum (x_i-y_i)^2/2
    virtual double calc(Vector x, Vector y)
    {
        double ret = 0;
        for(int i = 0;i < x.size;i++) {
            double t = x[i] - y[i];
            ret += t * t;
        }
        return ret / 2;
    }
    
    virtual Vector propagateDelta(Vector output, Vector y)
    {
        /*
         d \sum (x_i-y_i)^2/2
         = \sum x_i-y_i
         // x - y
         */
        return output - y;
    }
};

struct SoftmaxCostFun: CostFun
{
    // - \sum y_j * log( exp(x_j) / \sum exp(x_k) )
    virtual double calc(Vector x, Vector y)
    {
        /*
         log( exp(x_j) / \sum exp(x_k) )
         = x_j - log \sum exp(x_k)
         = x_j - (max + log \sum exp(x_k - max))
         */
        double maxX = x[0];
        for(int i = 0;i < x.size;i++) {
            if(x[i] > maxX) {
                maxX = x[i];
            }
        }
        
        double xSum = 0;
        for(int i = 0;i < x.size;i++) {
            xSum += exp(x[i] - maxX);
        }
        
        double ret = 0;
        for(int i = 0;i < x.size;i++) {
            ret += y[i] * (x[i] - (maxX + log(xSum)));
        }
        
        return -ret;
    }
    
    virtual Vector propagateDelta(Vector output, Vector y)
    {
        Vector x = output;
        /*
         - d \sum y_j * log( exp(x_j) / \sum exp(x_k) )
         = - d \sum y_j * x_j - d \sum y_j log (\sum exp(x_k) )
         = - y_i + \sum (y_j * exp(x_i) / \sum exp(x_k))
         = - y_i + exp(x_i) (\sum y_j) / (\sum exp(x_k))
         */
        
        double maxX = x[0];
        for(int i = 0;i < x.size;i++) {
            if(x[i] > maxX) {
                maxX = x[i];
            }
        }
        
        // y - exp(x) sum_of_y / sum_of_exp(x)
        double sumOfY = 0;
        double sumOfX = 0;
        Vector tmp(x.size);
        for(int i = 0;i < x.size;i++) {
            tmp[i] = exp(x[i] - maxX);
            sumOfY += y[i];
            sumOfX += tmp[i];
        }
        
        return sumOfY/sumOfX * tmp - y;
    }
};

// 单例
SqrCostFun SqrCostFunSingleton;
SoftmaxCostFun SoftmaxCostFunSingleton;

// 激活函数
struct Activator
{
    // forward
    virtual double forward(double v)
    {
        return v;
    }
    
    virtual double operator()(double v)
    {
        return forward(v);
    }
    
    virtual Vector operator()(Vector v)
    {
        Vector ret(v.size);
        for(int i = 0;i < v.size;i++) {
            ret[i] = forward(v[i]);
        }
        return ret;
    }
    
    // 求导数
    virtual double derive(double v)
    {
        return 1;
    }
    
    virtual Vector derive(Vector v)
    {
        Vector ret(v.size);
        for(int i = 0;i < ret.size;i++) {
            ret[i] = derive(v[i]);
        }
        return ret;
    }
};

// Sigmoid激活函数
struct SigmoidActivator : Activator
{
    virtual double forward(double v)
    {
        return 1 / (1 + exp(-v));
    }
    
    virtual double derive(double v)
    {
        double t = exp(-v);
        return t / ( (1 + t) * (1 + t) );
    }
};

struct ReluActivator: Activator
{
    virtual double forward(double v)
    {
        return v >= 0 ? v : 0;
    }
    
    virtual double derive(double v)
    {
        return v >= 0 ? 1 : 0;
    }
};

// 单例
SigmoidActivator SigmoidActivatorSingleton;
ReluActivator ReluActivatorSingleton;

// NN的一层
// 1. 输入不算一层
// 2. 层的w矩阵是从前面一层到当前层的w，和NG的定义有些出入
// 3. 层的b是前面一层到当前层的b，和NG的定义有些出入
struct Layer
{
    // 上一层的输出的个数，不包括bias
    int inSize;
    // 当前层的输出
    int outSize;
    
    Activator &activator;
    Matrix w;
    Vector b;
    
    void initWeights(double *p, int size)
    {
        // 采用 (0, 0.01)的正太分布初始化
        for(int i = 0;i < size;i++) {
            p[i] = gaussrand() * 0.01;
        }
    }
    
    Layer(int _inSize=1, int _outSize=1, Activator &_activator= SigmoidActivatorSingleton):
    inSize(_inSize),
    outSize(_outSize),
    w(_outSize, _inSize),
    b(_outSize),
    activator(_activator)
    {
        initWeights(w.data.get(), w.size);
        initWeights(b.data.get(), b.size);
    }
    
    // 最后一次forward计算之后保存的激活值
    Vector a;
    Vector z;
    // in是上一层的输出
    Vector operator()(Vector in)
    {
        z = w * in + b;
        return a = activator(z);
    }
    
    // 最后一次反向传播计算之后保存的delta值
    Vector delta;
    Vector propagateDelta()
    {
        return TandMul(w, delta);
    }
    
    // alpha是学习率
    // prevA是上一层的输出
    void updateParameters(double alpha, Vector prevA)
    {
        b = b + (-alpha) * delta;
        Matrix nw(w.row, w.col);
        for(int i = 0;i < w.row;i++) {
            for(int j = 0;j < w.col;j++) {
                nw[i][j] = w[i][j] - alpha * prevA[j] * delta[i];
            }
        }
        w = nw;
    }
    
    // 利用NG的办法进行BP算法正确性的检查
    void checkBp(Layer layerList[], int nLayer, Vector input, Vector y, CostFun& costFun, double alpha, Vector prevA)
    {
        Vector forward(Layer layerList[], int nLayer, Vector input);
        
        // BP算法计算的结果
        Vector db = delta;
        Matrix dw(w.row, w.col);
        for(int i = 0;i < w.row;i++) {
            for(int j = 0;j < w.col;j++) {
                dw[i][j] = prevA[j] * delta[i];
            }
        }
        
        // NG办法计算的结果
        // 1. 考虑对b增加一个e
        double EPS = 0.0001;
        for(int i = 0;i < b.size;i++) {
            double tmp = b[i];
            // +eps
            b[i] = tmp + EPS;
            Vector output1 = forward(layerList, nLayer, input);
            double err1 = costFun(output1, y);
            // -eps
            b[i] = tmp - EPS;
            Vector output2 = forward(layerList, nLayer, input);
            double err2 = costFun(output2, y);
            
            // 恢复原值
            b[i] = tmp;
            
            // NG方法的估算值
            double v = (err1 - err2) / (2 * EPS);
            if(v > 0) {
                double x = fabs( (v-db[i]) / v);
                if(x > 0.0001) {
                    cerr << "BP算法结果误差太大了" << endl;
                }
            }
        }
        
        // 2. 考虑对w增加一个e
        for(int i = 0;i < w.row;i++) {
            for(int j = 0;j < w.col;j++) {
                double tmp = w[i][j];
                // +eps
                w[i][j] = tmp + EPS;
                Vector output1 = forward(layerList, nLayer, input);
                double err1 = costFun(output1, y);
                // -eps
                w[i][j] = tmp - EPS;
                Vector output2 = forward(layerList, nLayer, input);
                double err2 = costFun(output2, y);
                
                // 恢复原值
                w[i][j] = tmp;
                
                // NG方法的估算值
                double v = (err1 - err2) / (2 * EPS);
                if(v > 0) {
                    double x = fabs( (v-dw[i][j]) / v);
                    if(x > 0.0001) {
                        cerr << "BP算法结果误差太大了" << endl;
                    }
                }
            }
        }
    }
};

ostream& operator<<(ostream& out, Layer& layer)
{
    out << "Layer {" << endl;
    out << "w = " << layer.w << endl;
    out << "b = " << layer.b << endl;
    out << "z = " << layer.z << endl;
    out << "a = " << layer.a << endl;
    out << "delta = " << layer.delta << endl;
    out << "}" << endl;
    return out;
}

Vector forward(Layer layerList[], int nLayer, Vector input)
{
    Vector tmp = input;
    for(int i = 0;i < nLayer;i++) {
        tmp = layerList[i](tmp);
    }
    return tmp;
}

void backward(Layer layerList[], int nLayer, Vector input, Vector y, CostFun& costFun, double alpha)
{
    // 反向传播delta
    Layer &lastLayer = layerList[nLayer - 1];
    // Sqr cost function为例是: -(y - a) f'(z)
    lastLayer.delta = costFun.propagateDelta(lastLayer.a, y) * lastLayer.activator.derive(lastLayer.z);
    
    for(int i = nLayer - 2;i >= 0;i--) {
        Layer &layer = layerList[i];
        Layer &nextLayer = layerList[i + 1];
        layer.delta = nextLayer.propagateDelta() * layer.activator.derive(layer.z);
    }
    
    /*
     // 检查BP算法正确性（只做一次）
     static bool hasDoneBpChecking = false;
     if(!hasDoneBpChecking) {
     hasDoneBpChecking = true;
     for(int i = 0;i < nLayer;i++) {
     layerList[i].checkBp(layerList, nLayer, input, y, costFun, alpha, i == 0 ? input : layerList[i - 1].a);
     }
     }
     //*/
    
    // 更新所有的w和b
    for(int i = 0;i < nLayer;i++) {
        layerList[i].updateParameters(alpha, i == 0 ? input : layerList[i - 1].a);
    }
}

int MsbInt(char buf[], int len=4)
{
    int base = 1;
    int ret = 0;
    for(int i = len - 1;i >= 0;i--) {
        ret += (unsigned char)buf[i] * base;
        base *= 256;
    }
    return ret;
}
vector<int> ReadMnistLabels(string fileName)
{
    vector<int> ret;
    ifstream ifs(fileName.c_str(), ios::binary);
    char buf[1000];
    
    // MAGIC
    ifs.read(buf, 4);
    int magic = MsbInt(buf);
    if(magic != 0x00000801) {
        cerr << "incorrect label file magic number" << endl;
    }
    
    // num of images
    ifs.read(buf, 4);
    int nImages = MsbInt(buf);
    
    while(nImages--) {
        ret.push_back(ifs.get());
    }
    
    return ret;
}
vector<Vector> ReadMnistData(string fileName)
{
    vector<Vector> ret;
    ifstream ifs(fileName.c_str(), ios::binary);
    char buf[1000];
    
    // MAGIC
    ifs.read(buf, 4);
    int magic = MsbInt(buf);
    if(magic != 0x00000803) {
        cerr << "incorrect data file magic number" << endl;
    }
    
    // num of images
    ifs.read(buf, 4);
    int nImages = MsbInt(buf);
    
    int row, col;
    ifs.read(buf, 4);
    row = MsbInt(buf);
    ifs.read(buf, 4);
    col = MsbInt(buf);
    if(row != 28 || col != 28) {
        cerr << "incorrect image size" << endl;
    }
    
    while(nImages--) {
        Vector image(row * col);
        for(int i = 0;i < row * col;i++) {
            image[i] = ifs.get() / 256.0; // 归一化
        }
        ret.push_back(image);
    }
    
    return ret;
}

vector<Vector> translateLabels(vector<int> labels, int k=10)
{
    vector<Vector> ret;
    for(int i = 0;i < labels.size();i++) {
        Vector tmp(k);
        assert(labels[i] >= 0 && labels[i] < k);
        tmp[labels[i]] = 1;
        ret.push_back(tmp);
    }
    return ret;
}

int getMaxIdx(Vector x)
{
    int maxIdx = 0;
    double maxV = x[0];
    for(int i = 0;i < x.size;i++) {
        if(x[i] > maxV) {
            maxV = x[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

int main()
{
    srand(1000);
    
    cout << "Loading data" << endl;
    // 读取数据
    vector<int> trainLabels = ReadMnistLabels("mnist/train-labels-idx1-ubyte");
    vector<Vector> trainData = ReadMnistData("mnist/train-images-idx3-ubyte");
    vector<Vector> trainLabels2 = translateLabels(trainLabels);
    
    vector<int> testLabels = ReadMnistLabels("mnist/t10k-labels-idx1-ubyte");
    vector<Vector> testData = ReadMnistData("mnist/t10k-images-idx3-ubyte");
    
    // 输入是28 x 28
    // 输出是10
    int nInput = 28 * 28;
    int nOutput = 10;
    
    // NN网络结构
    Layer layerList[] = {
        Layer(nInput, nOutput, ReluActivatorSingleton),
        //Layer(100, nOutput, ReluActivatorSingleton)
    };
    
    // Cost fun
    CostFun &costFun = SoftmaxCostFunSingleton;
    
    // 不包括输入层在内的层的个数
    int nLayer = sizeof(layerList) / sizeof(layerList[0]);
    
    int M = trainData.size();
    int T = testData.size();
    
    // 开始训练
    cout << "Start training" << endl;
    time_t fullStartedAt = time(NULL);
    for(int step = 0;step < 100000;step++) {
        time_t startedAt = time(NULL);
        
        double avgError = 0;
        
        for(int i = 0;i < M;i++) {
            Vector x = trainData[i];
            Vector y = trainLabels2[i];
            
            Vector output = forward(layerList, nLayer, x);
            double error = costFun(output, y);
            //cout << output << " " << y << " " << error << endl;
            avgError += error;
            
            backward(layerList, nLayer, x, y, costFun, 0.001);
        }
        avgError /= M;
        
        time_t endedAt = time(NULL);
        
        cout << "step=" << step << " time_cost=" << (endedAt - startedAt) << " avgErr=" << avgError << " ";
        
        // validate
        int nTotal = 0;
        int nGood = 0;
        for(int i = 0;i < M;i++) {
            Vector x = trainData[i];
            Vector output = forward(layerList, nLayer, x);
            int maxIdx = getMaxIdx(output);
            if(maxIdx == trainLabels[i]) {
                nGood++;
            }
            nTotal++;
        }
        cout << "train_accuracy " << nGood << "/" << nTotal << "=" << nGood*1.0/nTotal << " ";
        bool doBreak = false;
        if(nGood * 1.0 / nTotal > 0.95) {
            doBreak = true;
        }
        
        // check
        nTotal = 0;
        nGood = 0;
        for(int i = 0;i < T;i++) {
            Vector x = testData[i];
            Vector output = forward(layerList, nLayer, x);
            int maxIdx = getMaxIdx(output);
            if(maxIdx == testLabels[i]) {
                nGood++;
            }
            nTotal++;
        }
        cout << "test_accuracy " << nGood << "/" << nTotal << "=" << nGood*1.0/nTotal;
        cout << "\n";
        
        if(doBreak) {
            break;
        }
    }
    
    time_t fullEndedAt = time(NULL);
    cout << "Total cost " << (fullEndedAt - fullStartedAt) << " seconds" << endl;
    
    return 0;
}
