#pragma once

#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include "Eigen/Dense"

typedef Eigen::Matrix<double, -1, -1, Eigen::ColMajor> ColMatrixXd;

class Model;

struct Stack
{
	Stack(int32_t v, int32_t d, int32_t c);
	Stack operator+(const Stack& other) const;
	Stack operator+=(const Stack& other);
	static Stack Zeros(const Stack& other); // Creates stack of same size but set to zero

	void setConstant(double c);
	void setRandom(double r);
	void setZero();
	void print() const;

	Eigen::MatrixXd L; // Embeddings (d x v)
	Eigen::MatrixXd Ws; // Sentiment classification matrix (c x d)
	Eigen::MatrixXd bs; // Sentiment classification bias (c x 1)
	Eigen::MatrixXd W; // Parameters (d x 2d)
	Eigen::MatrixXd b; // Parameter bias (d x 1)
	std::vector<Eigen::MatrixXd> V; // Tensor (2d x 2d x d)
};

class Data
{
public:
	virtual double build(Model* model) = 0;
	virtual void derivative(Stack& stack, Model* model) = 0;
	virtual void print() = 0;
};


class Model
{
public:

	Model(int32_t v, int32_t d, int32_t c, double r, double lambda);
	virtual ~Model();

	int32_t getWordDim() const;
	int32_t getClassDim() const;

	double f(double x) const; // Nonlinearity function
	double dfdx(double fx) const; // Nonlinearity derivative (taking f(x) as input)
	Eigen::VectorXd f(const Eigen::VectorXd& x) const; // cwise nonlinearity function
	Eigen::VectorXd dfdx(const Eigen::VectorXd& fx) const; // cwise nonlinearity derivative

	double regulariser(); // Regulariser energy
	void addRegulariserDerivative(Stack& stack); // Add regulariser energy derivatives to given stack

	void resetAda();
	
	const Eigen::VectorXd getL(int32_t i) const;
	const Eigen::MatrixXd& getWs() const;
	const Eigen::MatrixXd& getbs() const;
	const Eigen::MatrixXd& getW() const;
	const Eigen::MatrixXd& getb() const;
	const Eigen::MatrixXd& getV(int32_t i) const;

	// Tests derivatives using finite difference
	void diffTest(Data* data, double h);
	void diffTest(const Stack& stack, double hE, Data* data, double h, int32_t k, const std::string& name);
	void diffTest(const Stack& stack, double hE, Data* data, double h, Eigen::MatrixXd Stack::*M, const std::string& name);

	// Applies AdaGrad
	void gradientDescent(const Stack& dX, double r);

	// Classifies data
	void predict(Data* data);

	// Copy/Reverse state to store e.g. best params
	Stack backupState() const;
	void reverseState(const Stack& stack);

	///////////////////////////////////////////////////////////////////////////////
public:

	double lambda; // Regulariser strength
	bool verbose; // Whether difftest reports are wordy
	bool adaGrad; // Whether we are currently using adagrad

	///////////////////////////////////////////////////////////////////////////////
protected:

	int32_t _d; // Word dimension
	int32_t _c; // Number of classes
	int32_t _v; // Vocabulary size

	///////////////////////////////////////////////////////////////////////////////
private:

	Stack _X;
	Stack _G;
};
