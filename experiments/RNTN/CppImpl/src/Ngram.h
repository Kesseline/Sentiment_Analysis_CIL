#pragma once

#include "Model.h"

typedef Eigen::Matrix<int32_t, -1, 2, Eigen::RowMajor> PairMatrixXi;

//#define ROOT_ONLY
#define ROOT_INDEX 0

class Database;

class Ngram : public Data
{
public:

	Ngram(const std::vector<int32_t>& leaves, int32_t label);
	Ngram(const std::vector<Eigen::Vector2i>& children, const std::vector<int32_t>& labels, const std::vector<int32_t>& indices);	
	virtual ~Ngram();

	void arrange(std::mt19937& engine);

	// Build classifications and error
	virtual double build(Model* model) override;
	double build(Model* model, int32_t index);

	// Build derivatives
	virtual void derivative(Stack& stack, Model* model) override;
	void derivative(Stack& stack, Model* model, int32_t index, const Eigen::VectorXd& e);	// Backpropagation
	
	// Check prediction for correctness
	bool hasCorrectRootPrediction() const;
	int32_t getPredictedLabel() const;
	Eigen::VectorXd getPredictedProbs() const;

	// Print current values
	void print() override;

	// Print tree in PTB format
	void printTree(std::ostream& stream, Database* database) const;
	void printTree(std::ostream& stream, int32_t index, Database* database) const;

	///////////////////////////////////////////////////////////////////////////////
protected:

	int32_t _n; // Number of nodes
	int32_t _p; // Number of parent nodes
	Eigen::VectorXi _I; // Word indices (n-p x 1)
	PairMatrixXi _C; // Children for each node (p x 2)
	Eigen::VectorXi _T; // Targets (n x 1)
	// IMPORTANT: Targets *CAN ONLY* have one 1.0 and only zeroes for the rest, so they are defined as index instead

	///////////////////////////////////////////////////////////////////////////////
private:

	Eigen::MatrixXd _P; // Node embeddings with activation (d x n)
	Eigen::MatrixXd _L; // Concatenated child embeddings (2d+1 x p)
	Eigen::MatrixXd _Y; // Classifications (c x n)
	Eigen::MatrixXd _D; // Error (d x n)
};

