#include "Ngram.h"
#include "Database.h"
#include <random>


Ngram::Ngram(const std::vector<int32_t>& leaves, int32_t label)
{
	// Allocate dimensions
	const int32_t count = (int32_t)leaves.size();
	_p = count - 1;
	_n = _p * 2 + 1;

	// Allocate and assign leaves
	_I = Eigen::VectorXi(count);
	for (int i = 0; i < count; i++)
	{
		_I[i] = leaves[i];
	}

	// Reallocate children
	_C = PairMatrixXi(_p, 2);

	// Set labels
	_T = Eigen::VectorXi(_n);
	_T.setConstant(label);
}

Ngram::Ngram(const std::vector<Eigen::Vector2i>& children, const std::vector<int32_t>& labels, const std::vector<int32_t>& indices)
{
	// Allocate dimensions
	const int32_t count = (int32_t)indices.size();
	_p = count - 1;
	_n = _p * 2 + 1;

	_I = Eigen::VectorXi(count);
	_C = PairMatrixXi(_p, 2);
	_T = Eigen::VectorXi(_n);

	// Assign leaves (Make proper index for leaves if not defined yet)
	for (int i = 0; i < _p; i++)
	{
		Eigen::Vector2i p = children[i];
		if (p.x() <= 0) { p.x() = (int32_t)children.size() - p.x(); }
		if (p.y() <= 0) { p.y() = (int32_t)children.size() - p.y(); }
		_C.row(i) = p;
	}

	// Assign leaves
	for (int i = 0; i < count; i++)
	{
		_I[i] = indices[i];
	}

	// Assign labels
	for (int i = 0; i < _n; i++)
	{
		_T[i] = labels[i];
	}
}

Ngram::~Ngram()
{
}

void Ngram::arrange(std::mt19937& engine)
{
	// Recursive tree builder
	int32_t counter = 0;
	std::function<int(int, int)> tree = [&](int32_t a, int32_t b) -> int32_t
	{
		if (a == b)
		{
			// Leaf node with proper offset
			return _p + a;
		}
		else
		{
			// Split range randomly
			std::uniform_int_distribution<int32_t> dist(a, b - 1);
			const int32_t split = dist(engine);

			// Divide and conquer
			const int32_t index = counter++;
			_C.row(index).x() = tree(a, split);
			_C.row(index).y() = tree(split + 1, b);
			return index;
		}
	};

	// Build tree
	tree(0, _p);
}

double Ngram::build(Model* model)
{
	const int32_t d = model->getWordDim();
	const int32_t c = model->getClassDim();

	// Allocate and set bias
	_P = Eigen::MatrixXd(d, _n);
	_L = ColMatrixXd(d * 2, _p);

	_D = Eigen::MatrixXd(d, _n);
	_Y = Eigen::MatrixXd(c, _n);
	
	// Initialise children (starting at root)
	const double E = build(model, ROOT_INDEX);

	// Add regulariser
	return E + model->regulariser();
}

double Ngram::build(Model* model, int32_t index)
{
	double E = 0.0;
	const int32_t d = model->getWordDim();

	// Compute
	Eigen::VectorXd out(d);
	const int32_t a = index - _p;
	if (a >= 0)
	{
		// Grab from database
		out = model->getL(_I[a]);
	}
	else
	{
		// Compute from children
		const Eigen::RowVector2i& r = _C.row(index);
		auto& x = _L.col(index);
		E += build(model, r.x());
		E += build(model, r.y());

		x.segment(0, d) = _P.col(r.x());
		x.segment(d, d) = _P.col(r.y());

		// Compute tensor product
		Eigen::VectorXd h = Eigen::VectorXd::Zero(d);
		for (int i = 0; i < d; i++)
		{
			h[i] = x.transpose() * model->getV(i) * x;
		}

		out = model->f((h + model->getW() * x + model->getb()));
	}

	_P.col(index) = out;

	// Classify node
	const Eigen::VectorXd el = (model->getWs() * out) + model->getbs();
	const Eigen::VectorXd em = el.array() - el.maxCoeff(); // For better numeric stability
	const Eigen::VectorXd ee = el.unaryExpr([](double p) -> double { return std::exp(p); });
	_Y.col(index) = ee / ee.sum();

	double e = -std::log(_Y(_T[index], index));
#ifdef ROOT_ONLY
	if (index != 0)
	{
		e = 0.0;
	}
#endif
	if (isnan(e))
	{
		std::cout << "NAN error detected" << std::endl;
	}

	return E + e;
	
}


void Ngram::derivative(Stack& stack, Model* model)
{
	const int32_t d = model->getWordDim();

	// Compute backwards propagation
	derivative(stack, model, ROOT_INDEX, Eigen::VectorXd::Zero(d));

	// Add regulariser
	model->addRegulariserDerivative(stack);
}

void Ngram::derivative(Stack& stack, Model* model, int32_t index, const Eigen::VectorXd& e)
{
	const int32_t d = model->getWordDim();

	const int32_t t = _T[index];
	Eigen::VectorXd y = _Y.col(index);
	y[t] -= 1.0; // y - t

#ifdef ROOT_ONLY
	if (index != 0)
	{
		y.setZero();
	}
#endif

	// Compute derivative for Ws
	stack.Ws += y * _P.col(index).transpose();
	stack.bs += y;

	// Compute error derivative
	const Eigen::VectorXd error = (model->getWs().transpose() * y);
	
	// Leaf nodes contribute to the dictionary
	const int32_t a = index - _p;
	if (a >= 0)
	{
		// Compute complete error vector
		stack.L.col(_I[a]) += error + e;
	}
	else
	{
		const Eigen::VectorXd ecom = (error + e).cwiseProduct(model->dfdx(_P.col(index)));

		// Compute complete error vector
		const auto& l = _L.col(index);

		// Compute and add derivatives
		const Eigen::MatrixXd outer = l * l.transpose();
		for (int i = 0; i < d; i++)
		{
			stack.V[i] += ecom[i] * outer;
		}
		stack.W += ecom * l.transpose();
		stack.b += ecom;

		// Compute e down
		Eigen::VectorXd S = Eigen::VectorXd::Zero(d * 2);
		for (int i = 0; i < d; i++)
		{
			const Eigen::VectorXd v = ecom[i] * l;
			const Eigen::MatrixXd& V = model->getV(i);
			S += (V * v + V.transpose() * v);
		}
		const Eigen::VectorXd edown = model->getW().transpose() * ecom + S;

		// Compute childrend derivatives
		const Eigen::RowVector2i& r = _C.row(index);
		derivative(stack, model, r.x(), edown.segment(0, d));
		derivative(stack, model, r.y(), edown.segment(d, d));
	}
}

bool Ngram::hasCorrectRootPrediction() const
{
	//std::cout << Eigen::RowVectorXd(_Y.col(0)) << " . " << _T[0] << std::endl;

	// Check whether the label was most likely predicted
	return _T[0] == getPredictedLabel();
}

int32_t Ngram::getPredictedLabel() const
{
	int32_t maxIndex = 0;
	const Eigen::VectorXd props = getPredictedProbs();
	for (int i = 1; i < _Y.rows(); i++)
	{
		if (props[i] > props[maxIndex]) maxIndex = i;
	}
	return maxIndex;
}

Eigen::VectorXd Ngram::getPredictedProbs() const
{
	return _Y.col(0);
}

void Ngram::print()
{
	std::cout << "Count: " << _n << std::endl;
	std::cout << "Parents: " << _p << std::endl;

	std::cout << "Indices: " << std::endl;
	std::cout << _I << std::endl;

	std::cout << "Children: " << std::endl;
	std::cout << _C << std::endl;

	std::cout << "Targets: " << std::endl;
	std::cout << _T << std::endl;
}

void Ngram::printTree(std::ostream& stream, Database* database) const
{
	return printTree(stream, ROOT_INDEX, database);
}

void Ngram::printTree(std::ostream& stream, int32_t index, Database* database) const
{
	stream << "(" << _T[index] << " ";

	const int32_t a = index - _p;
	if (a >= 0)
	{
		std::string word = database->getDictionaryWord(_I[a]);
		std::replace(word.begin(), word.end(), '(', '<');
		std::replace(word.begin(), word.end(), ')', '>');
		stream << word;
	}
	else
	{
		const Eigen::RowVector2i& r = _C.row(index);
		printTree(stream, r.x(), database);
		stream << " ";
		printTree(stream, r.y(), database);
	}

	stream << ")";
}
