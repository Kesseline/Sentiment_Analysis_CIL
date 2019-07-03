#include "Model.h"

Stack::Stack(int32_t v, int32_t d, int32_t c)
{
	L = ColMatrixXd(d, v);
	Ws = Eigen::MatrixXd(c, d);
	bs = Eigen::VectorXd(c, 1);
	W = Eigen::MatrixXd(d, d * 2);
	b = Eigen::VectorXd(d, 1);

	V.resize(d);
	for (int i = 0; i < d; i++)
	{
		V[i] = Eigen::MatrixXd(d * 2, d * 2);
	}
}

Stack Stack::operator+(const Stack& other) const
{
	Stack stack(*this);
	return stack += other;
}

Stack Stack::operator+=(const Stack& other)
{
	L += other.L;
	Ws += other.Ws;
	bs += other.bs;
	W += other.W;
	b += other.b;

	for (int i = 0; i < V.size(); i++)
	{
		V[i] += other.V[i];
	}
	return *this;
}

Stack Stack::Zeros(const Stack& other)
{
	Stack stack = other;
	stack.setZero();
	return stack;
}

void Stack::setConstant(double c)
{
	L.setConstant(c);
	Ws.setConstant(c);
	bs.setConstant(c);
	W.setConstant(c);
	b.setConstant(c);

	for (int i = 0; i < V.size(); i++)
	{
		V[i].setConstant(c);
	}
}

void Stack::setRandom(double r)
{
	L.setRandom() *= r;
	Ws.setRandom() *= r;
	bs.setZero();
	W.setRandom() *= r;
	b.setZero();

	for (int i = 0; i < V.size(); i++)
	{
		V[i].setRandom() *= r;
	}
}

void Stack::setZero()
{
	L.setZero();
	Ws.setZero();
	bs.setZero();
	W.setZero();
	b.setZero();

	for (int i = 0; i < V.size(); i++)
	{
		V[i].setZero();
	}
}

void Stack::print() const
{
	std::cout << std::endl;
	std::cout << L << std::endl;
	std::cout << "----------------" << std::endl;
	std::cout << Ws << std::endl;
	std::cout << "----------------" << std::endl;
	std::cout << bs << std::endl;
	std::cout << "----------------" << std::endl;
	std::cout << W << std::endl;
	std::cout << "----------------" << std::endl;
	std::cout << b << std::endl;
	std::cout << "----------------" << std::endl;
	for (int i = 0; i < V.size(); i++)
	{
		std::cout << V[i] << std::endl;
		std::cout << "----------------" << std::endl;
	}
	std::cout << std::endl;
}



Model::Model(int32_t v, int32_t d, int32_t c, double r, double lambda)
	:
	lambda(lambda),
	adaGrad(false),
	verbose(false),
	_v(v), _d(d), _c(c),
	_X(v, d, c), _G(v, d, c)
{
	_X.setRandom(r);
	_G.setZero();
}

Model::~Model()
{

}

int32_t Model::getWordDim() const
{
	return _d;
}

int32_t Model::getClassDim() const
{
	return _c;
}

double Model::f(double x) const
{
	return std::tanh(x);
}

double Model::dfdx(double fx) const
{
	return 1.0 - fx * fx;
}

Eigen::VectorXd Model::f(const Eigen::VectorXd& x) const
{
	//return x * 0.01;
	return x.unaryExpr([this](double p) -> double { return f(p); });
}
Eigen::VectorXd Model::dfdx(const Eigen::VectorXd& fx) const
{
	//return Eigen::VectorXd::Ones(x.size()) * 0.01;
	return fx.unaryExpr([this](double p) -> double { return dfdx(p); });
}


double Model::regulariser()
{
	double E = 0.0;
	for (int i = 0; i < _d; i++)
	{
		E += _X.V[i].squaredNorm() * lambda;
	}
	E += _X.W.squaredNorm() * lambda;
	E += _X.Ws.squaredNorm() * lambda;
	return E;
}

void Model::addRegulariserDerivative(Stack& stack)
{
	for (int i = 0; i < _d; i++)
	{
		stack.V[i] += _X.V[i] * (2 * lambda);
	}
	stack.W += _X.W * (2 * lambda);
	stack.Ws += _X.Ws * (2 * lambda);
}

void Model::resetAda()
{
	// Add fudge factor in case of zero entries
	_G.setConstant(0.000001);
}

const Eigen::VectorXd Model::getL(int32_t i) const
{
	return _X.L.col(i);
}

const Eigen::MatrixXd& Model::getWs() const
{
	return _X.Ws;
}

const Eigen::MatrixXd& Model::getbs() const
{
	return _X.bs;
}

const Eigen::MatrixXd& Model::getW() const
{
	return _X.W;
}

const Eigen::MatrixXd& Model::getb() const
{
	return _X.b;
}

const Eigen::MatrixXd& Model::getV(int32_t i) const
{
	return _X.V[i];
}


void Model::diffTest(const Stack& stack, double hE, Data* data, double h, Eigen::MatrixXd Stack::* M, const std::string& name)
{
	// Finite difference
	Eigen::MatrixXd H = _X.*M;
	for (int i = 0; i < (_X.*M).rows(); i++)
	{
		for (int j = 0; j < (_X.*M).cols(); j++)
		{
			(_X.*M)(i, j) += h;
			const double E = data->build(this);
			(_X.*M)(i, j) -= h;
			H(i, j) = (E - hE) / h;
		}
	}

	// Output results
	if (verbose)
	{
		std::cout << "..........." << name << "..........." << std::endl;
		std::cout << stack.*M << std::endl;
		std::cout << "..........." << std::endl;
		std::cout << H << std::endl;
		std::cout << "..........." << std::endl;
	}
	std::cout << "Squared error " << name << ": " << (H - stack.*M).squaredNorm() << std::endl;
}

void Model::diffTest(const Stack& stack, double hE, Data* data, double h, int32_t k, const std::string& name)
{
	// Finite difference
	Eigen::MatrixXd H = _X.V[k];
	for (int i = 0; i < (_X.V[k]).rows(); i++)
	{
		for (int j = 0; j < (_X.V[k]).cols(); j++)
		{
			(_X.V[k])(i, j) += h;
			const double E = data->build(this);
			(_X.V[k])(i, j) -= h;
			H(i, j) = (E - hE) / h;
		}
	}

	// Output results
	if (verbose)
	{
		std::cout << "..........." << name << " " << k << "..........." << std::endl;
		std::cout << stack.V[k] << std::endl;
		std::cout << "..........." << std::endl;
		std::cout << H << std::endl;
		std::cout << "..........." << std::endl;
	}
	std::cout << "Squared error " << name << " " << k << ": " << (H - stack.V[k]).squaredNorm() << std::endl;
}

void Model::diffTest(Data* data, double h)
{
	/*
	_X.L.setRandom() * .8;
	_X.Ws.setRandom() * .8;
	_X.bs.setRandom() * .8;
	_X.W.setRandom() * .8;
	_X.b.setRandom() * .8;
	for (int i = 0; i < _d; i++)
	{
		_X.V[i].setRandom() * .8;
	}
	*/

	Stack dX(_v, _d, _c);
	dX.setZero();
	// Get initial state
	const double hE = data->build(this);
	data->derivative(dX, this);

	diffTest(dX, hE, data, h, &Stack::L, "dEdL");
	//diffTest(dX, hE, data, h, &Stack::Ws, "dEdWs");
	//diffTest(dX, hE, data, h, &Stack::bs, "dEdbs");
	//diffTest(dX, hE, data, h, &Stack::W, "dEdW");
	//diffTest(dX, hE, data, h, &Stack::b, "dEdb");
	for (int i = 0; i < _d; i++)
	{
		//diffTest(dX, hE, data, h, i, "dEdV");
	}
}


void Model::gradientDescent(const Stack& dX, double r)
{
	if (adaGrad)
	{
		// Adapt G to current gradients
		_G.L += dX.L.cwiseProduct(dX.L);
		_G.Ws += dX.Ws.cwiseProduct(dX.Ws);
		_G.bs += dX.bs.cwiseProduct(dX.bs);
		_G.W += dX.W.cwiseProduct(dX.W);
		_G.b += dX.b.cwiseProduct(dX.b);
		for (int i = 0; i < _d; i++)
		{
			_G.V[i] += dX.V[i].cwiseProduct(dX.V[i]);
		}

		// Apply gradient descent
		_X.L -= r * dX.L.cwiseQuotient(_G.L.cwiseSqrt());
		_X.Ws -= r * dX.Ws.cwiseQuotient(_G.Ws.cwiseSqrt());
		_X.bs -= r * dX.bs.cwiseQuotient(_G.bs.cwiseSqrt());
		_X.W -= r * dX.W.cwiseQuotient(_G.W.cwiseSqrt());
		_X.b -= r * dX.b.cwiseQuotient(_G.b.cwiseSqrt());
		for (int i = 0; i < _d; i++)
		{
			_X.V[i] -= r * dX.V[i].cwiseQuotient(_G.V[i].cwiseSqrt());
		}
	}
	else
	{
		// Apply gradient descent
		_X.L -= r * dX.L;
		_X.Ws -= r * dX.Ws;
		_X.bs -= r * dX.bs;
		_X.W -= r * dX.W;
		_X.b -= r * dX.b;
		for (int i = 0; i < _d; i++)
		{
			_X.V[i] -= r * dX.V[i];
		}
	}
}

void Model::predict(Data* data)
{
	data->build(this);
}


Stack Model::backupState() const
{
	return _X;
}

void Model::reverseState(const Stack& stack)
{
	_X = stack;
}
