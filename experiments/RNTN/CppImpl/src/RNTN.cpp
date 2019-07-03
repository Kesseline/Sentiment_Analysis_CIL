#include <iostream>
#include <fstream>
#include <omp.h>
#include <sstream>

#include "Database.h"




int32_t readSubTree(Database* database, std::basic_istringstream<char>& buf, std::vector<Eigen::Vector2i>& children, std::vector<int32_t>& labels, std::vector<int32_t>& indices)
{
	char p;
	buf >> p; // (

	int num;
	buf >> num;
	labels.push_back(num);

	buf.get();
	char c = std::char_traits<char>::to_char_type(buf.peek());
	if (c == '(')
	{
		const int32_t i = (int32_t)children.size();
		children.push_back(Eigen::Vector2i());

		const int32_t a = readSubTree(database, buf, children, labels, indices);
		const int32_t b = readSubTree(database, buf, children, labels, indices);

		children[i] = Eigen::Vector2i(a, b);

		buf >> p; // ')'
		return i;
	}
	else
	{
		std::string word = "";
		buf >> p;
		while (p != ')')
		{
			word.push_back(p);
			buf >> p;
		}
		indices.push_back(database->insertWord(word));
		return -(int32_t)indices.size() + 1; // Store index as negative number
	}
}


bool readTrees(Database* database, const std::string& filename, int32_t maxLines)
{
	std::ifstream file(filename);

	if (!file)
	{
		std::cout << "Opening file failed!" << std::endl;
		return false;
	}
	std::cout << "Reading " + filename + " into database..." << std::endl;

	// Read every line into data array
	std::string str;
	int32_t lineCount = 0;
	while (std::getline(file, str))
	{
		std::vector<Eigen::Vector2i> children;
		std::vector<int32_t> labels;
		std::vector<int32_t> indices;
		readSubTree(database, std::basic_istringstream<char>(str), children, labels, indices);
		database->addNgram(new Ngram(children, labels, indices));

		if ((++lineCount % 250) == 0)
		{
			std::cout << ".";
		}

		if (lineCount >= maxLines)
		{
			break;
		}
	}
	file.close();
	std::cout << std::endl;
	return true;
}

bool writeTrees(Database* database, const std::string& filename, int32_t offset, int32_t n)
{
	std::ofstream file(filename);

	if (!file)
	{
		std::cout << "Opening file failed!" << std::endl;
		return false;
	}
	std::cout << "Writing " + filename + " from database..." << std::endl;
	for (int i = offset; i < offset + n; i++)
	{
		database->getNgram(i)->printTree(file, database);
		file << std::endl;
	}

	file.close();
	return true;
}

bool readLines(Database* database, const std::string& filename, int32_t label, int32_t maxLines, std::mt19937& engine)
{
	std::ifstream file(filename);

	if (!file)
	{
		std::cout << "Opening file failed!" << std::endl;
		return false;
	}
	std::cout << "Reading " + filename + " into database..." << std::endl;

	// Read every line into data array
	std::string str;
	int32_t lineCount = 0;
	while (std::getline(file, str))
	{
		std::vector<std::string> words;

		size_t last = 0;
		size_t next = 0;
		while ((next = str.find(' ', last)) != std::string::npos)
		{
			const std::string word = str.substr(last, next - last);
			words.push_back(word);
			last = next + 1;
		}
		words.push_back(str.substr(last));

		if (label >= 0)
		{
			database->insertWords(words, label);
		}
		else
		{
			database->insertTest(words, engine);
		}

		if ((++lineCount % 250) == 0)
		{
			std::cout << ".";
		}

		if (lineCount >= maxLines)
		{
			break;
		}
	}
	file.close();
	std::cout << std::endl;
	return true;
}

// Compute total energy over all ngrams
double totalEnergy(Database* database, Model* model, int32_t offset, int32_t n)
{
	double energy = 0.0;
	#pragma omp parallel for reduction( + : energy )
	for (int i = offset; i < offset + n; i++)
	{
		Ngram* ngram = database->getNgram(i);
		energy += ngram->build(model);
	}
	return energy;
}


int main(int argc, const char* argv[])
{
	const bool convertTreeFile = false; // Convert files into PTB instead of training
	const bool useTreeFile = false; // Whether to load PTB files instead of the twitter dataset
	const bool finiteDifferenceTest = false; // Whether to apply finite different test to validate derivatives
	const bool batchProcessing = true; // Batch processing is parallelized and gives more stable results
	const bool variableStepSize = true; // Applying only gradients that decrease energy
	const bool reshuffleNgrams = true; // Reshuffle not only the ngram order but also the tree structure
	const bool applySelfTest = false; // Whether to test on the training set

	const int32_t numThreads = 8;
	const int32_t trainPercent = 90;
	const int32_t batch = 192;
	const int32_t epochs = 20;

	const int32_t dimensions = 30;
	const int32_t classes = 2;
	const double randScale = 0.001; // Paper: 0.0001
	const double regulariser = 1e-6;

	const double learnrate = 0.01;

	const std::string twitterTest = "../../../../data/test_data.txt";
	const std::string twitterPositive = "../../../../data/train_pos.txt";
	const std::string twitterNegative = "../../../../data/train_neg.txt";

	const std::string treeTest = "../../../../RNTN/RNTN/trees/test.txt";
	const std::string treeTrain = "../../../../RNTN/RNTN/trees/train.txt";

	const std::string outputLabels = "out.txt";
	const std::string outputProbs = "out_prod.txt";

	// Random generator for reproducible results
	srand(0);
	std::random_device randomDevice;
	std::mt19937 randomEngine(randomDevice());
	randomEngine.seed(42);

	Database* database = new Database();
	if (convertTreeFile)
	{
		// Conversion code
		if (!readLines(database, twitterPositive, 0, 10000, randomEngine)) return 1;
		if (!readLines(database, twitterNegative, 1, 10000, randomEngine)) return 1;
		database->generateSequence(0);
		database->generateNgrams(randomEngine);
		const int32_t n = database->getNgramCount();
		database->shuffleNgrams(n, false, randomEngine);

		const int32_t train = n * trainPercent / 100;
		const int32_t test = n - train;

		writeTrees(database, "train_generated.txt", 0, train);
		writeTrees(database, "test_generated.txt", train, test);

		delete database;
		return 0;
	}

	int32_t trainCount = 0;
	if (useTreeFile)
	{
		if (finiteDifferenceTest)
		{
			if (!readTrees(database, treeTrain, 1)) return 1;
		}
		else
		{
			if (!readTrees(database, treeTrain, 10000)) return 1;
		}
		trainCount = database->getNgramCount();
		if (!readTrees(database, treeTest, 100000)) return 1;
	}
	else
	{
		// Only load few words when doing finite difference test
		// (Test iterates over *all* words)
		if (finiteDifferenceTest)
		{
			//database->insertWords({"1", "2"}, 1);
			//database->insertWords({ "1", "2", "3" }, 1);
			if (!readLines(database, twitterPositive, 0, 2, randomEngine)) return 1;
			if (!readLines(database, twitterNegative, 1, 2, randomEngine)) return 1;
			database->generateSequence(0);
		}
		else
		{
			if (!readLines(database, twitterPositive, 0, 20000, randomEngine)) return 1;
			if (!readLines(database, twitterNegative, 1, 20000, randomEngine)) return 1;

			std::cout << "Generating sequence..." << std::endl;
			database->generateSequence(2);
		}

		std::cout << "Generating ngrams..." << std::endl;
		database->generateNgrams(randomEngine);
		trainCount = database->getNgramCount();

		if (!readLines(database, twitterTest, -1, 100000, randomEngine)) return 1;
	}

	std::cout << "Creating model..." << std::endl;
	const int32_t v = database->getDictionarySize();
	const int32_t n = database->getNgramCount();

	omp_set_num_threads(numThreads);
	const int32_t thrn = batch / numThreads;
	if (batch % numThreads != 0)
	{
		std::cerr << "batch needs to be a multiple of numThreads!" << std::endl;
		exit(1);
	}


	if (finiteDifferenceTest)
	{
		Model* model = new Model(v, 4, classes, randScale, 0.001);

		std::cout << "Testing..." << std::endl;
		Ngram* ngram = database->getNgram(0);
		//ngram->print();
		model->diffTest(ngram, 0.00001);
		return 0;
	}


	const int32_t train = trainCount * trainPercent / 100;
	const int32_t test = trainCount - train;
	std::cout << "Training on " << train << " lines..." << std::endl;
	database->shuffleNgrams(trainCount, false, randomEngine);

	// For batch learning we omit the rest of the batch, chances are high it gets picked up by another epoch after shuffle
	const int32_t batches = train / batch;

	Model* model = new Model(v, dimensions, classes, randScale, regulariser);
	model->resetAda();

	// Keep track of best attempt
	int32_t bestCount = test / 2;
	Stack bestStack = model->backupState();

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "Shuffle epoch " << epoch << "..." << std::endl;
		database->shuffleNgrams(train, reshuffleNgrams, randomEngine);

		std::cout << "Training epoch " << epoch << "..." << std::endl;

		if (batchProcessing)
		{
			double r = learnrate / batch;
			for (int b = 0; b < batches; b++)
			{
				Stack dX(v, dimensions, classes);
				dX.setZero();

				const double energy = totalEnergy(database, model, b * batch, batch);

				///////////////////////
				#pragma omp parallel for
				for (int k = 0; k < numThreads; k++)
				{
					Stack local(v, dimensions, classes);
					local.setZero();

					const int32_t start = b * batch + thrn * k;
					for (int j = start; j < start + thrn; j++)
					{
						Ngram* ngram = database->getNgram(j);
						const double energy = ngram->build(model);
						if (isnan(energy))
						{
							std::cout << "NAN detected!" << energy << std::endl;
						}
						ngram->derivative(local, model);
					}

					#pragma omp critical
					dX += local;
				}
				///////////////////////

				// Apply gradient
				model->gradientDescent(dX, r);

				if (variableStepSize)
				{
					// Flip r to backtrack gradient until error is small enough
					// (Only works correctly if we half the learning rate)
					while (totalEnergy(database, model, b * batch, batch) > energy && r > 1e-5)
					{
						r /= 2;
						model->gradientDescent(dX, -r);
					}
					r *= 1.2;
				}
				std::cout << ".";
			}
			std::cout << r << std::endl;
			std::cout << std::endl;
		}
		else
		{
			double r = learnrate;
			for (int i = 0; i < train; i++)
			{
				Stack dX(v, dimensions, classes);
				dX.setZero();

				Ngram* ngram = database->getNgram(i);
				const double energy = ngram->build(model);
				if (isnan(energy))
				{
					std::cout << "NAN detected!" << energy << std::endl;
				}
				ngram->derivative(dX, model);
				model->gradientDescent(dX, r);

				if (variableStepSize)
				{
					while (ngram->build(model) > energy && r > 1e-5)
					{
						r /= 2;
						model->gradientDescent(dX, -r);
					}
					r *= 1.2;
				}
			}
			std::cout << r << std::endl;
		}

		std::cout << "Validating epoch " << epoch << "..." << std::endl;
		int32_t correct = 0; // (TP + TN)
		const int32_t offset = applySelfTest ? 0 : train;
		for (int i = offset; i < offset + test; i++)
		{
			Ngram* ngram = database->getNgram(i);
			model->predict(ngram);
			if (ngram->hasCorrectRootPrediction())
			{
				correct++;
			}
		}

		// Evaluate test-set
		if (correct > bestCount)
		{
			bestStack = model->backupState();
		}

		std::cout << "correctly predicted " << correct << "/" << test << " or " << (correct * 100 / test) << "%" << std::endl;
	}

	// Create output file
	std::ofstream fileLabels(outputLabels);
	if (!fileLabels)
	{
		std::cout << "Opening label file failed!" << std::endl;
		return false;
	}
	fileLabels << "Id, Prediction" << std::endl;

	std::ofstream fileProbs(outputProbs);
	if (!fileProbs)
	{
		std::cout << "Opening probs file failed!" << std::endl;
		return false;
	}
	fileProbs << "Id, ppos, pneg" << std::endl;

	std::cout << "Writing " + outputProbs + " from prediction..." << std::endl;

	model->reverseState(bestStack);
	for (int i = trainCount; i < n; i++)
	{
		Ngram* ngram = database->getNgram(i);
		model->predict(ngram);

		const int32_t label = ngram->getPredictedLabel();
		const Eigen::VectorXd probs = ngram->getPredictedProbs();

		if (useTreeFile)
		{
		}
		else
		{
			fileLabels << i << ", " << (label == 0 ? 1 : -1) << std::endl;
			fileProbs << i << ", " << probs[0] << ", " << probs[1] << std::endl;
		}
	}

	fileProbs.close();
	fileLabels.close();

	delete(model);
	delete(database);
	
	return 0;
}