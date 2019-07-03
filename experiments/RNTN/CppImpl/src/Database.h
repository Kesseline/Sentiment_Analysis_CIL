#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "Ngram.h"

struct Entry
{
	int32_t count = 1;
	int32_t index = -1;
};

class Database
{
	public:
	   Database();
	   virtual ~Database();

	   // Insert list of words into database
	   void insertWords(const std::vector<std::string>& words, int32_t label);
	   void insertTest(const std::vector<std::string>& words, std::mt19937& engine);

	   // Insert word into dictionary if not exists and returns dequence index (DO NOT mix with insertWords)
	   int32_t insertWord(const std::string& word);

	   // Sort data by word-count into sequence, cutoff at given threshold
	   void generateSequence(int32_t threshold);

	   // Creates list of ngrams based on data and dictionary
	   void generateNgrams(std::mt19937& engine);
	   void generateNgram(const std::pair<std::vector<std::string>, int32_t>& words, std::mt19937& engine);

	   // Adds ngram to database (DO NOT mix with generateNgrams)
	   void addNgram(Ngram* ngram);

	   // Shuffles list of ngrams up to n, also rearranges ngrams if specified
	   void shuffleNgrams(size_t n, bool rearrange, std::mt19937& engine);

	   // Get word from dictionary with index
	   std::string getDictionaryWord(int32_t index) const;

	   // Get number of words in dictionary
	   int32_t getDictionarySize() const;

	   // Ngram access (loops)
	   int32_t getNgramCount();
	   Ngram* getNgram(int32_t index);

	   ///////////////////////////////////////////////////////////////////////////////
	private:

		std::unordered_map<std::string, Entry> _dictionary;
		std::vector<std::pair<std::string, int32_t>> _sequence;
		std::vector<std::pair<std::vector<std::string>, int32_t>> _data;
		std::vector<Ngram*> _ngrams;
};
