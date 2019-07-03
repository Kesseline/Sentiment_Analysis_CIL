#include "Database.h"
#include <algorithm>

Database::Database()
{
	_dictionary.clear();
}

Database::~Database()
{
	for (Ngram* ngram : _ngrams)
	{
		delete(ngram);
	}
	_ngrams.clear();
}

void Database::insertWords(const std::vector<std::string>& words, int32_t label)
{
	_data.push_back(std::pair<std::vector<std::string>, int32_t>(words, label));
	for (const std::string& word : words)
	{
		// Add to dictionary or increase count if already in dictionary
		std::unordered_map<std::string, Entry>::iterator it = _dictionary.find(word);
		if (it != _dictionary.end())
		{
			it->second.count++;
		}
		else
		{
			_dictionary.emplace(word, Entry());
		}
	}
}

void Database::insertTest(const std::vector<std::string>& words, std::mt19937& engine)
{
	std::pair<std::vector<std::string>, int32_t> pair(words, 0);
	_data.push_back(pair);
	generateNgram(pair, engine);
}

int32_t Database::insertWord(const std::string& word)
{
	std::unordered_map<std::string, Entry>::iterator it = _dictionary.find(word);
	if (it != _dictionary.end())
	{
		// Add to existing entry
		it->second.count++;
		return it->second.index;
	}
	else
	{
		// Add to disctionary and sequence
		Entry entry;
		entry.index = (int32_t)_sequence.size();
		_sequence.push_back(std::pair<std::string, int32_t>(word, -1));
		_dictionary.emplace(word, entry);
		return entry.index;
	}
}

void Database::generateSequence(int32_t threshold)
{
	// Generate sequence structure
	_sequence.clear();
	_sequence.reserve(_dictionary.size());
	for (auto it = _dictionary.begin(); it != _dictionary.end(); it++)
	{
		_sequence.push_back(std::pair<std::string, int32_t>(it->first, it->second.count));
	}

	// Sort by number of occurences
	std::sort(_sequence.begin(), _sequence.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

	// Assign indices, remove entries below threshold
	for (int i = 0; i < _sequence.size(); i++)
	{
		const auto& pair = _sequence[i];
		if (pair.second < threshold)
		{
			_sequence.resize(i);
			break;
		}
		else
		{
			_dictionary[pair.first].index = i;
		}
	}
}


void Database::generateNgrams(std::mt19937& engine)
{
	// Clear previous batch
	for (Ngram* ngram : _ngrams)
	{
		delete(ngram);
	}
	_ngrams.clear();

	for (const std::pair<std::vector<std::string>, int32_t>& words : _data)
	{
		generateNgram(words, engine);
	}
}

void Database::generateNgram(const std::pair<std::vector<std::string>, int32_t>& words, std::mt19937& engine)
{
	// Find words in dictionary and assign indices
	std::vector<int32_t> indices;
	for (const std::string& word : words.first)
	{
		std::unordered_map<std::string, Entry>::iterator it = _dictionary.find(word);
		if (it != _dictionary.end())
		{
			if (it->second.index != -1)
			{
				indices.push_back(it->second.index);
			}
		}
	}

	// Only add ngrams is any word is used
	if (indices.size() > 0)
	{
		Ngram* ngram = new Ngram(indices, words.second);
		ngram->arrange(engine);
		_ngrams.push_back(ngram);
	}
}

void Database::addNgram(Ngram* ngram)
{
	_ngrams.push_back(ngram);
}

void Database::shuffleNgrams(size_t n, bool rearrange, std::mt19937& engine)
{
	n = std::min(n, _ngrams.size());
	std::shuffle(std::begin(_ngrams), std::begin(_ngrams) + n, engine);

	if (rearrange)
	{
		for (int i = 0; i < n; i++)
		{
			_ngrams[i]->arrange(engine);
		}
	}
}

std::string Database::getDictionaryWord(int32_t index) const
{
	return _sequence[index].first;
}

int32_t Database::getDictionarySize() const
{
	return (int32_t)_sequence.size();
}

int32_t Database::getNgramCount()
{
	return (int32_t)_ngrams.size();
}

Ngram* Database::getNgram(int32_t index)
{
	return _ngrams[index];
}