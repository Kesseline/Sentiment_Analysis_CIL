# Custom implementation instead of 'CountVectorizer' to have more fine-grained control
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

maxGroupSize = 2
writeReport = True
maxFeatures = 2000000
cutoff = 5

# Generate groups from a list of words
def generateGroups(prev, next, index):
	
	# Finally return accumulated list of groups
	if index >= len(next):
		return prev
	
	ratio = index / len(next)
		
	# Add extended version of previously established groups
	ext = [(l + [next[index]], abs(ratio - r)) for (l, r) in prev if len(l) < maxGroupSize];
	
	# Process words ahead
	return generateGroups(prev + ext, next, index + 1)

	
# Generate sparse matrix from dataset (list of wordlists, *not strings*)
def vectorise(data):

	# Sort features
	print("Create feature space...")
	
	dict = {} # Keeps track of found features and how often they appear
	samples = [] # Caches samples with their labels
	for words in data:

		# Cache groups for later use
		rowSamples = []

		# Generate a list of group-lists, empty list makes sure single words are also added as features
		groups = generateGroups([([], 0.0)], words, 0)
		for (group, r) in groups:

			# Make string out of group and add to dictionary
			feature = '_'.join(group)
			rowSamples.append((feature, r))

			if not feature in dict:
				dict[feature] = 1
			else:
				dict[feature] += 1

		samples.append(rowSamples)
	
	# Sort features
	print("Sort %d features..." % len(dict))
	
	# Create list of tuples so we can sort features by number of occurance
	sorted = []
	for key in dict:
		if dict[key] >= cutoff and not key == "":
			sorted.append((key, dict[key]))
	
	sorted.sort(key=lambda entry: entry[1], reverse=True);
	sorted = sorted[:min(len(sorted), maxFeatures)]
	
	if writeReport:

		# Write report about features
		print("Building report from %d words..." % len(sorted))

		print("Writing to file...")
		with open('report.txt', 'w') as file:
			file.write('\n'.join('%s %d' % entry for entry in sorted))

	# Assign index to features
	print("Building features from %d words..." % len(sorted))
	features = {}
	for i, (key, c) in enumerate(sorted):
		features[key] = i;
	
	# Build data matrix
	print("Building data from %d features..." % len(features))
	indptr = [0]
	indices = []
	data = []
	for words in samples :
		acc = []
		for (word, r) in words :
			if word in features:
				acc.append((features[word], 1.0, 1.0 + 4 * (r * r)))
		acc.sort(key=lambda entry: entry[0], reverse=False);
		for (i, d, e) in acc:
			indices.append(i*2)
			data.append(d)
			indices.append(i*2+1)
			data.append(e)
		indptr.append(len(indices))
		
	return csr_matrix((data, indices, indptr), dtype=float)
	
	
	
	