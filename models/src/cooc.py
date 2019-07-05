from scipy.sparse import *  
import numpy as np
import pickle


def create_cooc(pathpos, pathneg, vocab_path, cooc_path):
    # Create Cooc-Matrix from extracted vocabulary
    
    print("Loading vocabulary")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    cooc = coo_matrix((vocab_size, vocab_size))

    counter = 1
    window_size = 5
    batch_size = 50000
    print("Computing Cooc-Matrix with batch size %s and window size %s" % (batch_size,window_size))

    # We use batches and windows to reduce memory demand
    # Windows use the rational that the correlation of words is stronger if they are closer together

    # Loop positive and negative tweets
    for fn in [pathpos, pathneg]:
        with open(fn) as f:
            lines = f.readlines()

            # Loop through lines of file in batches
            for i in range(len(lines) // batch_size + 1):
                data, row, col = list(cooc.data), list(cooc.row), list(cooc.col)
                
                # Loop through tokens of line
                for line in lines[i*batch_size:(i+1)*batch_size]:
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]

                    # Save recognized tokens
                    for idx, t in enumerate(tokens):
                        for t2 in tokens[idx-window_size:idx+window_size]:
                            data.append(1)
                            row.append(t)
                            col.append(t2)

                    if counter % 10000 == 0:
                        print(counter)
                    counter += 1
                
                # Update Matrix
                cooc = coo_matrix((data,(row,col)))
                cooc.sum_duplicates()
        
    print("Writing Cooc-Matrix")
    with open(cooc_path, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

    print("Cooc-Matrix successfully computed and written")


if __name__ == '__main__':
    create_cooc()
