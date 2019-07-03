import pickle
import subprocess

import os

def generate_vocab(pathpos, pathneg, vocab_path):
    vocab = dict()
    
    base = os.path.dirname(os.path.abspath(__file__))
    vocab_txt_path = base + "/output/vocab.txt"
    vocab_cut_path = base + "/output/vocab_cut.txt"
    
    bashCommand = "sh %s/build_vocab.sh %s %s %s" % (base, pathpos, pathneg, vocab_txt_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print (error)
    
    bashCommand = "sh %s/cut_vocab.sh %s %s" % (base, vocab_txt_path, vocab_cut_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print (error)
    
    with open(vocab_cut_path) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
        