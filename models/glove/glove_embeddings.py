from scipy.sparse import *
import numpy as np
import pickle
import random


def create_embeddings(cooc_path, embeddings_path):
    # Create Glove embeddings using the previously computed cooc-matrix

    print("Loading cooccurrence matrix")
    with open(cooc_path, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    # Cutoff to reduce weights of high-frequency words
    nmax = 100000
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("Initializing embeddings");
    print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
    embedding_dim = 300
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 20

    # Main loop as discussed in lecture
    print("Computing embeddings")

    for epoch in range(epochs):
        print("epoch {}".format(epoch))

        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x

    np.savez(embeddings_path, xs, ys)
    print("Embeddings successfully computed and written")


if __name__ == '__main__':
    create_embeddings()
