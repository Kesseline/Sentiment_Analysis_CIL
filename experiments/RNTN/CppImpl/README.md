# RNTN

RNTN implementation with Eigen based on this paper:
https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

Use cmake with your generator of choice (Project has been built and tested on windows, Visual Studio 15 2017)
```
mkdir build
cd build
cmake ..
```
This should automatically download eigen, otherwise clone into ./eigen from:
https://github.com/eigenteam/eigen-git-mirror

The program has no command line parser yet (WIP), options can be change in main function.
It includes a translator from twitter dataset to PTB, finite difference test and validation.

This classifier only achieves up to 75% accuracy and is not used for the ensemble.
