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

The program has no command line parser yet (WIP), options are available in main function (sorry).
It includes a translator from twitter dataset to PTB, finite difference test and validation.

The very lacking performance points to a fundamental error somewhere in the implementation,
but this part of the project has been dropped due to bad performance with other numpy implementations as well.
