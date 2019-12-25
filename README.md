SVR
================

[![Build Status](https://travis-ci.org/madsjulia/SVR.jl.svg?branch=master)](https://travis-ci.org/madsjulia/SVR.jl)
[![Coverage Status](https://coveralls.io/repos/madsjulia/SVR.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/SVR.jl?branch=master)

Support Vector Regression (SVR) analysis in [Julia](http://julialang.org) utilizing the [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library.

SVR is a module of [MADS](http://madsjulia.github.io/Mads.jl) (Model Analysis & Decision Support).

Installation
------------

```julia
import Pkg; Pkg.add("SVR")
```

Examples
-------

Matching sine function:

```julia
import SVR
import Mads

X = sort(rand(40) * 5)
y = sin.(X)
```

Predict `y` based on `X` using `RBF`

```
Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.RBF)], "figures/rbf.png"; title="RBF", names=["Truth", "Prediction"])
```


Predict `y` based on `X` using `LINEAR`

```
Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.LINEAR)], "figures/linear.png"; title="Linear", names=["Truth", "Prediction"])

```

Predict `y` based on `X` using `POLY`

```
Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.POLY, coef0=1.)], "figures/poly.png"; title="Polynomial", names=["Truth", "Prediction"])
```

libSVM test example:

```julia
import SVR

x, y = SVR.readlibsvmfile(joinpath(dirname(pathof(SVR)), "..", "test", "mg.libsvm")) # read a libSVM input file

pmodel = SVR.train(y, permutedims(x)) # train a libSVM model

y_pr = SVR.predict(pmodel, permutedims(x)); # predict based on the libSVM model

SVR.savemodel(pmodel, "mg.model") # save the libSVM model

SVR.freemodel(pmodel) # free the memory allocation of the libSVM model
```

Projects using SVR
-----------------

* [MADS](https://github.com/madsjulia)
* [TensorDecompositions](https://github.com/TensorDecompositions)

Publications, Presentations, Projects
--------------------------

* [mads.gitlab.io](http://mads.gitlab.io)
* [TensorDecompositions](https://tensordecompositions.github.io)
* [monty.gitlab.io](http://monty.gitlab.io)
* [ees.lanl.gov/monty](https://www.lanl.gov/orgs/ees/staff/monty)
