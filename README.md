SVR
================

[![SVR](http://pkg.julialang.org/badges/SVR_0.5.svg)](http://pkg.julialang.org/?pkg=SVR&ver=0.5)
[![SVR](http://pkg.julialang.org/badges/SVR_0.6.svg)](http://pkg.julialang.org/?pkg=SVR&ver=0.6)
[![SVR](http://pkg.julialang.org/badges/SVR_0.7.svg)](http://pkg.julialang.org/?pkg=SVR&ver=0.7)
[![Build Status](https://travis-ci.org/madsjulia/SVR.jl.svg?branch=master)](https://travis-ci.org/madsjulia/SVR.jl)
[![Coverage Status](https://coveralls.io/repos/madsjulia/SVR.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/SVR.jl?branch=master)

Support Vector Regression (SVR) analysis in [Julia](http://julialang.org) utilizing the [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library.

SVR is a module of [MADS](http://madsjulia.github.io/Mads.jl) (Model Analysis & Decision Support).

Installation
------------

```julia
import SVR; Pkg.add("SVR")
```

Example
-------

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