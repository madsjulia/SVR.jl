SVR
================

[![SVR](http://pkg.julialang.org/badges/SVR_0.5.svg)](http://pkg.julialang.org/?pkg=SVR&ver=0.5)

[![Build Status](https://travis-ci.org/madsjulia/SVR.jl.svg?branch=master)](https://travis-ci.org/madsjulia/SVR.jl)

[![Coverage Status](https://coveralls.io/repos/madsjulia/SVR.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/SVR.jl?branch=master)

This package perfoms Support Vector Regression (SVR) analysis using [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library in [Julia](http://julialang.org).

SVR is a module of MADS.

Installation
------------

```julia
Pkg.add("SVR")
```

Example
-------

```julia
import SVR

# read a libSVM input file
x, y = SVR.readlibsvmfile("mg.libsvm")

# train a libSVM model
pmodel = SVR.train(y, x');

# train based on the libSVM model
y_pr = SVR.predict(pmodel, x');

# save the libSVM model
SVR.savemodel(pmodel, "mg.model")

# free the memory allocation of the libSVM model
SVR.freemodel(pmodel)
```

MADS
====

MADS is an integrated open-source high-performance computational (HPC) framework written in [Julia](http://julialang.org) performing a wide range of data- and model-based analyses:

* Sensitivity Analysis
* Parameter Estimation
* Model Inversion and Calibration
* Uncertainty Quantification
* Model Selection and Averaging
* Model Reduction and Surrogate Modeling
* Machine Learning and Blind Source Separation
* Decision Support

MADS utilizes adaptive rules and techniques which allows the analyses to be performed with minimum user input.
The code provides a series of alternative algorithms to perform each type of data- and model-based analyses.

Documentation
=============

All the available MADS modules and functions are described at [madsjulia.github.io](http://madsjulia.github.io/Mads.jl)

Installation
------------

```julia
Pkg.add("Mads")
```

Installation of MADS behind a firewall
------------------------------

Julia uses git for package management. Add in the `.gitconfig` file in your home directory:

```
[url "https://"]
        insteadOf = git://
```

or execute:

```
git config --global url."https://".insteadOf git://
```

Set proxies:

```
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the
following lines in your bash command-line environment:

```
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```

MADS examples
=============

In Julia REPL, do the following commands:

`import Mads`

To explore getting-started instructions, execute:

`Mads.help()`

There are various examples located in the `examples` directory of the `Mads` repository.

For example, execute

`include(Mads.madsdir * "/../examples/contamination/contamination.jl")`

to perform various analyses related to contaminant transport, or execute

`include(Mads.madsdir * "/../examples/bigdt/bigdt.jl")`

to perform BIG-DT analysis