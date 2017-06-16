SVR
================

[![SVR](http://pkg.julialang.org/badges/SVR_0.5.svg)](http://pkg.julialang.org/?pkg=SVR&ver=0.5)
[![Build Status](https://travis-ci.org/madsjulia/SVR.jl.svg?branch=master)](https://travis-ci.org/madsjulia/SVR.jl)
[![Coverage Status](https://coveralls.io/repos/madsjulia/SVR.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/SVR.jl?branch=master)

Support Vector Regression (SVR) analysis in [Julia](http://julialang.org) utilizing the [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library.

SVR is a module of [MADS](http://madsjulia.github.io/Mads.jl) (Model Analysis & Decision Support).

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

# predict based on the libSVM model
y_pr = SVR.predict(pmodel, x');

# save the libSVM model
SVR.savemodel(pmodel, "mg.model")

# free the memory allocation of the libSVM model
SVR.freemodel(pmodel)
```

MADS
====

[MADS](http://madsjulia.github.io/Mads.jl) (Model Analysis & Decision Support) is an integrated open-source high-performance computational (HPC) framework in [Julia](http://julialang.org).
MADS can execute a wide range of data- and model-based analyses:

* Sensitivity Analysis
* Parameter Estimation
* Model Inversion and Calibration
* Uncertainty Quantification
* Model Selection and Model Averaging
* Model Reduction and Surrogate Modeling
* Machine Learning and Blind Source Separation
* Decision Analysis and Support

MADS has been tested to perform HPC simulations on a wide-range multi-processor clusters and parallel environments (Moab, Slurm, etc.).
MADS utilizes adaptive rules and techniques which allows the analyses to be performed with a minimum user input.
The code provides a series of alternative algorithms to execute each type of data- and model-based analyses.

Documentation
=============

All the available MADS modules and functions are described at [madsjulia.github.io](http://madsjulia.github.io/Mads.jl)

Installation
============

After starting Julia, execute:

```julia
Pkg.add("Mads")
```

Installation of MADS behind a firewall
------------------------------

Julia uses git for package management. Add in the `.gitconfig` file in your home directory:

```git
[url "https://"]
        insteadOf = git://
```

or execute:

```bash
git config --global url."https://".insteadOf git://
```

Set proxies:

```bash
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the
following lines in your bash command-line environment:

```bash
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```

MADS examples
=============

In Julia REPL, do the following commands:

```julia
import Mads
```

To explore getting-started instructions, execute:

```julia
Mads.help()
```

There are various examples located in the `examples` directory of the `Mads` repository.

For example, execute

```julia
include(Mads.madsdir * "/../examples/contamination/contamination.jl")
```

to perform various example analyses related to groundwater contaminant transport, or execute

```julia
include(Mads.madsdir * "/../examples/bigdt/bigdt.jl")
```

to perform Bayesian Information Gap Decision Theory (BIG-DT) analysis.

Developers
==========

* [Velimir (monty) Vesselinov](http://www.lanl.gov/orgs/ees/staff/monty) [(publications)](http://scholar.google.com/citations?user=sIFHVvwAAAAJ)
* [Daniel O'Malley](http://www.lanl.gov/expertise/profiles/view/daniel-o'malley) [(publications)](http://scholar.google.com/citations?user=rPzCVjEAAAAJ)
* [see also](https://github.com/madsjulia/SVR.jl/graphs/contributors)

Publications, Presentations, Projects
=====================================

* [mads.lanl.gov/](http://mads.lanl.gov/)
* [ees.lanl.gov/monty](http://ees.lanl.gov/monty)