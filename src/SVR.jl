__precompile__()

module SVR

using Base
using Libdl
using DelimitedFiles
import DocumentFunction

verbosity = false

struct svm_node
	index::Cint
	value::Cdouble
end

mutable struct svm_problem
	l::Cint
	y::Ptr{Cdouble}
	x::Ptr{Ptr{svm_node}}
end

mutable struct svm_parameter
	svm_type::Cint
	kernel_type::Cint
	degree::Cint
	gamma::Cdouble
	coef0::Cdouble
	cache_size::Cdouble
	eps::Cdouble
	C::Cdouble
	nr_weight::Cint
	weight_label::Ptr{Cint}
	weight::Ptr{Cdouble}
	nu::Cdouble
	p::Cdouble
	shrinking::Cint
	probability::Cint
end

mutable struct svm_model
	param::svm_parameter
	nr_class::Cint
	l::Cint
	SV::Ptr{Ptr{svm_node}}
	sv_coef::Ptr{Ptr{Cdouble}}
	rho::Ptr{Cdouble}
	probA::Ptr{Cdouble}
	probB::Ptr{Cdouble}
	sv_indices::Ptr{Cint}
	label::Ptr{Cint}
	nSV::Ptr{Cint}
	free_sv::Cint
end

mutable struct svmmodel
	plibsvmmodel::Ptr{svm_model}
	param::svm_parameter
	problem::svm_problem
	nodes::Array{svm_node}
end

const C_SVC = Cint(0)
const NU_SVC = Cint(1)
const ONE_CLASS = Cint(2)
const EPSILON_SVR = Cint(3)
const NU_SVR = Cint(4)

const LINEAR = Cint(0)
const POLY = Cint(1)
const RBF = Cint(2)
const SIGMOID = Cint(3)
const PRECOMPUTED = Cint(4)

# const svmlib = abspath(joinpath(Pkg.dir("SVR"), "deps", "libsvm.so.2"))
# const svmlib = abspath(joinpath(Pkg.dir("SVR"), "deps", "libdensesvm.so.2"))


"""
catch lib output

$(DocumentFunction.documentfunction(liboutput; argtext=Dict("str"=>"string")))
"""
function liboutput(str::Ptr{UInt8})
	if verbosity
		print(unsafe_string(str))
	end
	nothing
end

# get library
let libsvm = C_NULL
	global get_lib
	function get_lib()
		if libsvm == C_NULL
			libpath = joinpath(dirname(@__FILE__), "..", "deps", "libsvm-3.22")
			libfile = Sys.iswindows() ? joinpath(libpath, "libsvm$(Sys.WORD_SIZE).dll") : joinpath(libpath, "libsvm.so.2")
			libsvm = Libdl.dlopen(libfile)
			ccall(Libdl.dlsym(libsvm, :svm_set_print_string_function), Nothing, (Ptr{Nothing},), @cfunction(liboutput, Nothing, (Ptr{UInt8},)))
		end
		libsvm
	end
end

# make lib function calls
macro cachedsym(symname::Symbol)
	cached = gensym()
	quote
		let $cached = C_NULL
			global ($symname)
			($symname)() = ($cached) == C_NULL ?
				($cached = Libdl.dlsym(get_lib(), $(string(symname)))) :
					$cached
		end
	end
end

@cachedsym svm_train
@cachedsym svm_predict
@cachedsym svm_save_model
@cachedsym svm_load_model
@cachedsym svm_free_model_content

"""
$(DocumentFunction.documentfunction(mapparam;
keytext=Dict("svm_type"=>"SVM type [default=`EPSILON_SVR`]",
            "kernel_type"=>"kernel type [default=`RBF`]",
            "degree"=>"degree of the polynomial kernel [default=`3`]",
            "gamma"=>"coefficient for RBF, POLY and SIGMOND kernel types [default=`1.0`]",
            "coef0"=>"independent term in kernel function; important only in POLY and  SIGMOND kernel types [default=`0.0`]",
            "C"=>"cost; penalty parameter of the error term [default=`1.0`]",
            "nu"=>"upper bound on the fraction of training errors / lower bound of the fraction of support vectors; acceptable range (0, 1]; applied if NU_SVR model [default=`0.5`]",
            "p"=>"epsilon for EPSILON_SVR [default=`0.1`]",
            "cache_size"=>"size of the kernel cache [default=`100.0`]",
            "eps"=>"epsilon in the EPSILON_SVR model; defines an epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value [default=`0.001`]",
            "shrinking"=>"apply shrinking heuristic [default=`true`]",
            "probability"=>"train to estimate probabilities [default=`false`]",
            "nr_weight"=>"[default=`0`]",
            "weight_label"=>"[default=`Ptr{Cint}(0x0000000000000000)`]",
            "weight"=>"[default=`Ptr{Cdouble}(0x0000000000000000)`]")))

Returns:

- parameter
"""
function mapparam(;
	svm_type::Cint=EPSILON_SVR,
	kernel_type::Cint=RBF,
	degree::Integer=3,
	gamma::Float64=1.0,
	coef0::Float64=0.0,
	C::Float64=1.0,
	nu::Float64=0.5,
	p::Float64=0.1, # epsilon for EPSILON_SVR
	cache_size::Cdouble=100.0,
	eps::Cdouble=0.001, # solution tolerance
	shrinking::Bool=true,
	probability::Bool=false,
	nr_weight::Integer = 0,
	weight_label = Ptr{Cint}(0x0000000000000000),
	weight = Ptr{Cdouble}(0x0000000000000000))

	param = svm_parameter(Cint(svm_type),
		Cint(kernel_type),
		Cint(degree),
		Cdouble(gamma),
		Cdouble(coef0),
		Cdouble(cache_size),
		Cdouble(eps),
		Cdouble(C),
		Cint(nr_weight),
		weight_label,
		weight,
		Cdouble(nu),
		Cdouble(p),
		Cint(shrinking),
		Cint(probability))
	return param
end

"""
$(DocumentFunction.documentfunction(mapnodes;
argtext=Dict("x"=>"")))
"""
function mapnodes(x::AbstractArray)
	nfeatures = size(x, 1)
	ninstances = size(x, 2)
	nodeptrs = Array{Ptr{svm_node}}(undef, ninstances)
	nodes = Array{svm_node, 2}(undef, nfeatures + 1, ninstances)
	for i=1:ninstances
		for j=1:nfeatures
			nodes[j, i] = svm_node(Cint(j), Float64(x[j, i]))
		end
		nodes[nfeatures + 1, i] = svm_node(Cint(-1), NaN)
		nodeptrs[i] = pointer(nodes, (i - 1) * (nfeatures + 1) +1)
	end
	(nodes, nodeptrs)
end

"""
Train based on a libSVM model

$(DocumentFunction.documentfunction(train;
argtext=Dict("y"=>"vector of dependent variables",
            "x"=>"array of independent variables"),
keytext=Dict("svm_type"=>"SVM type [default=`EPSILON_SVR`]",
            "kernel_type"=>"kernel type [default=`RBF`]",
            "degree"=>"degree of the polynomial kernel [default=`3`]",
            "gamma"=>"coefficient for RBF, POLY and SIGMOND kernel types [default=`1/size(x, 1)`]",
            "coef0"=>"independent term in kernel function; important only in POLY and  SIGMOND kernel types [default=`0.0`]",
            "C"=>"cost; penalty parameter of the error term [default=`1.0`]",
            "nu"=>"upper bound on the fraction of training errors / lower bound of the fraction of support vectors; acceptable range (0, 1]; applied if NU_SVR model [default=`0.5`]",
            "eps"=>"epsilon in the EPSILON_SVR model; defines an epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value [default=`0.1`]",
            "cache_size"=>"size of the kernel cache [default=`100.0`]",
            "tol"=>"tolerance of termination criterion [default=`0.001`]",
            "shrinking"=>"apply shrinking heuristic [default=`true`]",
            "probability"=>"train to estimate probabilities [default=`false`]",
            "verbose"=>"verbose output [default=`false`]")))

Returns:

- SVM model
"""
function train(y::AbstractVector, x::AbstractArray; svm_type::Int32=EPSILON_SVR, kernel_type::Int32=RBF, degree::Integer=3, gamma::Float64=1/size(x, 1), coef0::Float64=0.0, C::Float64=1.0, nu::Float64=0.5, eps::Float64=0.1, cache_size::Float64=100.0, tol::Float64=0.001, shrinking::Bool=true, probability::Bool=false, verbose::Bool=false)
	@assert length(y) == size(x, 2)
	param = mapparam(svm_type=svm_type, kernel_type=kernel_type, degree=degree, gamma=gamma, coef0=coef0, C=C, nu=nu, p=eps, cache_size=cache_size, eps=tol, shrinking=shrinking, probability=probability)
	(nodes, nodeptrs) = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	plibsvmmodel = ccall(svm_train(), Ptr{svm_model}, (Ptr{svm_problem}, Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	return svmmodel(plibsvmmodel, param, prob, nodes)
end
export train

"""
Predict based on a libSVM model

$(DocumentFunction.documentfunction(predict;
argtext=Dict("pmodel"=>"the model that prediction is based on",
            "x"=>"array of independent variables")))

Return:

- predicted dependent variables
"""
function predict(pmodel::svmmodel, x::AbstractArray)
	nx = size(x, 2)
	y = Array{Float64}(undef, nx)
	if pmodel.plibsvmmodel != Ptr{SVR.svm_model}(C_NULL)
		(nodes, nodeptrs) = mapnodes(x)
		for i = 1:nx
			y[i] = ccall(svm_predict(), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel.plibsvmmodel, nodeptrs[i])
		end
	else
		warn("SVR model is not defined")
		y .= NaN
	end
	return y
end
export predict

"""
Predict based on a libSVM model

$(DocumentFunction.documentfunction(apredict;
argtext=Dict("y"=>"vector of dependent variables",
            "x"=>"array of independent variables")))

Return:

- predicted dependent variables
"""
function apredict(y::AbstractVector, x::AbstractArray; kw...)
	svmmodel = train(y, x; kw...)
	freemodel(svmmodel)
	p = predict(svmmodel, x)
end

"""
Load a libSVM model

$(DocumentFunction.documentfunction(loadmodel;
argtext=Dict("filename"=>"input file name")))

Returns:

- SVM model
"""
function loadmodel(filename::String)
	param = mapparam()
	x = Array{Float64}(undef, 0)
	y = Array{Float64}(undef, 0)
	(nodes, nodeptrs) = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	plibsvmmodel = ccall(svm_load_model(), Ptr{svm_model}, (Ptr{UInt8},), filename)
	return svmmodel(plibsvmmodel, param, prob, nodes)
end
export loadmodel

"""
Save a libSVM model

$(DocumentFunction.documentfunction(savemodel;
argtext=Dict("pmodel"=>"svm model",
            "filename"=>"output file name")))

Dumps:

- file with saved model
"""
function savemodel(pmodel::svmmodel, filename::String)
	if pmodel.plibsvmmodel != Ptr{SVR.svm_model}(C_NULL)
		ccall(svm_save_model(), Cint, (Ptr{UInt8}, Ptr{svm_model}), filename, pmodel.plibsvmmodel)
	end
	nothing
end

"""
Free a libSVM model

$(DocumentFunction.documentfunction(freemodel;
argtext=Dict("pmodel"=>"svm model")))
"""
function freemodel(pmodel::svmmodel)
	if pmodel.plibsvmmodel != Ptr{SVR.svm_model}(C_NULL)
		ccall(svm_free_model_content(), Nothing, (Ptr{Nothing},), pmodel.plibsvmmodel)
		pmodel.plibsvmmodel = Ptr{SVR.svm_model}(C_NULL)
	end
	nothing
end
export savemodel

"""
Read a libSVM file

$(DocumentFunction.documentfunction(readlibsvmfile;
argtext=Dict("file"=>"file name")))

Returns:

- array of independent variables
- vector of dependent variables
"""
function readlibsvmfile(file::String)
	d = readdlm(file)
	(o, p) = size(d)
	x = Array{Float64}(undef, o, p - 1)
	y = []
	try
		y = Float64.(d[:,1])
	catch
		y = d[:,1]
	end
	for i = 2:p
		x[:,i-1] = map(j->(parse(Float64, split(d[j,i], ':')[2])), collect(1:o))
	end
	return x, y
end

"""
Compute the coefficient of determination (r2)

$(DocumentFunction.documentfunction(r2;
argtext=Dict("x"=>"observed data", "y"=>"predicted data")))

Returns:

- coefficient of determination (r2)
"""
function r2(x, y)
	stot = sum((x .- mean(x)).^2)
	sres = sum((x - y).^2)
	return(1. - (sres / stot))
end

end
