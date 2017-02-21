module SVR

import JLD

verbosity = false

immutable svm_node
	index::Cint
	value::Cdouble
end

immutable svm_problem
	l::Cint
	y::Ptr{Cdouble}
	x::Ptr{Ptr{svm_node}}
end

immutable svm_parameter
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

immutable svm_modelall
	plibsvmmodel::Any
	param::svm_parameter
	problem::svm_problem
end

immutable svm_model
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

# get library
let libsvm = C_NULL
	global get_lib
	function get_lib()
		if libsvm == C_NULL
			libpath = joinpath(dirname(@__FILE__), "..", "deps", "libsvm-3.22")
			libfile = is_windows() ? joinpath(libpath, "libsvm$(Sys.WORD_SIZE).dll") : joinpath(libpath, "libsvm.so.2")
			libsvm = Libdl.dlopen(libfile)
			ccall(Libdl.dlsym(libsvm, :svm_set_print_string_function), Void, (Ptr{Void},), cfunction(liboutput, Void, (Ptr{UInt8},)))
		end
		libsvm
	end
end

# catch lib output
function liboutput(str::Ptr{UInt8})
	if verbosity
		print(unsafe_string(str))
	end
	nothing
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
@cachedsym svm_free_model_content

function mapparam(;
	svm_type::Cint=EPSILON_SVR,
	kernel_type::Cint=RBF,
	degree::Integer=3,
	gamma::Float64=1.0,
	coef0::Float64=0.0,
	C::Float64=1.0,
	nu::Float64=0.5,
	p::Float64=0.1,
	cache_size::Cdouble=100.0,
	eps::Cdouble=0.001,
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

function mapnodes(x::Array)
	nfeatures = size(x, 1)
	ninstances = size(x, 2)
	nodeptrs = Array(Ptr{svm_node}, ninstances)
	nodes = Array(svm_node, nfeatures + 1, ninstances)
	for i=1:ninstances
		for j=1:nfeatures
			nodes[j, i] = svm_node(Cint(j), Float64(x[j, i]))
		end
		nodes[nfeatures + 1, i] = svm_node(Cint(-1), NaN)
		nodeptrs[i] = pointer(nodes, (i - 1) * (nfeatures + 1) +1)
	end
	(nodes, nodeptrs)
end

"Train based on a libSVM model"
function train(y::Vector, x::Array; svm_type::Int32=EPSILON_SVR, kernel_type::Int32=RBF, degree::Integer=3, gamma::Float64=1/size(x, 1), coef0::Float64=0.0, C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1, cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true, probability::Bool=false, verbose::Bool=false)
	param = mapparam(svm_type=svm_type, kernel_type=kernel_type, gamma=gamma, coef0=coef0, C=C, nu=nu, p=p, cache_size=cache_size, eps=eps, shrinking=shrinking, probability=probability)
	(nodes, nodeptrs) = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	plibsvmmodel = ccall(svm_train(), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	return svm_modelall(plibsvmmodel, param, prob)
end

"Predict based on a libSVM model"
function predict(pmodel::svm_modelall, x::Array)
	(nodes, nodeptrs) = mapnodes(x)
	nx = size(x, 2)
	y = Array(Float64, nx)
	for i = 1:nx
		p = ccall(svm_predict(), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel.plibsvmmodel, nodeptrs[i])
		y[i] = p
	end
	return y
end

"Save a libSVM model"
function savemodel(pmodel::svm_modelall, filename::String)
	ccall(svm_save_model(), Cint, (Ptr{UInt8}, Ptr{svm_model}), filename, pmodel.plibsvmmodel)
	nothing
end

"Free a libSVM model"
function freemodel(pmodel::svm_modelall)
	ccall(svm_free_model_content(), Void, (Ptr{Void},), pmodel.plibsvmmodel)
	nothing
end

"Read a libSVM file"
function readlibsvmfile(file::String)
	d = readdlm(file)
	(o, p) = size(d)
	x = Array(Float64, o, p - 1)
	y = []
	try
		y = Float64.(d[:,1])
	catch
		y = d[:,1]
	end
	for i = 2:p
		x[:,i-1] = map(j->(parse(split(d[j,i], ':')[2])), collect(1:o))
	end
	return x, y
end

end
