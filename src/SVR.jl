__precompile__()

module SVR

import Base
import Libdl
import DelimitedFiles
import DocumentFunction
import Statistics

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
	tolerance::Cdouble
	C::Cdouble
	nr_weight::Cint
	weight_label::Ptr{Cint}
	weight::Ptr{Cdouble}
	nu::Cdouble
	epsilon::Cdouble
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
			libfile = joinpath(libpath, "libsvm.so.2")
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
            "epsilon"=>"epsilon for EPSILON_SVR model; defines an epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value [default=`1e-9`]",
            "cache_size"=>"size of the kernel cache [default=`100.0`]",
            "tolerance"=>"tolerance; stopping criteria[default=`0.001`]",
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
	gamma::Cdouble=0.1,
	coef0::Cdouble=0.0,
	C::Cdouble=1.0,
	nu::Cdouble=0.1,
	epsilon::Cdouble=1e-9, # epsilon for EPSILON_SVR
	cache_size::Cdouble=100.0,
	tolerance::Cdouble=0.001, # solution tolerance; stopping criteria
	shrinking::Bool=true,
	probability::Bool=false,
	nr_weight::Integer = 0,
	weight_label = Ptr{Cint}(0x0000000000000000),
	weight = Ptr{Cdouble}(0x0000000000000000))

	param = svm_parameter(
	    Cint(svm_type),
		Cint(kernel_type),
		Cint(degree),
		Cdouble(gamma),
		Cdouble(coef0),
		Cdouble(cache_size),
		Cdouble(tolerance),
		Cdouble(C),
		Cint(nr_weight),
		weight_label,
		weight,
		Cdouble(nu),
		Cdouble(epsilon),
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
            "epsilon"=>"epsilon in the EPSILON_SVR model; defines an epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value [default=`1e-9`]",
            "cache_size"=>"size of the kernel cache [default=`100.0`]",
            "tol"=>"tolerance of termination criterion [default=`0.001`]",
            "shrinking"=>"apply shrinking heuristic [default=`true`]",
            "probability"=>"train to estimate probabilities [default=`false`]",
            "verbose"=>"verbose output [default=`false`]")))

Returns:

- SVM model
"""
function train(y::AbstractVector{Float64}, x::AbstractArray{Float64}; svm_type::Int32=EPSILON_SVR, kernel_type::Int32=RBF, degree::Integer=3, gamma::Float64=1/maximum(x), coef0::Float64=0.0, C::Float64=1.0, nu::Float64=0.1, epsilon::Float64=1e-4, cache_size::Float64=100.0, tol::Float64=0.001, shrinking::Bool=true, probability::Bool=false, verbose::Bool=false)
	@assert length(y) == size(x, 2)
	if maximum(y) > 1 || minimum(y) < -1
		@warn("Dependent variables should be normalized!")
	end
	param = mapparam(; svm_type=svm_type, kernel_type=kernel_type, degree=degree, gamma=gamma, coef0=coef0, C=C, nu=nu, epsilon=epsilon, cache_size=cache_size, tolerance=tol, shrinking=shrinking, probability=probability)
	(nodes, nodeptrs) = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	plibsvmmodel = ccall(svm_train(), Ptr{svm_model}, (Ptr{svm_problem}, Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	return svmmodel(plibsvmmodel, param, prob, nodes)
end
function train(y::AbstractVector, x::AbstractArray; kw...)
	SVR.train(Float64.(y), Float64.(x); kw...)
end
function train(y::AbstractArray, x::AbstractArray; kw...)
	@assert size(y, 1) == size(x, 2)
	nm = size(y, 2)
	m = Vector{svmmodel}(undef, nm)
	for i = 1:nm
		m[i] = SVR.train(vec(y[:,i]), x; kw...)
	end
	return m
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
function predict(pmodel::svmmodel, x::AbstractArray{Float64})
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

function fit(y::AbstractVector{Float64}, x::AbstractArray{Float64}; ymin=minimum(y), ymax=maximum(y), kw...)
	a = (y .- ymin) ./ (ymax - ymin)
	pmodel = SVR.train(a, x; kw...)
	y_pr = SVR.predict(pmodel, x)
	SVR.freemodel(pmodel)
	if any(isnan.(y_pr))
		@warn("SVR output contains NaN's")
	end
	return (y_pr * (ymax - ymin)) .+ ymin
end
function fit(y::AbstractVector{T}, x::AbstractArray{T}; kw...) where {T}
	T.(SVR.fit(Float64.(y), Float64.(x); kw...))
end
function fit(y::AbstractArray{T}, x::AbstractArray{T}; kw...) where {T}
	@assert size(y, 1) == size(x, 2)
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i] = SVR.fit(vec(y[:,i]), x; kw...)
	end
	return yp
end

function fit_test(y::AbstractVector{Float64}, x::AbstractArray{Float64}; ratio::Number=0.1, repeats::Number=1, pm=nothing, keepcases=nothing, scale::Bool=false, ymin::Number=minimum(y), ymax::Number=maximum(y), quiet::Bool=false, veryquiet::Bool=true, total::Bool=false, rmse::Bool=true, callback::Function=(y::AbstractVector, y_pr::AbstractVector, pm::AbstractVector)->nothing, kw...)
	if keepcases !== nothing
		@assert length(keepcases) == size(x, 2)
	end
	@assert length(y) == size(x, 2)
	ymin = scale ? 0 : ymin
	a = (y .- ymin) ./ (ymax - ymin)
	m = Vector{Float64}(undef, repeats)
	y_pra = Vector{Float64}(undef, 0)
	ya = Vector{Float64}(undef, 0)
	pma = Vector{Bool}(undef, 0)
	local y_pr
	for r in 1:repeats
		if repeats > 1 || pm === nothing
			pm = get_prediction_mask(length(y), ratio; keepcases=keepcases)
		else
			@assert length(pm) == size(x, 2)
			@assert eltype(pm) <: Bool
		end
		ic = sum(.!pm)
		if !quiet && repeats == 1 && length(y) > ic
			@info("Training on $(ic) out of $(length(y)) (prediction ratio $ratio)")
		end
		pmodel = SVR.train(a[.!pm], x[:,.!pm]; kw...)
		y_pr = SVR.predict(pmodel, x)
		SVR.freemodel(pmodel)
		if any(isnan.(y_pr))
			@warn("SVR output contains NaN's")
		end
		if rmse
			m[r] = total ? SVR.rmse(y_pr, a) : SVR.rmse(y_pr[pm], a[pm])
		else
			m[r] = total ? SVR.r2(y_pr, a) : SVR.r2(y_pr[pm], a[pm])
		end
		if !veryquiet && repeats > 1
			println("Repeat $r: $(m[r])")
		end
		y_pra = vcat(y_pra, y_pr)
		ya = vcat(ya, y)
		pma = vcat(pma, pm)
	end
	y_pra = y_pra * (ymax - ymin) .+ ymin
	callback(ya, y_pra, pma)
	y_pr = y_pr * (ymax - ymin) .+ ymin
	return y_pr, pm, Statistics.mean(m)
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}; ratio::Number=0.1, kw...) where {T}
	y_pr, pm, rmse = SVR.fit_test(Float64.(y), Float64.(x); ratio=ratio, kw...)
	return T.(y_pr), pm, rmse
end
function fit_test(y::AbstractArray{T}, x::AbstractArray{T}; ratio::Number=0.1, pm=nothing, keepcases=nothing, kw...) where {T}
	@assert size(y, 1) == size(x, 2)
	if keepcases !== nothing
		@assert length(keepcases) == size(x, 2)
	end
	if pm === nothing
		pm = get_prediction_mask(size(y, 1), ratio; keepcases=keepcases)
	end
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i], _, rmse = SVR.fit_test(vec(y[:,i]), x; ratio=ratio, pm=pm, kw...)
	end
	return yp, pm, rmse
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr=:gamma, rmse::Bool=true, check::Function=(v::AbstractVector)->nothing, kw...) where {T}
	@assert length(vattr) > 0
	@info("Grid search on $attr with prediction ratio $ratio")
	ma = Vector{T}(undef, length(vattr))
	for (i, g) in enumerate(vattr)
		k = Dict(attr=>g)
		y_pr, pm, ma[i] = SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., quiet=true)
		@info("$attr=>$g: $(ma[i])")
	end
	c = check(ma)
	if c === nothing
		m, i = rmse ? findmin(ma) : findmax(ma)
	else
		i = c
		m = ma[i]
	end
	k = Dict(attr=>vattr[i])
 	return m, vattr[i], SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr1::Union{AbstractVector,AbstractRange}, vattr2::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr1=:gamma, attr2=:epsilon, rmse::Bool=true, kw...) where {T}
	@assert length(vattr1) > 0
	@assert length(vattr2) > 0
	@info("Grid search on $attr1/$attr2 with prediction ratio $ratio")
	ma = Matrix{T}(undef, length(vattr1), length(vattr2))
	for (i, a1) in enumerate(vattr1)
		for (j, a2) in enumerate(vattr2)
			k = Dict(attr1=>a1, attr2=>a2)
			y_pr, pm, ma[i, j] = SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k...)
			@info("$attr1=>$a1 $attr2=>$a2: $(ma[i,j])")
		end
	end
	m, i = rmse ? findmin(ma) : findmax(ma)
	k = Dict(attr1=>vattr1[i.I[1]], attr2=>vattr2[i.I[2]])
	return m, vattr1[i.I[1]], vattr2[i.I[2]], SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr1::Union{AbstractVector,AbstractRange}, vattr2::Union{AbstractVector,AbstractRange}, vattr3::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr1=:gamma, attr2=:epsilon, attr3=:C, rmse::Bool=true, kw...) where {T}
	@assert length(vattr1) > 0
	@assert length(vattr2) > 0
	@assert length(vattr3) > 0
	@info("Grid search on $attr1/$attr2/$attr3 with prediction ratio $ratio")
	ma = Array{T}(undef, length(vattr1), length(vattr2), length(vattr3))
	for (i, a1) in enumerate(vattr1)
		for (j, a2) in enumerate(vattr2)
			for (k, a3) in enumerate(vattr3)
				kk = Dict(attr1=>a1, attr2=>a2, attr3=>a3)
				y_pr, pm, ma[i, j, k] = SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., kk...)
				@info("$attr1=>$a1 $attr2=>$a2 $attr3=>$a3: $(ma[i,j,k])")
			end
		end
	end
	m, i = rmse ? findmin(ma) : findmax(ma)
	k = Dict(attr1=>vattr1[i.I[1]], attr2=>vattr2[i.I[2]], attr3=>vattr3[i.I[3]])
	return m, vattr1[i.I[1]], vattr2[i.I[2]], vattr3[i.I[3]], SVR.fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end

"""
Get prediction mask

$(DocumentFunction.documentfunction(get_prediction_mask;
argtext=Dict("ns"=>"number of samples",
            "ratio"=>"prediction ratio")))

Return:

- prediction mask
"""
function get_prediction_mask(ns::Number, ratio::Number; keepcases=nothing)
	pm = trues(ns)
	ic = convert(Int64, ceil(ns * (1. - ratio)))
	if keepcases !== nothing
		@assert length(keepcases) == length(pm)
		kn = sum(keepcases)
		if ic > kn && ns > kn
			pm[keepcases] .= false
			ic -= kn
			ns -= kn
		else
			kn > 0 && @warn("Number of cases to keep is larger ($(kn)) than allowed samples to keep ($(ic))!")
		end
	end
	ir = sortperm(rand(ns))[1:ic]
	if keepcases !== nothing && ic > kn
		m = trues(ns)
		m[ir] .= false
		pm[.!keepcases] .= m
	else
		pm[ir] .= false
	end
	return pm
end

"""
Predict based on a libSVM model

$(DocumentFunction.documentfunction(apredict;
argtext=Dict("y"=>"vector of dependent variables",
            "x"=>"array of independent variables")))

Return:

- predicted dependent variables
"""
function apredict(y::AbstractVector{Float64}, x::AbstractArray{Float64}; kw...)
	svmmodel = train(y, x; kw...)
	p = predict(svmmodel, x)
	freemodel(svmmodel)
	if any(isnan.(p))
		@warn("SVR output contains NaN's")
	end
	return p
end
function apredict(y::AbstractVector{T}, x::AbstractArray{T}; kw...) where {T}
	T.(SVR.apredict(Float64.(y), Float64.(x); kw...))
end
function apredict(y::AbstractArray{T}, x::AbstractArray{T}; kw...) where {T}
	@assert size(y, 1) == size(x, 2)
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i] = SVR.apredict(vec(y[:,i]), x; kw...)
	end
	return yp
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
	d = DelimitedFiles.readdlm(file)
	(o, p) = size(d)
	x = Array{Float64}(undef, o, p - 1)
	y = Vector{Float64}(undef, o)
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
function r2(x::AbstractVector, y::AbstractVector)
	ix = .!isnan.(x)
	iy = .!isnan.(y)
	ii = ix .& iy
	mx = x[ii] .- Statistics.mean(x[ii])
	my = y[ii] .- Statistics.mean(y[ii])
	r2 = (sum(mx .* my) / sqrt(sum((mx .^ 2) * sum(my .^ 2))))^2
	# sres = sum((x[ii] .- y[ii]) .^ 2)
	# stot = sum((y[ii] .- Statistics.mean(y[ii])) .^ 2)
	# r2 = 1. - (sres / stot)
	r2 = isnan(r2) ? 0 : r2
 	return r2
end

function rmse(t::AbstractVector, o::AbstractVector)
	return sqrt( sum( (t .- o) .^ 2.) ./ length(t) )
end

end