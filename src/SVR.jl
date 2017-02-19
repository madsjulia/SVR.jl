module SVR

import JLD

include("extras.jl")

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

const svmlib = abspath(joinpath(Pkg.dir("SVR"), "deps", "libsvm.so.2"))
const densesvmlib = abspath(joinpath(Pkg.dir("SVR"), "deps", "libdensesvm.so.2"))

function mapnodes(instances)
	nfeatures = size(instances, 1)
	ninstances = size(instances, 2)
	nodeptrs = Array(Ptr{svm_node}, ninstances)
	nodes = Array(svm_node, nfeatures + 1, ninstances)
	for i=1:ninstances
		k = 1
		for j=1:nfeatures
			nodes[k, i] = svm_node(Cint(j), Float64(instances[j, i]))
			k += 1
		end
		nodes[k, i] = svm_node(Cint(-1), 0.0)
		nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
	end
	(nodes, nodeptrs)
end

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

function train(y::Vector, x::Array; dense::Bool=false, svm_type::Int32=EPSILON_SVR, kernel_type::Int32=RBF, degree::Integer=3, gamma::Float64=1/length(y), coef0::Float64=0.0, C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1, cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true, probability::Bool=false, verbose::Bool=false)
	param = mapparam(svm_type=svm_type, kernel_type=kernel_type, gamma=gamma, coef0=coef0, C=C, nu=nu, p=p, cache_size=cache_size, eps=eps, shrinking=shrinking, probability=probability)
	(nodes, nodeptrs) = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	if !dense
		pmodel = ccall((:svm_train, svmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	else
		pmodel = ccall((:svm_train, densesvmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	end
	return pmodel
end

function predict(model::svm_model, x::Array; dense::Bool=false)
	(nodes, nodeptrs) = mapnodes(x)
	nx = size(instances, 2)
	predicted = Array(Float64, nx)
	for i = 1:nx
		if !dense
			pred = ccall((:svm_predict, svmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pointer(model), nodeptrs[i])
		else
			pred = ccall((:svm_predict, densesvmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pointer(model), nodeptrs[i])
		end
		predicted[i] = pred
	end
	return predicted
end

end