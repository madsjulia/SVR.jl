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