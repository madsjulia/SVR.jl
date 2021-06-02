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
	for i = 1:ninstances
		for j = 1:nfeatures
			nodes[j, i] = svm_node(Cint(j), Float64(x[j, i]))
		end
		nodes[nfeatures + 1, i] = svm_node(Cint(-1), NaN)
		nodeptrs[i] = pointer(nodes, (i - 1) * (nfeatures + 1) +1)
	end
	return nodes, nodeptrs
end