const svrdir = splitdir(splitdir(pathof(SVR))[1])[1]

"Test SVR"
function test()
	include(joinpath(svrdir, "test", "runtests.jl"))
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
            "coef0"=>"independent term in kernel function; important only in POLY and SIGMOND kernel types [default=`0.0`]",
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
	param = mapparam(; svm_type=svm_type, kernel_type=kernel_type, degree=degree, gamma=gamma, coef0=coef0, C=C, nu=nu, epsilon=epsilon, cache_size=cache_size, tolerance=tol, shrinking=shrinking, probability=probability)
	nodes, nodeptrs = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	if !verbose
		local plibsvmmodel
		@Suppressor.suppress (plibsvmmodel = ccall((:svm_train, libsvm_jll.libsvm), Ptr{svm_model}, (Ptr{svm_problem}, Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param)))
	else
		plibsvmmodel = ccall((:svm_train, libsvm_jll.libsvm), Ptr{svm_model}, (Ptr{svm_problem}, Ptr{svm_parameter}), pointer_from_objref(prob), pointer_from_objref(param))
	end
	return svmmodel(plibsvmmodel, param, prob, nodes)
end
function train(y::AbstractVector, x::AbstractArray; kw...)
	train(Float64.(y), Float64.(x); kw...)
end
function train(y::AbstractArray, x::AbstractArray; kw...)
	@assert size(y, 1) == size(x, 2)
	nm = size(y, 2)
	m = Vector{svmmodel}(undef, nm)
	for i = 1:nm
		m[i] = train(vec(y[:,i]), x; kw...)
	end
	return m
end

"""
Predict based on a libSVM model

$(DocumentFunction.documentfunction(predict;
argtext=Dict("pmodel"=>"the model that prediction is based on",
            "x"=>"array of independent variables")))

Return:

- predicted dependent variables
"""
function predict(pmodel::svmmodel, x::AbstractArray{Float64})
	nn = size(x, 1)
	nx = size(x, 2)
	y = Vector{Float64}(undef, nx)
	if pmodel.plibsvmmodel != Ptr{svm_model}(C_NULL)
		nn2, nx2 = size(pmodel.nodes)
		if nn2 - 1 != nn
			@error("SVR model node count $(nn2 - 1) does not match input dimensions $(nn)!")
			y .= NaN
		else
			nodes, nodeptrs = mapnodes(x)
			for i = 1:nx
				y[i] = ccall((:svm_predict, libsvm_jll.libsvm), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel.plibsvmmodel, nodeptrs[i])
			end
		end
	else
		@warn("SVR model is not defined!")
		y .= NaN
	end
	return y
end
function predict(pmodel::svmmodel, x::AbstractArray{T}) where {T}
	T.(predict(pmodel, Float64.(x)))
end

function fit(y::AbstractVector{Float64}, x::AbstractArray{Float64}; scale::Bool=false, ymin=minimum(y), ymax=maximum(y), kw...)
	ymin = scale ? 0 : ymin
	a = (y .- ymin) ./ (ymax - ymin)
	pmodel = train(a, x; kw...)
	y_pr = predict(pmodel, x)
	freemodel(pmodel)
	if any(isnan.(y_pr))
		@warn("SVR output contains NaN's!")
	end
	return (y_pr * (ymax - ymin)) .+ ymin
end
function fit(y::AbstractVector{T}, x::AbstractArray{T}; kw...) where {T <: Number}
	T.(fit(Float64.(y), Float64.(x); kw...))
end
function fit(y::AbstractArray{T}, x::AbstractArray{T}; kw...) where {T <: Number}
	@assert size(y, 1) == size(x, 2)
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i] = fit(vec(y[:,i]), x; kw...)
	end
	return yp
end

function fit_test(y::AbstractVector{Float64}, x::AbstractArray{Float64}; ratio::Number=0.1, repeats::Number=1, pm=nothing, keepcases::Union{BitArray,Nothing}=nothing, scale::Bool=false, ymin::Number=minimum(y), ymax::Number=maximum(y), quiet::Bool=false, veryquiet::Bool=true, total::Bool=false, rmse::Bool=true, callback::Function=(y::AbstractVector, y_pr::AbstractVector, pm::AbstractVector)->nothing, kw...)
	if !isnothing(keepcases)
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
		if repeats > 1 || isnothing(pm)
			pm = get_prediction_mask(length(y), ratio; keepcases=keepcases)
		else
			@assert length(pm) == size(x, 2)
			@assert eltype(pm) <: Bool
		end
		ic = sum(.!pm)
		if !quiet && repeats == 1 && length(y) > ic
			@info("Training on $(ic) out of $(length(y)) (prediction ratio $ratio) ...")
		end
		pmodel = train(a[.!pm], x[:,.!pm]; kw...)
		y_pr = predict(pmodel, x)
		freemodel(pmodel)
		if any(isnan.(y_pr))
			@warn("SVR output contains NaN's!")
		end
		if rmse
			m[r] = total ? rmse(y_pr, a) : rmse(y_pr[pm], a[pm])
		else
			m[r] = total ? r2(y_pr, a) : r2(y_pr[pm], a[pm])
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
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}; ratio::Number=0.1, kw...) where {T <: Number}
	y_pr, pm, rmse = fit_test(Float64.(y), Float64.(x); ratio=ratio, kw...)
	return T.(y_pr), pm, rmse
end
function fit_test(y::AbstractArray{T}, x::AbstractArray{T}; ratio::Number=0.1, pm=nothing, keepcases::Union{BitArray,Nothing}=nothing, kw...) where {T <: Number}
	@assert size(y, 1) == size(x, 2)
	if !isnothing(keepcases)
		@assert length(keepcases) == size(x, 2)
	end
	if isnothing(pm)
		pm = get_prediction_mask(size(y, 1), ratio; keepcases=keepcases)
	end
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i], _, rmse = fit_test(vec(y[:,i]), x; ratio=ratio, pm=pm, kw...)
	end
	return yp, pm, rmse
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr=:gamma, rmse::Bool=true, check::Function=(v::AbstractVector)->nothing, kw...) where {T <: Number}
	@assert length(vattr) > 0
	@info("Grid search on $attr with prediction ratio $ratio ...")
	ma = Vector{T}(undef, length(vattr))
	for (i, g) in enumerate(vattr)
		k = Dict(attr=>g)
		y_pr, pm, ma[i] = fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., quiet=true)
		@info("$attr=>$g: $(ma[i])")
	end
	c = check(ma)
	if isnothing(c)
		m, i = rmse ? findmin(ma) : findmax(ma)
	else
		i = c
		m = ma[i]
	end
	k = Dict(attr=>vattr[i])
 	return m, vattr[i], fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr1::Union{AbstractVector,AbstractRange}, vattr2::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr1=:gamma, attr2=:epsilon, rmse::Bool=true, kw...) where {T <: Number}
	@assert length(vattr1) > 0
	@assert length(vattr2) > 0
	@info("Grid search on $attr1/$attr2 with prediction ratio $ratio ...")
	ma = Matrix{T}(undef, length(vattr1), length(vattr2))
	for (i, a1) in enumerate(vattr1)
		for (j, a2) in enumerate(vattr2)
			k = Dict(attr1=>a1, attr2=>a2)
			y_pr, pm, ma[i, j] = fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k...)
			@info("$attr1=>$a1 $attr2=>$a2: $(ma[i,j])")
		end
	end
	m, i = rmse ? findmin(ma) : findmax(ma)
	k = Dict(attr1=>vattr1[i.I[1]], attr2=>vattr2[i.I[2]])
	return m, vattr1[i.I[1]], vattr2[i.I[2]], fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end
function fit_test(y::AbstractVector{T}, x::AbstractArray{T}, vattr1::Union{AbstractVector,AbstractRange}, vattr2::Union{AbstractVector,AbstractRange}, vattr3::Union{AbstractVector,AbstractRange}; ratio::Number=0.1, attr1=:gamma, attr2=:epsilon, attr3=:C, rmse::Bool=true, kw...) where {T <: Number}
	@assert length(vattr1) > 0
	@assert length(vattr2) > 0
	@assert length(vattr3) > 0
	@info("Grid search on $attr1/$attr2/$attr3 with prediction ratio $ratio ...")
	ma = Array{T}(undef, length(vattr1), length(vattr2), length(vattr3))
	for (i, a1) in enumerate(vattr1)
		for (j, a2) in enumerate(vattr2)
			for (k, a3) in enumerate(vattr3)
				kk = Dict(attr1=>a1, attr2=>a2, attr3=>a3)
				y_pr, pm, ma[i, j, k] = fit_test(y, x; ratio=ratio, rmse=rmse, kw..., kk...)
				@info("$attr1=>$a1 $attr2=>$a2 $attr3=>$a3: $(ma[i,j,k])")
			end
		end
	end
	m, i = rmse ? findmin(ma) : findmax(ma)
	k = Dict(attr1=>vattr1[i.I[1]], attr2=>vattr2[i.I[2]], attr3=>vattr3[i.I[3]])
	return m, vattr1[i.I[1]], vattr2[i.I[2]], vattr3[i.I[3]], fit_test(y, x; ratio=ratio, rmse=rmse, kw..., k..., repeats=1)...
end

"""
Get prediction mask

$(DocumentFunction.documentfunction(get_prediction_mask;
argtext=Dict("ns"=>"number of samples",
            "ratio"=>"prediction ratio")))

Return:

- prediction mask
"""
function get_prediction_mask(ns::Number, ratio::Number; keepcases::Union{AbstractVector,Nothing}=nothing, debug::Bool=false)
	nsi = copy(ns)
	pm = trues(ns)
	ic = convert(Int64, ceil(ns * (1. - ratio)))
	if !isnothing(keepcases)
		@assert length(keepcases) == length(pm)
		kn = sum(keepcases)
		if ic > kn && ns > kn
			pm[keepcases] .= false
			ic -= kn
			nsi -= kn
		else
			kn > ic && @warn("Number of cases requested to keep is larger ($(kn)) than the number of cases allowed samples to keep ($(ic))!")
		end
	end
	if nsi >= ic
		ir = sortperm(rand(nsi))[1:ic]
		if !isnothing(keepcases) && ic > kn
			m = trues(nsi)
			m[ir] .= false
			pm[.!keepcases] .= m
		else
			pm[ir] .= false
		end
	end
	if debug
		@info("Number of cases for training: $(ns - sum(pm))")
		@info("Number of cases for prediction: $(sum(pm))")
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
		@warn("SVR output contains NaN's!")
	end
	return p
end
function apredict(y::AbstractVector{T}, x::AbstractArray{T}; kw...) where {T <: Number}
	T.(apredict(Float64.(y), Float64.(x); kw...))
end
function apredict(y::AbstractArray{T}, x::AbstractArray{T}; kw...) where {T <: Number}
	@assert size(y, 1) == size(x, 2)
	yp = similar(y)
	for i = 1:size(y, 2)
		yp[:,i] = apredict(vec(y[:,i]), x; kw...)
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
function loadmodel(filename::AbstractString)
	param = mapparam()
	nnodes, ssize = Int.(DelimitedFiles.readdlm(splitext(filename)[1] .* ".nodes"))
	x = Vector{Float64}(undef, nnodes - 1)
	y = Vector{Float64}(undef, ssize - 1)
	nodes, nodeptrs = mapnodes(x)
	prob = svm_problem(length(y), pointer(y), pointer(nodeptrs))
	plibsvmmodel = ccall((:svm_load_model, libsvm_jll.libsvm), Ptr{svm_model}, (Ptr{UInt8},), filename)
	return svmmodel(plibsvmmodel, param, prob, nodes)
end

"""
Save a libSVM model

$(DocumentFunction.documentfunction(savemodel;
argtext=Dict("pmodel"=>"svm model",
            "filename"=>"output file name")))

Dumps:

- file with saved model
"""
function savemodel(pmodel::svmmodel, filename::AbstractString)
	if pmodel.plibsvmmodel != Ptr{svm_model}(C_NULL)
		ccall((:svm_save_model, libsvm_jll.libsvm), Cint, (Ptr{UInt8}, Ptr{svm_model}), filename, pmodel.plibsvmmodel)
		nnodes, ssize = size(pmodel.nodes)
		DelimitedFiles.writedlm(splitext(filename)[1] .* ".nodes", (nnodes, ssize))
	end
	return nothing
end

"""
Free a libSVM model

$(DocumentFunction.documentfunction(freemodel;
argtext=Dict("pmodel"=>"svm model")))
"""
function freemodel(pmodel::svmmodel)
	if pmodel.plibsvmmodel != Ptr{svm_model}(C_NULL)
		ccall((:svm_free_model_content, libsvm_jll.libsvm), Nothing, (Ptr{Nothing},), pmodel.plibsvmmodel)
		pmodel.plibsvmmodel = Ptr{svm_model}(C_NULL)
	end
	return nothing
end

"""
Read a libSVM file

$(DocumentFunction.documentfunction(readlibsvmfile;
argtext=Dict("file"=>"file name")))

Returns:

- array of independent variables
- vector of dependent variables
"""
function readlibsvmfile(file::AbstractString)
	d = DelimitedFiles.readdlm(file)
	(o, p) = size(d)
	x = Matrix{Float64}(undef, o, p - 1)
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