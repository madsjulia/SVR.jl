module SVR

import JLD

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
const densesvmlib = abspath(joinpath(Pkg.dir("SVR"), "deps", "denselibsvm.so.2"))

function convertSVM(infile::String, outfile::String)
	fin = open(infile, "r")
	fout = open(outfile, "a")
	while true
		a = readline(fin)
		if a == ""
			break
		end
		a = split(a, ",")
		a = map(x->Float64(parse(x)), a)
		printout(fout, a)
	end
	close(fin)
	close(fout)
end

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

function fillparam(;svm_type=EPSILON_SVR,
	kernel_type=RBF,
	degree=3,
	gamma=0.0,
	coef0=0.0,
	nu=0.5,
	cache_size=100.0,
	C=1.0,
	eps=1e-3,
	p=0.1,
	shrinking=1,
	probability=0,
	nr_weight = 0,
	weight_label = Ptr{Int32}(0x0000000000000000),
	weight = Ptr{Float64}(0x0000000000000000))

	param = svm_parameter(svm_type,
		kernel_type,
		degree,
		gamma,
		coef0,
		cache_size,
		eps,
		C,
		nr_weight,
		weight_label,
		weight,
		nu,
		p,
		shrinking,
		probability)
	return param
end

function do_cross_validation(trailfile, nr_fold; options::String="")
	fileend = trailfile[end-3:end]
	if fileend == ".csv"
		pprob, prob = csvreadproblem(trailfile)
	elseif fileend == ".jld"
		pprob, prob = jldreadproblem(trailfile)
	else
		pprob = readproblem(trailfile)
	end

	pparam, param = params_from_opts(options)
	do_cross_validation(pprob, pparam, nr_fold)
end

function do_cross_validation(pprob, pparam, nr_fold)
	prob = unsafe_load(pprob)
	param = unsafe_load(pparam)
	total_correct = 0
	total_error = sumv = sumy = sumvv = sumyy = sumvy = 0.0
	target = Array(Float64, prob.l)

	ccall((:svm_cross_validation, svmlib), Void, (Ptr{svm_problem}, Ptr{svm_parameter}, Cint, Ptr{Float64}), pprob, pparam, nr_fold, pointer(target))

	if param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR
		for i=1:prob.l
			y = unsafe_load(prob.y, i)
			v = target[i]
			total_error += (v-y)*(v-y)
			sumv+=v
			sumy+=y
			sumvv+v*v
			sumyy+=y*y
			sumvy+=v*y
		end
		@printf("Cross Validation Mean squared error = %g\n",total_error/prob.l)
		@printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			)
	else
		for i=1:prob.l
			if target[i] == unsafe_load(prob.y, i)
	total_correct+=1
	@printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l)
	end
		end
	end
end

function trainSVM(pprob, pparam, modelfile; dense::Bool=false)
	if !dense
		timeElapsed = @elapsed pmodel = ccall((:svm_train, svmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
		success = ccall((:svm_save_model, svmlib), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)
	else
		timeElapsed = @elapsed pmodel = ccall((:svm_train, densesvmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
		success = ccall((:svm_save_model, densesvmlib), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)
	end
	return (pmodel, timeElapsed, success)
end

function predictSVM(ptest, pmodel; dense::Bool=false)
	test = unsafe_load(ptest)
	amountdone = 0
	target = Array(Float64, test.l)
	predicted = Array(Float64, test.l)
	#   println("checkpoint 1")
	timeElapsed2 = @elapsed for i=1:test.l
		target[i] = unsafe_load(test.y, i)
		point = unsafe_load(test.x, i)
		if !dense
			pred = ccall((:svm_predict, svmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
		else
			pred = ccall((:svm_predict, densesvmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
		end
		predicted[i] = pred
	end
	return predicted, target, test, timeElapsed2
end

function resultanalysis(predicted, target, param, outfolder, timeElapsed, timeElapsed2)
	error = sum((predicted .- target).*(predicted .- target))
	sump = sum(predicted)
	sumt = sum(target)
	sumpp = sum(predicted .* predicted)
	sumtt = sum(target .* target)
	sumpt = sum(predicted .* target)
	total = size(predicted, 2)

	if param.svm_type==NU_SVR || param.svm_type==EPSILON_SVR
		sqErr = error/total
		sqCorr = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
		println("Mean squared error =", trunc(sqErr, 2), "(regression)")
		println("Squared correlation coefficient =", trunc(sqCorr, 6), "(regression)")
	end

	writecsv(joinpath(outfolder, "predicted.csv"), predicted)
	writecsv(joinpath(outfolder, "target.csv"), target)
	f = open(joinpath(outfolder, "info"), "a")
	write(f, string("Mean squared error = ", trunc(sqErr, 2), "(regression)\n",
		"Squared correlation coefficient = ", trunc(sqCorr, 6), "(regression)\n",
		"time to train = ", timeElapsed, " seconds\n",
		"time to predict = ", timeElapsed2, " seconds\n"
		))
	close(f)
	return sqErr, sqCorr
end

function runSVM(trailfile, testfile, outfolder, modelfile; options::String="", dense::Bool=false)
	pparam, param = params_from_opts(options)
	pprob, prob = csvreadproblem(trailfile)

	pmodel, timeElapsed, success = trainSVM(pprob, pparam, dense=dense)

	ptest = readproblem(testfile)
	predicted, target, test, timeElapsed2 = predictSVM(ptest, pmodel, dense=dense)

	resultanalysis(predicted, target, param, test, outfolder, timeElapsed, timeElapsed2)
end

end