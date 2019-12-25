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

function csvreadproblem(csvinfile)
	p1 = trunc(readcsv(csvinfile), 5)
	pp1 = p1[:, 2:end]
	pp1 = pp1'
	ppn1, ppp1 = nodes(pp1)
	ppx1 = pointer(ppp1)
	y1 = p1[:, 1]
	py1 = pointer(y1)
	prob1 = svm_problem(size(y1, 1), py1, ppx1)
	pprob1 = pointer_from_objref(prob1)
	pprob1 = convert(Ptr{svm_problem}, pprob1)
	return pprob1, prob1
end

function jldreadproblem(jldinfile::String)
	p = JLD.load(jldinfile)
	p = trunc(p[collect(keys(p))[1]], 5)
	pp = p[:, 2:end]
	pp = pp'
	ppn, ppp = nodes(pp)
	ppx = pointer(ppp)
	y = p[:, 1]
	py = pointer(y)
	prob = svm_problem(size(y, 1), py, ppx)
	pprob = pointer_from_objref(prob)
	pprob = convert(Ptr{svm_problem}, pprob)
	return pprob, prob
end

function printout(fout, a)
	@printf(fout, "%f ", a[1])
	for i=2:size(a, 1)
		if a[i] != 0.0
			@printf(fout, " %d:%f", i-1, a[i])
		end
	end
	@printf(fout, "\n")
end

function readproblem(file)
	pprob = ccall((:read_problem, simlib), Ptr{svm_problem}, (Ptr{UInt8},), file)
end

function setupoutput(outfolder, modelfile)
	s = split(outfolder, ['\\', '/'])

	for i =1:size(s, 1)
		d = join(s[1:i], "/")
		if !isdir(d)
			mkdir(d)
		end
	end

	outfile = joinpath(outfolder, "wells_output")

	if ispath(outfile)
		rm(outfile)
	end
	if ispath(joinpath(outfolder, "predicted.csv"))
		rm(joinpath(outfolder, "predicted.csv"))
	end
	if ispath(joinpats = split(outfolder, ['\\', '/'])h(outfolder, "target.csv"))
		rm(joinpath(outfolder, "target.csv"))
	end
	if ispath(joinpath(outfolder, "info"))
		rm(joinpath(outfolder, "info"))
	end
	if ispath(modelfile)
		rm(modelfile)
	end
	outfile
end

function getbar()
	barlen = Base.displaysize()[2]-18
	return barlen
end

function params_from_opts(options::String)
	parammaker = string("fillparam(", options, ")")
	param = eval(parse(parammaker))
	pparam = convert(Ptr{svm_parameter}, pointer_from_objref(param))
	return pparam, param
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
	target = Array{Float64}(prob.l)

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

function runSVM(trailfile, testfile, outfolder, modelfile; dense::Bool=false)
	pparam, param = params_from_opts(options)
	pprob, prob = csvreadproblem(trailfile)

	pmodel, timeElapsed, success = train(pprob, pparam, dense=dense)

	ptest = readproblem(testfile)
	predicted, target, test, timeElapsed2 = predict(ptest, pmodel, dense=dense)

	resultanalysis(predicted, target, param, test, outfolder, timeElapsed, timeElapsed2)
end