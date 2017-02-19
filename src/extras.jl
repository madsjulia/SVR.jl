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
	if VERSION < v"0.5"
		barlen = Base.tty_size()[2]-18
	else
		barlen = Base.displaysize()[2]-18
	end
	return barlen
end

function params_from_opts(options::String)
	parammaker = string("fillparam(", options, ")")
	param = eval(parse(parammaker))
	pparam = convert(Ptr{svm_parameter}, pointer_from_objref(param))
	return pparam, param
end