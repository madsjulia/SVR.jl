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

const C_SVC = Int32(0)
const NU_SVC = Int32(1)
const ONE_CLASS = Int32(2)
const EPSILON_SVR = Int32(3)
const NU_SVR = Int32(4)

const LINEAR = Int32(0)
const POLY = Int32(1)
const RBF = Int32(2)
const SIGMOID = Int32(3)
const PRECOMPUTED = Int32(4)



shell_path = pwd()
cd(dirname(Base.source_path()))
const svmlib = abspath("../libsvm.so.2")
const densesvmlib = abspath("../denselibsvm.so.2")
const testlib = abspath("../trainlib.so.2")
# const conpath = abspath("../convertlib.so.2")
cd(shell_path)

function convertSVM(infile, outfile)
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

function nodes(instances)
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs = Array(Ptr{svm_node}, ninstances)
    nodes = Array(svm_node, nfeatures + 1, ninstances)

    for i=1:ninstances
        k = 1
        for j=1:nfeatures
            nodes[k, i] = svm_node(Int32(j), Float64(instances[j, i]))
            k += 1
        end
        nodes[k, i] = svm_node(Int32(-1), 0.0)
        nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
    end

    (nodes, nodeptrs)
end

function csvreadproblem(csvinfile)
  p = trunc(readcsv(csvinfile), 5)
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

function jldreadproblem(jldinfile)
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
  pprob = ccall((:read_problem, testlib), Ptr{svm_problem}, (Ptr{UInt8},), file)
end

function fillparam(;svm_type=C_SVC,
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

function setupoutput(outfolder, modelfile)
  s = split(outfolder, ['\\', '/'])

  for i =1:size(s, 1)
    d = join(s[1:i], "/")
    if !isdir(d)
      mkdir(d)
    end
  end

  outfile = string(outfolder, "/wells_output")

  if ispath(outfile)
    rm(outfile)
  end
  if ispath(string(outfolder, "/predicted.csv"))
    rm(string(outfolder, "/predicted.csv"))
  end
  if ispath(string(outfolder, "/target.csv"))
    rm(string(outfolder, "/target.csv"))
  end
  if ispath(string(outfolder, "/info"))
    rm(string(outfolder, "/info"))
  end
  if ispath(modelfile)
    rm(modelfile)
  end
  outfile
end


function do_cross_validation(trailfile, nr_fold; options="")
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





function trainSVM(pprob, pparam, modelfile; dense=false)
  println("\n\nbegin training\n")
  if !dense
    timeElapsed = @elapsed pmodel = ccall((:svm_train, svmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
    success = ccall((:svm_save_model, svmlib), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)
  else
    timeElapsed = @elapsed pmodel = ccall((:svm_train, densesvmlib), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
    success = ccall((:svm_save_model, densesvmlib), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)
  end
  return (pmodel, timeElapsed, success)
end

function getbar()
  barlen = Base.displaysize()[2]-18
  return barlen
end

function predictSVM(ptest, outfile, pmodel; dense=false)
  println("\n\nbegin predictions\n")
  test = unsafe_load(ptest)

  f = open(outfile, "a")

  amountdone = 0
  target = Array{Float64}(test.l)
  predicted = Array{Float64}(test.l)
  barlen = getbar()
#   println("checkpoint 1")
  timeElapsed2 = @elapsed for i=1:test.l
      if i%round(Int, test.l/barlen)==0
	  barlen = getbar()
	  print("\r")
	  amountdone = round(Int, i/(round(Int, test.l/barlen)))
	  percentage = (round((i/test.l)*10000))/100
	  print(percentage, "% done \t|")
	  for j=1:barlen
	      if j<=amountdone
		  print("█")
	      else
		  print(" ")
	      end
	  end
	  print("|")
      elseif i==1
	print("\r")
	  percentage = (round((i/test.l)*10000))/100
	  print(percentage, "% done \t|")
	  for j=1:barlen
	      print(" ")
	  end
	  print("|")
      elseif i==test.l
	print("\r")
	  percentage = (round((i/test.l)*10000))/100
	  print(percentage, "% done \t|")
	  for j=1:barlen
	      print("█")
	  end
	  print("|")
      end

#       println("checkpoint 2")

      target[i] = unsafe_load(test.y, i)
      point = unsafe_load(test.x, i)
      if !dense
	pred = ccall((:svm_predict, svmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
      else
	pred = ccall((:svm_predict, densesvmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
      end
      predicted[i] = pred
      write(f, string(pred, "\n"))
#       println("checkpoint 3")
  end
  close(f)
  println()
  return predicted, target, test, timeElapsed2
end












function predictSVM2(testfile, outfile, pmodel; dense=false)
  println("\n\nbegin predictions\n")
  ftest = open(testfile, "r")
  f = open(outfile, "a")

  templ = 100000
  amountdone = 0
  finaltarget = Array{Float64}(0)
  finalpredicted = Array{Float64}(0)
  target = Array{Float64}(templ)
  predicted = Array{Float64}(templ)
  a = "derp"
  done = false
  endi = 0
  timeElapsed2 = @elapsed while true
    for i=1:templ
	a = readline(ftest)
	if a == ""
	  done = true
	  endi = i
	  break
	end
	if i%10000 == 0
	  print(".")
	end
# 	println(a)
	a = split(a, ",")
# 	println(a)
	a = map(x->parse(x), a)
	target[i] = a[1]
	a = a[2:end]
	na, pna = nodes(a)
	point = pna[1]
	if !dense
	  pred = ccall((:svm_predict, svmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
	else
	  pred = ccall((:svm_predict, densesvmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
	end
	predicted[i] = pred
	write(f, string(pred, "\n"))
    end
    print("*")
    if !done
      append!(finaltarget, target)
      append!(finalpredicted, predicted)
    else
      append!(finaltarget, target[1:endi])
      append!(finalpredicted, predicted[1:endi])
      break
    end
  end
  close(ftest)
  close(f)
  println()
  return finalpredicted, finaltarget, timeElapsed2
end








function predictSVM3(testdirectory, outfile, pmodel; dense=false)
  println("\n\nbegin predictions\n")
  f = open(outfile, "a")

  templ = 0
  amountdone = 0
  finaltarget = Array{Float64}(0)
  finalpredicted = Array{Float64}(0)
  target = Array{Float64}(templ)
  predicted = Array{Float64}(templ)

  s = testdirectory
  s = split(s, "/")
  s = s[1:end-1]
  s = join(s, "/")
  dirs = readdir(s)

  timeElapsed2 = @elapsed for j=1:size(dirs, 1)
    dir = dirs[j]
    d = load(string(testdirectory, dir))
    data = d[collect(keys(d))[1]]
    target = data[:, 1]
    predicted = Array{Float64}(size(target)...)
    features = data[:, 2:end]'
    nodefeatures, pna = nodes(features)
    templ = size(target, 1)

    for i=1:templ
	if i%10000 == 0
	  print(".")
	end
	point = pna[i]
	if !dense
	  pred = ccall((:svm_predict, svmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
	else
	  pred = ccall((:svm_predict, densesvmlib), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
	end
	predicted[i] = pred
	write(f, string(pred, "\n"))
    end
    print("*")
    append!(finaltarget, target)
    append!(finalpredicted, predicted)
  end
  close(f)
  println()
  return finalpredicted, finaltarget, timeElapsed2
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


  writecsv(string(outfolder, "/predicted.csv"), predicted)
  writecsv(string(outfolder, "/target.csv"), target)
  f = open(string(outfolder, "/info"), "a")
  write(f, string("Mean squared error = ", trunc(sqErr, 2), "(regression)\n",
    "Squared correlation coefficient = ", trunc(sqCorr, 6), "(regression)\n",
    "time to train = ", timeElapsed, " seconds\n",
    "time to predict = ", timeElapsed2, " seconds\n"
    ))
  close(f)
  return sqErr, sqCorr
end

function params_from_opts(options)
  parammaker = string("fillparam(", options, ")")
  param = eval(parse(parammaker))
  pparam = convert(Ptr{svm_parameter}, pointer_from_objref(param))
  return pparam, param
end

function runSVM(trailfile, testfile, outfolder, modelfile; options="", dense=false)

  pparam, param = params_from_opts(options)
  pprob, prob = csvreadproblem(trailfile)

  modelfile = string(outfolder, "/", modelfile)
  outfile = setupoutput(outfolder, modelfile)


  pmodel, timeElapsed, success = trainSVM(pprob, pparam, modelfile, dense=dense)

  ptest = readproblem(testfile)
  predicted, target, test, timeElapsed2 = predictSVM(ptest, outfile, pmodel, dense=dense)

  resultanalysis(predicted, target, param, test, outfolder, timeElapsed, timeElapsed2)

end

function runSVM2(trailfile, testfile, outfolder, modelfile; options="", dense=false)

  fileend = trailfile[end-3:end]
  if fileend == ".csv"
  pprob, prob = csvreadproblem(trailfile)
  elseif fileend == ".jld"
  pprob, prob = jldreadproblem(trailfile)
  else
  pprob = readproblem(trailfile)
  end

  pparam, param = params_from_opts(options)

  modelfile = string(outfolder, "/", modelfile)
  outfile = setupoutput(outfolder, modelfile)


  pmodel, timeElapsed, success = trainSVM(pprob, pparam, modelfile, dense=dense)

  predicted, target, timeElapsed2 = predictSVM3(testfile, outfile, pmodel, dense=dense)

  resultanalysis(predicted, target, param, outfolder, timeElapsed, timeElapsed2)

end
