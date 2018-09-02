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

function readproblem(file)
  pprob = ccall((:read_problem, "/n/srv/vessg/Downloads/libsvm-3.21/test.so"), Ptr{svm_problem}, (Ptr{UInt8},), file)
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

function setupoutput(outfolder)
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
  if ispath(string(outfolder, "/predict.csv"))
    rm(string(outfolder, "/predict.csv"))
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

function predict(pprob, param, modelfile)
  println("\n\nbegin training\n")

  timeElapsed = @elapsed pmodel = ccall((:svm_train, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
  success = ccall((:svm_save_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)

  return (pmodel, timeElapsed, success)
end

function test(outfile, pmodel)
  println("\n\nbegin predictions\n")
  ptest = ccall((:read_problem, "/n/srv/vessg/Downloads/libsvm-3.21/test.so"), Ptr{svm_problem}, (Ptr{UInt8},), testfile)
  test = unsafe_load(ptest)

  f = open(outfile, "a")


  function getbar()
    barlen = Base.displaysize()[2]-18
    return barlen
  end

  amountdone = 0
  target = Array{Float64}(test.l)
  predicted = Array{Float64}(test.l)
  barlen = getbar()
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



      target[i] = unsafe_load(test.y, i)
      point = unsafe_load(test.x, i)
      pred = ccall((:svm_predict, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), Float64, (Ptr{svm_model}, Ptr{svm_node}), pmodel, point)
      predicted[i] = pred
      write(f, string(pred, "\n"))
  end
  close(f)
  println()
  return predicted
end

function resultanalysis(predicted, target, param)
  error = sum((predicted .- target).*(predicted .- target))
  sump = sum(predicted)
  sumt = sum(target)
  sumpp = sum(predicted .* predicted)
  sumtt = sum(target .* target)
  sumpt = sum(predicted .* target)
  total = test.l



  if param.svm_type==NU_SVR || param.svm_type==EPSILON_SVR
    sqErr = error/total
    sqCorr = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
    println("Mean squared error =", trunc(sqErr, 2), "(regression)")
    println("Squared correlation coefficient =", trunc(sqCorr, 6), "(regression)")
  end


  writecsv(string(outfolder, "/predict.csv"), predict)
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







function run(trailfile, testfile, outfolder, modelfile, options...)

  s=C_SVC,
  t=RBF,
  d=3,
  g=0.0,
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
  weight = Ptr{Float64}(0x0000000000000000)


  for option in options
    eval(parse(option))
  end

  trailfile = "data/wells/wells_trails"
  testfile = "data/wells/wells_tests"
  outfolder = "data/output/curr"
  modelfile = string(outfolder, "/wells_model")


  pprob = readproblem(trailfile)

  param = fillparam(svm_type=s,
		    C=C,
		    gamma=g,
		    eps=eps,
		    kernel_type=t,
		    degree=d,
		    coef0=coef0,
		    nu=nu,
		    cache_size=cache_size,
		    p=p,
		    shrinking=shrinking,
		    probability=probability,
		    nr_weight=nr_weight,
		    weight_label=weight_label,
		    weight=weight)

  outfile = setupoutput(outfolder)

  pparam = convert(Ptr{svm_parameter}, pointer_from_objref(param))

  pmodel, timeElapsed, success = predict(pprob, param, modelfile)

  predicted = test(outfile, pmodel)

  resultanalysis(predicted, target, param)

end