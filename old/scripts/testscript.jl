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

#=
type SVMModel{T}
    ptr::Ptr{Void}
    param::Vector{SVMParameter}

    # Prevent these from being garbage collected
    problem::Vector{SVMProblem}
    nodes::Array{SVMNode}
    nodeptr::Vector{Ptr{SVMNode}}

    labels::Vector{T}
    weight_labels::Vector{Int32}
    weights::Vector{Float64}
    nfeatures::Int
    verbose::Bool
end
=#
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

immutable tester
  param::svm_parameter
  a1::UInt128
  a2::UInt128
  a3::UInt128
  a4::UInt128
  a5::UInt128
  a6::UInt128
  a7::UInt128
  a8::UInt128
  a9::UInt128
  a10::UInt128
  a11::UInt128
  a12::UInt128
  a13::UInt128
  a14::UInt128
  a15::UInt128
  a16::UInt128
  a17::UInt128
  a18::UInt128
  a19::UInt128
  a20::UInt128
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
        nodes[k, i] = svm_node(Int32(-1), NaN)
        nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
    end

    (nodes, nodeptrs)
end


#=
gcc -fPIC -c modsvm-train.c -o svm-test.o
gcc -shared -Wl,-soname,test.so svm-test.o -o test.so
=#

trailfile = "data/wells/wells_trails"
testfile = "data/wells/wells_tests"
outpathgeneral = "data/output/"

pprob = ccall((:read_problem, "/n/srv/vessg/Downloads/libsvm-3.21/test.so"), Ptr{svm_problem}, (Ptr{UInt8},), trailfile)
prob = unsafe_load(pprob)
pf = prob.y
labels = Array{Float64}(prob.l)
for i=1:prob.l
    labels[i] = unsafe_load(pf, i)
end

px = prob.x

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

params = Array(svm_parameter, 0)
push!(params,
fillparam(svm_type=EPSILON_SVR, C=8096.0, gamma=1.0, eps=0.03125),

#=fillparam(svm_type=EPSILON_SVR, C=1000.0, gamma=0.01, eps=0.1),
fillparam(svm_type=EPSILON_SVR, C=100.0, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=5000.0, gamma=0.1, eps=0.1),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=0.1, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=rand()*8000+1000, gamma=1.0, eps=0.03125),
=#
)

for iter=1:size(params, 1)


param = params[iter]
outfolder = string(outpathgeneral, "C=", round(Int, param.C), "_gamma=",
round(Int, param.gamma*10), "_eps=", round(Int, param.eps*1000))

s = split(outfolder, ['\\', '/'])

for i =1:size(s, 1)
  d = join(s[1:i], "/")
  if !isdir(d)
    mkdir(d)
  end
end

outfile = string(outfolder, "/wells_output")
modelfile = string(outfolder, "/wells_model")

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






pparam = convert(Ptr{svm_parameter}, pointer_from_objref(param))


timeElapsed = @elapsed pmodel = ccall((:svm_train, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), Ptr{svm_model}, (Ptr{svm_problem},Ptr{svm_parameter}), pprob, pparam)
success = ccall((:svm_save_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), Int32, (Ptr{UInt8},Ptr{svm_model}), modelfile, pmodel)

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
predict = Array{Float64}(test.l)
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
    predict[i] = pred
    write(f, string(pred, "\n"))
end
close(f)
println()

error = sum((predict .- target).*(predict .- target))
sump = sum(predict)
sumt = sum(target)
sumpp = sum(predict .* predict)
sumtt = sum(target .* target)
sumpt = sum(predict .* target)
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


end



#a = ccall((:svm_load_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), svm_model,(ASCIIString,), "/n/srv/vessg/Downloads/libsvm-3.21/data/wells/wells_model")
#b = ccall((:svm_load_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), svm_model,(Ptr{UInt8},), pointer("/n/srv/vessg/Downloads/libsvm-3.21/data/wells/wells_model"))
#c = ccall((:svm_load_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), tester, (ASCIIString,), "/n/srv/vessg/Downloads/libsvm-3.21/data/wells/wells_model")
#d = ccall((:svm_load_model, "/n/srv/vessg/Downloads/libsvm-3.21/libsvm.so.2"), tester, (Ptr{UInt8},), pointer("/n/srv/vessg/Downloads/libsvm-3.21/data/wells/wells_model"))




# #iter = 7410
#nu = 0.996970
#obj = -8882090.957074, rho = -223.197278
#nSV = 12468, nBSV = 12454
#svm_type epsilon_svr
#kernel_type rbf
#gamma 0.01
#nr_class 2
#total_sv 12468
#rho -223.197



#struct svm_model *svm_load_model(const char *model_file_name);
