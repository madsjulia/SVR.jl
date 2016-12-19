

#=
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
=#

#=
gcc -fPIC -c modsvm-train.c -o svm-test.o
gcc -shared -Wl,-soname,test.so svm-test.o -o test.so
=#

#=
features = readcsv("data/wells/1well_input_N=250.csv")
x = Array(svm_node, size(features, 2)+1, size(features, 1))
for i=1:size(x, 2)
    for j=1:size(features, 2)
	x[j, i] = svm_node(j, Float32(features[i, j]))
    end
    x[size(x, 1), i] = svm_node(-1, 0.0)
end
px = Array(Ptr{svm_node}, size(x, 2))
for i=1:size(px, 1)
    px[i] = pointer(x, 9*i+1)
end
ppx = pointer(px)
#y = map(x->Float32(x), readcsv("data/wells/1well_output_N=250.csv"))
y = readcsv("data/wells/1well_output_N=250.csv")
py = pointer(y)
l = size(y, 1)
prob = svm_problem(l, py, ppx)
mpprob = convert(Ptr{svm_problem}, pointer_from_objref(prob))
=#



#=
fillparam(svm_type=EPSILON_SVR, C=100.0, gamma=1.0, eps=0.03125),
fillparam(svm_type=EPSILON_SVR, C=1000.0, gamma=0.01, eps=0.1),
fillparam(svm_type=EPSILON_SVR, C=8096.0, gamma=1.0, eps=0.03125),
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