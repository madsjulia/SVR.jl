import SVR
import Distributions

this_test_folder_path = dirname(Base.source_path())
include(joinpath(this_test_folder_path, "..", "julia", "src1.jl"))
include(joinpath(this_test_folder_path, "..", "tools", "grid.jl"))
make()

ar = [-10, 10]
br = [-10, 10]
cr = [-5, 5]
nr = [-3, 3]
t = [1,2,3,4,5]

if isdir(string(this_test_folder_path, "/../examples/"))
	rm(string(this_test_folder_path, "/../examples/"), recursive=true)
end
if !isdir(string(this_test_folder_path, "/../examples/"))
	mkdir(string(this_test_folder_path, "/../examples"))
end
if !isdir(string(this_test_folder_path, "/../examples/testjlds"))
	mkdir(string(this_test_folder_path, "/../examples/testjlds"))
end

function f_cvs(numPoints, ar, br, cr, nr, t, outJLD)
	arr = Array{Float64}(5*numPoints, 6)
	for i=1:numPoints
		a_d = Distributions.Uniform((ar[1]), (ar[end]))
		b_d = Distributions.Uniform((br[1]), (br[end]))
		c_d = Distributions.Uniform((cr[1]), (cr[end]))
		n_d = Distributions.Uniform((nr[1]), (nr[end]))
		a = rand(a_d)
		b = rand(b_d)
		c = rand(c_d)
		n = rand(n_d)
		for (j, ts) in enumerate(t)
			f = (a*float(ts)^(n))+(b*ts)+c
			arr[5*(i-1)+j, :] = [f, a, b, c, n, ts]'
#			print(".")
		end
	end
	save(outJLD, "data", arr)
	return arr
end

arr = f_cvs(100, ar, br, cr, nr, t, string(this_test_folder_path, "/../examples/training.jld"))
writecsv(string(this_test_folder_path, "/../examples/training.csv"), arr)
#print("*")
f_cvs(1000, ar, br, cr, nr, t, string(this_test_folder_path, "/../examples/testjlds/testing.jld"))
println()
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
convertSVM(string(this_test_folder_path, "/../examples/training.csv"), string(this_test_folder_path, "/../examples/training"))
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
best_mse, best_param = grid(1, [], [], string(this_test_folder_path, "/../examples/training.csv"), "-log2c 1,15,1 -log2g -7,0,0.5 -log2p -7,-3,1")
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
options = string("svm_type=EPSILON_SVR,C=", best_param["c"], ",gamma=", best_param["g"], ",p=", best_param["p"])
#options = string("svm_type=EPSILON_SVR,C=", 100, ",gamma=", 0.0, ",p=", 0.1)

runSVMJLD(string(this_test_folder_path, "/../examples/training"), string(this_test_folder_path, "/../examples/testjlds/"), string(this_test_folder_path, "/../examples/resultfolder"), "model"; options=options)
