using Distributions

this_test_folder_path = dirname(Base.source_path())
include(string(this_test_folder_path, "/../julia/src1.jl"))
include(string(this_test_folder_path, "/../tools/grid.jl"))

ar = [-10, 10]
br = [-10, 10]
cr = [-5, 5]
nr = [-3, 3]
t = [1,2,3,4,5]

function f_cvs(numPoints, ar, br, cr, nr, t, outJLD)
      arr = Array(Float64, 5*numPoints, 6)
      for i=1:numPoints
              a_d =DiscreteUniform((ar[1]), (ar[end]))
              b_d = DiscreteUniform((br[1]), (br[end]))
              c_d = DiscreteUniform((cr[1]), (cr[end]))
              n_d = DiscreteUniform((nr[1]), (nr[end]))
              a = rand(a_d)
              b = rand(b_d)
              c = rand(c_d)
              n = rand(n_d)
              for (j, ts) in enumerate(t)
		      f = (a*float(ts)^(n))+(b*ts)+c
		      arr[5*(i-1)+j, :] = [f, a, b, c, n, ts]'
	              print(".")
              end
      end
      save(outJLD, "data", arr)
      return arr
end


        
arr = f_cvs(100, ar, br, cr, nr, t, "training.jld")
writecsv("training.csv", arr)
print("*")
f_cvs(1000, ar, br, cr, nr, t, "jlds/testing.jld")
convertSVM("training.csv", "training")
best_mse, best_param = grid(1, [], [], "training.csv", "-log2c 1,3,1")

#runSVMJLD("training", "testing.jld", "resultfolder", "model"; options=options)
