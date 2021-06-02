import SVR
import Test
import DelimitedFiles

currentdir = pwd()
cd(dirname(@__FILE__))

@Test.testset "SVR" begin
	y_true = vec(DelimitedFiles.readdlm("mg.result"))

	x, y = SVR.readlibsvmfile("mg.libsvm")

	pmodel = SVR.train(y, permutedims(x); tol=0.001, epsilon=0.1, scale=false, normalize=false)
	y_pr = SVR.predict(pmodel, permutedims(x))
	@Test.test isapprox(maximum(abs.(y_pr .- y_true)), 0.17675, atol=1e-4)
	SVR.savemodel(pmodel, "mg.model")
	SVR.freemodel(pmodel)

	pmodel2 = SVR.loadmodel("mg.model")
	y_pr2 = SVR.predict(pmodel2, permutedims(x))
	@Test.test maximum(abs.(y_pr .- y_pr2)) < 1e-4
end

cd(currentdir)