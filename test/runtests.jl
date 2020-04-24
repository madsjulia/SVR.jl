import SVR
import Test
import DelimitedFiles

currentdir = pwd()
cd(dirname(@__FILE__))

@Test.testset "SVR" begin
	y_true = vec(DelimitedFiles.readdlm("mg.result"))

	x, y = SVR.readlibsvmfile("mg.libsvm")

	pmodel = SVR.train(y, copy(permutedims(x)); tol=0.001, epsilon=0.1)
	y_pr = SVR.predict(pmodel, copy(permutedims(x)))
	@Test.test isapprox(maximum(abs.(y_pr .- y_true)), 0, atol=1e-4)
	SVR.savemodel(pmodel, "mg.model")
	SVR.freemodel(pmodel)

	pmodel = SVR.loadmodel("mg.model")
	y_pr = SVR.predict(pmodel, copy(permutedims(x)))
	# @assert maximum(abs.(y_pr .- y_true)) < 1e-4
	SVR.freemodel(pmodel)
end

cd(currentdir)
