import SVR
using Test
using DelimitedFiles

currentdir = pwd()
cd(dirname(@__FILE__))

@testset "SVR" begin
	y_true = vec(readdlm("mg.result"))

	x, y = SVR.readlibsvmfile("mg.libsvm")

	pmodel = SVR.train(y, copy(x'));
	y_pr = SVR.predict(pmodel, copy(x'));
	@test isapprox(maximum(abs.(y_pr .- y_true)), 0, atol=1e-4)
	SVR.savemodel(pmodel, "mg.model")
	SVR.freemodel(pmodel)

	pmodel = SVR.loadmodel("mg.model")
	y_pr = SVR.predict(pmodel, copy(x'));
	# @assert maximum(abs.(y_pr .- y_true)) < 1e-4
	SVR.freemodel(pmodel)
end

cd(currentdir)
