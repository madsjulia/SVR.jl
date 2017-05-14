import SVR

currentdir = pwd()
cd(dirname(@__FILE__))

@Base.Test.testset "SVR" begin

    y_true = vec(readdlm("mg.result"))

    x, y = SVR.readlibsvmfile("mg.libsvm")

    pmodel = SVR.train(y, x');
    y_pr = SVR.predict(pmodel, x');
    @Base.Test.test isapprox(maximum(abs.(y_pr .- y_true)), 0, atol=1e-4)
    SVR.savemodel(pmodel, "mg.model")
    SVR.freemodel(pmodel)

    pmodel = SVR.loadmodel("mg.model")
    y_pr = SVR.predict(pmodel, x');
    # @assert maximum(abs.(y_pr .- y_true)) < 1e-4
    SVR.freemodel(pmodel)
end

cd(currentdir)