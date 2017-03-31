import SVR

currentdir = pwd()
cd(dirname(@__FILE__))

x, y = SVR.readlibsvmfile("mg.libsvm")
pmodel = SVR.train(y, x');
y_pr = SVR.predict(pmodel, x');
# writedlm("mg.result", y_pr)
y_true = vec(readdlm("mg.result"))
@assert maximum(abs.(y_pr .- y_true)) < 1e-4
SVR.savemodel(pmodel, "mg.model")

SVR.freemodel(pmodel)
pmodel = SVR.loadmodel("mg.model")
y_pr = SVR.predict(pmodel, x');
y_true = vec(readdlm("mg.result"))
@assert maximum(abs.(y_pr .- y_true)) < 1e-4
SVR.freemodel(pmodel)

cd(currentdir)