import Mads
import SVR

t = collect(0:0.1:10)
y = [1., 0.5]
x = sin.(permutedims(t)) .* y

pmodel = SVR.train(x, permutedims(y); tol=0.001, epsilon=0.1)

y_predict = [0.75]
x_predict = [SVR.predict(pmodel[i], y_predict)[1] for i = 1:length(t)]

Mads.plotseries(x_predict)

Mads.plotseries(sin.(t) * y_predict[1])