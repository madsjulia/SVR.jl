import Distributed
Distributed.addprocs(4)
import Mads
import SVR

@Distributed.everywhere t = collect(0:0.1:10)
@Distributed.everywhere y = [1., 0.5]
@Distributed.everywhere x = sin.(permutedims(t)) .* y

Mads.plotseries(x'; names=["Train #1", "Train #2"])

@Distributed.everywhere pmodel = SVR.train(x, permutedims(y))
@Distributed.everywhere @show SVR.predict(pmodel[10], [0.75])

y_predict = [0.75]
x_predict = [SVR.predict(pmodel[i], y_predict)[1] for i = eachindex(t)]

Mads.plotseries([x' sin.(t) * y_predict[1]]; names=["Train #1", "Train #2", "True"], xmax=101)

Mads.plotseries([x' x_predict]; names=["Train #1", "Train #2", "Prediction"], xmax=101)

t = collect(0:0.1:10)
y = rand(100, 3)
x = y[:,1] .* t' .^ 0.5 + y[:,2] .* t' .+ y[:,3]

pmodel3 = SVR.train(x, permutedims(y))

y_predict = [0.75, 0.1, 0.2]
x_true = y_predict[1] .* t' .^ 0.5 + y_predict[2] .* t' .+ y_predict[3]
x_predict = [SVR.predict(pmodel3[i], y_predict)[1] for i = eachindex(t)]

Mads.plotseries(x')

Mads.plotseries([x_true' x_predict]; names=["True", "Prediction"], xmax=101)