import SVR
import Mads

X = sort(rand(40) * 5)
y = sin.(X)

Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.RBF)], "figures/rbf.png"; title="RBF", names=["Truth", "Prediction"])
Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.LINEAR)], "figures/linear.png"; title="Linear", names=["Truth", "Prediction"])
Mads.plotseries([y SVR.fit(y, permutedims(X); kernel_type=SVR.POLY, coef0=1.)], "figures/poly.png"; title="Polynomial", names=["Truth", "Prediction"])