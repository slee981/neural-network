include("./NeuralNet.jl")
include("./NeuralNetHelpers.jl")

using Random: randn
using .NeuralNet, .NeuralNetHelpers

# data 
#
# zero and one outcome var 
n = 10000 

c  = ones(n, 1)
x1 = [i / 1000 for i = 1:n]
x2 = round.(randn(n, 1) * 10)

u = randn(n, 1)

# dgp 
# 
# yhat = 4 + 2 * x1 - 3 * x2
# y    = yhat > mean(yhat) 
yhat = 4 * c + 2 * x1 - 10 * x2 + u
y    = yhat .> 0

Y = Float64.(y)
X = hcat(c, x1, x2)

# setup network 
net = Network(inputdim=3, cost=loss, dcost=dloss)
# addlayer!(net, 16, relu, drelu) 
# addlayer!(net, 32, relu, drelu) 
addlayer!(net, 8, sigmoid, dsigmoid) 
addlayer!(net, 16, sigmoid, dsigmoid) 
addlayer!(net, 4, sigmoid, dsigmoid) 
addlayer!(net, 1, sigmoid, dsigmoid) 

# fit network - stochastic gradient descent 
#
# 1- get minibatches 
# 2- for each batch in minibatch 
#    >> feedforward 
#    >> backpropagate 
fit!(net, X, Y, batchsize=16, epochs=16, learningrate=0.05)

# check predictions
# predict(net, X)
