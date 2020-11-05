include("./NeuralNet.jl")
include("./NeuralNetHelpers.jl")

using Random: randn, shuffle
using .NeuralNet, .NeuralNetHelpers

# data 

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

Y = yhat .> 0
X = hcat(c, x1, x2)

# test and training 
trainpct = 0.8
ntrain = Int(n * trainpct)

shuffledrows = shuffle(1:n)
trainrows = shuffledrows[1:ntrain]
testrows = shuffledrows[ ntrain + 1:end]

xtrain = X[trainrows, :]
ytrain = Y[trainrows, :]
xtest  = X[testrows , :]
ytest  = Y[testrows , :]

# setup network 
net = Network(inputdim=3, cost=loss, dcost=dloss)
addlayer!(net, 16, sigmoid, dsigmoid) 
addlayer!(net, 64, sigmoid, dsigmoid) 
addlayer!(net, 4, sigmoid, dsigmoid) 
addlayer!(net, 1, sigmoid, dsigmoid) 

# fit network - stochastic gradient descent 
fit!(net, X, Y, batchsize=4, epochs=16, learningrate=0.1)

# check predictions
# predict(net, X)
