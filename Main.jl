#############################################################################
# LOAD PACKAGES

try
    # try to install required packages
    include("./NeuralNet.jl")
    include("./NeuralNetHelpers.jl")
catch 
    # if error, install requirements and try again
    include("./requirements.jl")
    include("./NeuralNet.jl")
    include("./NeuralNetHelpers.jl")
end

using Random: randn, shuffle
using .NeuralNet, .NeuralNetHelpers
using MLDatasets

#############################################################################
# EXAMPLE 1 - MNIST

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

X_train = reshape(train_x, (size(train_x, 3), 28 * 28))
Y_train = to_categorical(train_y)

X_test  = reshape(test_x, (size(test_x, 3), 28 * 28))
Y_test  = to_categorical(test_y)

# SETUP NETWORK - MNIST

input_size = size(X_train, 2)
output_size = size(Y_train, 2)

net = Network(inputdim=input_size, cost=loss_mse, dcost=dloss_mse)
addlayer!(net, 64, relu, drelu) 
addlayer!(net, 32, relu, drelu) 
addlayer!(net, output_size, softmax, dsoftmax) 

# fit network - stochastic gradient descent 
fit!(net, X_train, Y_train, batchsize=32, epochs=8, learningrate=0.01)

#############################################################################
# EXAMPLE 2 - SIMULATION ON KNOWN DATA GENERATING PROCESS

# n = 100000

# c  = ones(n, 1)
# x1 = [i / 1000 for i = 1:n]
# x2 = round.(randn(n, 1) * 10)

# u = randn(n, 1)

# # dgp 
# # 
# # yhat = 4+ 2*x1 - 3*x2 + x1+x2
# # y    = yhat > mean(yhat) 
# yhat = 4 * c + 2 * x1 - 10 * x2 + x1 .* x2 + u

# Y = yhat .> 0
# X = hcat(c, x1, x2)

# # test and training 
# trainpct = 0.8
# ntrain = Int(n * trainpct)

# shuffledrows = shuffle(1:n)
# trainrows = shuffledrows[1:ntrain]
# testrows = shuffledrows[ ntrain + 1:end]

# X_train = X[trainrows, :]
# Y_train = Y[trainrows, :]
# X_test  = X[testrows , :]
# Y_test  = Y[testrows , :]

# # SETUP NETWORK 

# input_size = size(X_train, 2)
# output_size = size(Y_train, 2)

# net = Network(inputdim=input_size, cost=loss_mse, dcost=dloss_mse)
# addlayer!(net, 16, relu, drelu) 
# addlayer!(net, 4, relu, drelu) 
# addlayer!(net, output_size, sigmoid, dsigmoid) 

# # fit network - stochastic gradient descent 
# fit!(net, X_train, Y_train, batchsize=8, epochs=8, learningrate=0.01)

