module NeuralNetHelpers 
using LinearAlgebra, Statistics

export loss_mse, 
       dloss_mse, 
       sigmoid, 
       dsigmoid, 
       relu, 
       drelu, 
       to_categorical, 
       softmax, 
       dsoftmax

function loss_mse(output::Number, truth::Number)::Number
    return 0.5 * (output - truth)^2
end

function loss_mse(output::AbstractArray, truth::AbstractArray)::AbstractArray
    # we want to return a scalar valued loss i.e. a "reduction"
    # first, take the pairwise mean squared error loss
    # second, mean across the column vector of losses 
    loss_vec = loss_mse.(output, truth) 
    return mean(loss_vec, dims=1)
end

function dloss_mse(output::Number, truth::Number)::Number
    return output - truth
end

function dloss_mse(output::AbstractArray, truth::AbstractArray)::AbstractArray
    return dloss_mse.(output, truth)
end

function sigmoid(z::Number)::Number
    return 1 / (1 + exp(-z))
end

function sigmoid(z::AbstractArray)::AbstractArray
    return sigmoid.(z)
end

function dsigmoid(z::AbstractArray)::AbstractArray
    # we want to return the full jacobian
    # in this case a diagonal matric i.e. diagm
    dvec = sigmoid(z) .* (1 .- sigmoid(z))
    return diagm(dvec[:, 1])
end

function relu(z::AbstractArray)::AbstractArray
    return max.(0, z)
end

function drelu(z::Number)::Number
    return z > 0 ? 1.0 : 0.0
end

function drelu(z::AbstractArray)::AbstractArray
    # we want to return the jacobian
    # in this case a diagonal matric i.e. diagm
    dvec = drelu.(z)
    return diagm(dvec[:, 1])
end

function softmax(z::AbstractArray)::AbstractArray
    shift = maximum(z)

    z = exp.(z .- shift)
    total = sum(z)
    return z ./ total
end

function dsoftmax(z::AbstractArray)::AbstractArray
    # dsoftmax / dz should return a jacobian matrix 
    #
    # see here for explanation:
    # https://datascience.stackexchange.com/questions/51677/derivation-of-backpropagation-for-softmax 
    n = length(z)

    eye = Array{Number}(I, n, n)
    w   = softmax(z)
    e   = ones(n, 1)

    return w * e' .* (eye - e * w')
end

function to_categorical(v::Vector)::AbstractArray
    # input : vector of categorical vars e.g. [1, 3, 5, 2, 0, ...]
    # output: matrix of one hot obvs
    max = maximum(v)
    min = minimum(v)  # is zero included? 
    nobvs = length(v)

    # create zeros matrix of placeholder 
    # note, observations in the rows
    if min == 0
        # increment every element up one since julia isn't 0-indexed
        v .+= 1
        onehotmat = zeros(nobvs, max + 1)
    else 
        onehotmat = zeros(nobvs, max)
    end

    for i = 1:length(v)
        ele = v[i]
        onehotmat[i, ele] = 1
    end
    return onehotmat
end
end