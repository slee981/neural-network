module NeuralNetHelpers 

export loss, dloss, sigmoid, dsigmoid, relu, drelu

function loss(output::Number, truth::Number)::Number
    return (output - truth)^2
end

function loss(output::AbstractMatrix, truth::AbstractMatrix)::AbstractMatrix
    return loss.(output, truth)    
end

function dloss(output::Number, truth::Number)::Number
    return output - truth
end

function dloss(output::AbstractMatrix, truth::AbstractMatrix)::AbstractMatrix
    return dloss.(output, truth)
end

function sigmoid(z::Number)::Number
    return 1 / (1 + exp(-z))
end

function sigmoid(z::AbstractMatrix)::AbstractMatrix
    return 1 ./ (1 .+ exp.(-z))
end

function dsigmoid(z::AbstractMatrix)::AbstractMatrix
    return sigmoid(z) .* (1 .- sigmoid(z))
end

function relu(z::AbstractMatrix)::AbstractMatrix
    return max.(0, z)
end

function drelu(z::Number)::Number
    return z > 0 ? 1.0 : 0.0
end

function drelu(z::AbstractMatrix)::AbstractMatrix
    return drelu.(z)
end
end