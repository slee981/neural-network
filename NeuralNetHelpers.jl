module NeuralNetHelpers 

export loss, dloss, sigmoid, dsigmoid, relu, drelu

function loss(output::Float64, truth::Float64)::Float64
    return (output - truth)^2
end

function loss(output::Array{Float64,2}, truth::Array{Float64,2})::Array{Float64,2}
    return loss.(output, truth)    
end

function dloss(output::Float64, truth::Float64)::Float64
    return output - truth
end

function dloss(output::Array{Float64,2}, truth::Array{Float64,2})::Array{Float64,2}
    return dloss.(output, truth)
end

function sigmoid(z::Float64)::Float64
    return 1 / (1 + exp(-z))
end

function sigmoid(z::Array{Float64,2})::Array{Float64,2}
    return 1 ./ (1 .+ exp.(-z))
end

function dsigmoid(z::Array{Float64,2})::Array{Float64,2}
    return sigmoid(z) .* (1 .- sigmoid(z))
end

function relu(z::Array{Float64,2})::Array{Float64,2}
    return max.(0, z)
end

function drelu(z::Float64)::Float64
    return z > 0 ? 1.0 : 0.0
end

function drelu(z::Array{Float64,2})::Array{Float64,2}
    return drelu.(z)
end
end