module NeuralNet

using LinearAlgebra, Random, Statistics, Logging, Plots
export Network, addlayer!, fit!

mutable struct Layer
    weights::AbstractMatrix
    bias::AbstractMatrix

    # each layer can have its own activation function
    activation::Function
    dactivation::Function

    # cache 
    # 1- the last linear transformation z = Ax + b
    # 2- activated output = activation(z) 
    linear_out::AbstractMatrix
    activated_out::AbstractMatrix

    # keep track of partial derivative error for each batch 
    dC_dlinear::AbstractMatrix 
    dC_dweights::AbstractMatrix 
    dC_dbias::AbstractMatrix 

    function Layer(insize::Number, outsize::Number, activation::Function, dactivation::Function)
        weights = randn(outsize, insize)
        bias = randn(outsize, 1)

        linear_out = zeros(outsize, 1)
        activated_out = zeros(outsize, 1)

        dC_dlinear = zeros(outsize, 1)
        dC_dweights = zeros(outsize, insize)
        dC_dbias = zeros(outsize, 1)
        new(weights, bias, activation, dactivation, linear_out, activated_out, dC_dlinear, dC_dweights, dC_dbias)
    end
end

mutable struct Network 
    inputdim::Int16
    layers::Vector{Layer}
    cost::Function
    dcost::Function

    function Network(;inputdim, cost, dcost)  
        new(inputdim, Array{Layer}[], cost, dcost)
    end
end

function addlayer!(net::Network, outsize::Number, activation::Function, dactivation::Function)
    if length(net.layers) == 0
        insize = net.inputdim
    else 
        lastlayer = net.layers[end]
        insize = size(lastlayer.weights, 1)     # num rows in previous output
    end
    layer = Layer(Number(insize), outsize, activation, dactivation)
    push!(net.layers, layer)
end

function fit!(net::Network, x::AbstractMatrix, y::AbstractMatrix; batchsize=1, epochs=2, learningrate=0.5)
    # input
    #     ~ net : neural network to fit 
    #     ~ x   : input args with variables in columns, observation in rows   
    #     ~ y   : the "true" values
    sgd!(net, x, y, batchsize, epochs, learningrate)
end

function sgd!(net::Network, x::AbstractMatrix, y::AbstractMatrix, batchsize::Number, epochs::Number, learningrate::Number)
    # stochastic gradient descent (sgd)

    lossvals = Vector{Number}()
    for epoch = 1:epochs

        # shuffle rows of matrix  
        nobs, nvars = size(x)
        shuffledrows = shuffle(1:nobs)
        x = x[shuffledrows, :]
        y = y[shuffledrows, :]

        # create mini batches and loop through each batch 
        # note: julia is NOT zero indexed 
        #       i.e. x[1] is the first element
        for batchend = batchsize:batchsize:nobs
            batchstart = batchend - batchsize + 1

            # get sample of rows i.e. observations 
            # and transpose into columns for feedforward
            xbatch = x[batchstart:batchend, :]'
            ybatch = y[batchstart:batchend, :]'

            # average losses for each sample in batch
            losses = zeros(size(ybatch))
            for icol in 1:batchsize
                xi    = xbatch[:, icol:icol]
                ytrue = ybatch[:, icol:icol]

                # feedforward
                out = feedforward!(net, xi)

                # calculate loss 
                iloss = net.cost(out, ytrue)

                # store loss for later average
                losses[:, icol:icol] = iloss

                # calculate partial derivatives of each weight and bias
                # i.e. backpropagate
                backpropagate!(net, xi, ytrue)
            end

            if batchend % 100 == 0 

                # sample average loss from batch to plot 
                meanloss = mean(losses)
                push!(lossvals, meanloss)
            
                plotloss(lossvals)
            end

            # update weights and bias 
            update!(net, batchsize, learningrate)
        end
    end
    plotloss(lossvals, enter2close=true)
end

function plotloss(lossvalues::Vector{Number}; enter2close=false)
    p = plot(lossvalues, legend=false)
    gui(p)
    if enter2close
        println("\nPress 'Enter' to close the plot:")
        x = readline()
    end
end

function update!(net::Network, batchsize::Number, learningrate::Number)

    # update weights in each layer based on the error terms dC_dweights, dC_dbias
    for i = 1:length(net.layers)
        layer = net.layers[i]

        layer.weights -= learningrate / batchsize * layer.dC_dweights
        layer.bias    -= learningrate / batchsize * layer.dC_dbias

        # reset the error terms for next batch 
        nrows, ncols = size(layer.weights)

        layer.dC_dlinear  = zeros(nrows, 1)
        layer.dC_dweights = zeros(nrows, ncols)
        layer.dC_dbias    = zeros(nrows, 1)
    end
end

function calcpartials!(net::Network, x::AbstractMatrix, truth::AbstractMatrix)

    # calculate last layer partials (i.e. deltas)
    # wrt the linear transformation z = Ax + b
    # thus, 
    # dc/dzL = dloss(output) * dactivation(z)
    lastlayer = net.layers[end]

    lastlayer.dC_dlinear   = net.dcost(lastlayer.activated_out, truth) .* lastlayer.dactivation(lastlayer.linear_out)
    lastlayer.dC_dweights += lastlayer.dC_dlinear * net.layers[end - 1].activated_out'
    lastlayer.dC_dbias    += lastlayer.dC_dlinear

    # iterate through previous layer partials (i.e. deltas)
    # note arrays are indexed at 1, not 0
    for i = 1:(length(net.layers) - 1)
        layer     = net.layers[end - i]      # layer "l"
        nextlayer = net.layers[end - i + 1]  # nextlayer "l + 1"

        if (i + 1 < length(net.layers))
            prevout = net.layers[end - i - 1].activated_out
        elseif (i + 1 == length(net.layers))
            prevout = x
        else
            throw(DomainError(i, "counter i is out of bounds"))
        end

        layer.dC_dlinear   = ( nextlayer.weights' * nextlayer.dC_dlinear ) .* layer.dactivation(layer.linear_out)
        layer.dC_dweights += layer.dC_dlinear * prevout'
        layer.dC_dbias    += layer.dC_dlinear 
    end
end

function backpropagate!(net::Network, x::AbstractMatrix, truth::AbstractMatrix)

    # use chain rule to calculate partial derivatives     
    calcpartials!(net, x, truth)
end

function feedforward!(net::Network, input::AbstractMatrix)::AbstractMatrix
    nlayers = length(net.layers)

    lastoutput = input 
    for i = 1:nlayers
        layer = net.layers[i]

        # z = Ax + b 
        layer.linear_out = layer.weights * lastoutput + layer.bias

        # output = activation(Ax + b)
        layer.activated_out = layer.activation(layer.linear_out)

        # update for the next input
        lastoutput = layer.activated_out 
    end
    return lastoutput
end
end