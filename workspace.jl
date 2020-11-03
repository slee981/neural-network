# ----------------------------------------------------------------------------
# NN 

mutable struct Layer
    weights::Array{Float64,2};
    bias::Array{Float64,2};

    # each layer can have its own activation function
    activation::Function;
    dactivation::Function;

    # cache 
    # 1- the last linear transformation z = Ax + b
    # 2- activated output = activation(z) 
    linear_out::Array{Float64,2};
    activated_out::Array{Float64,2};

    # keep track of partial derivative error for each batch 
    dC_dlinear::Array{Float64,2}; 
    dC_dweights::Array{Float64,2}; 
    dC_dbias::Array{Float64,2}; 

    function Layer(insize::Int64, outsize::Int64, activation::Function, dactivation::Function)
        weights = randn(outsize, insize);
        bias = randn(outsize, 1);

        linear_out = zeros(outsize, 1);
        activated_out = zeros(outsize, 1);

        dC_dlinear = zeros(outsize, 1);
        dC_dweights = zeros(outsize, insize);
        dC_dbias = zeros(outsize, 1);
        new(weights, bias, activation, dactivation, linear_out, activated_out, dC_dlinear, dC_dweights, dC_dbias);
    end
end

mutable struct Network 
    inputdim::Int16;
    layers::Array{Layer,1};
    cost::Function;
    dcost::Function;

    function Network(;inputdim, cost, dcost)  
        new(inputdim, Array{Layer}[], cost, dcost);
    end
end

function addlayer!(net::Network, outsize::Int64, activation::Function, dactivation::Function)
    if length(net.layers) == 0
        insize = net.inputdim;
    else 
        lastlayer = net.layers[end];
        insize = size(lastlayer.weights, 1);     # num rows in previous output
    end
    layer = Layer(Int64(insize), outsize, activation, dactivation);
    push!(net.layers, layer);
end

function fit!(net::Network, x::Array{Float64,2}, y::Array{Float64,2}; batchsize=1, epochs=2, learningrate=0.5)
    # input
    #     ~ net : neural network to fit 
    #     ~ x   : input args with variables in columns, observation in rows   
    #     ~ y   : the "true" values
    sgd!(net, x, y, batchsize, epochs, learningrate);
end

function sgd!(net::Network, x::Array{Float64,2}, y::Array{Float64,2}, batchsize::Int64, epochs::Int64, learningrate::Float64)
    # stochastic gradient descent (sgd)

    lossvals = Vector{Float64}();
    for epoch = 1:epochs

        # shuffle rows of matrix  
        nobs, nvars = size(x);
        shuffledrows = shuffle(1:nobs);
        x = x[shuffledrows, :];
        y = y[shuffledrows, :];

        # create mini batches and loop through each batch 
        # note: julia is NOT zero indexed 
        #       i.e. x[1] is the first element
        for batchend = batchsize:batchsize:nobs
            batchstart = batchend - batchsize + 1;

            # get sample of rows i.e. observations 
            # and transpose into columns for feedforward
            xbatch = x[batchstart:batchend, :]';
            ybatch = y[batchstart:batchend, :]';

            # average losses for each sample in batch
            losses = zeros(size(ybatch));
            for icol in 1:batchsize
                xi    = xbatch[:, icol:icol];
                ytrue = ybatch[:, icol:icol];

                # feedforward
                out = feedforward!(net, xi);
                @info "output: " icol
                println(xi)
                println("guess: ", out)
                println("true : ", ytrue)

                # calculate loss 
                iloss = net.cost(out, ytrue);

                # store loss for later average
                losses[:, icol:icol] = iloss;

                # calculate partial derivatives of each weight and bias
                # i.e. backpropagate
                backpropagate!(net, xi, ytrue);
            end

            if batchend % 100 == 0 

                # take average loss across the rows 
                # results: column vector of average losses for current weights
                # meanloss = mean(losses, dims=2);
                meanloss = mean(losses);
                push!(lossvals, meanloss);
            
                plotloss(lossvals);
            end

            # update weights and bias 
            update!(net, batchsize, learningrate);
        end
    end
    plotloss(lossvals, enter2close=true)
end

function plotloss(lossvalues::Array{Float64,1}; enter2close=false)
    p = plot(lossvalues, legend = false);
    gui(p);
    if enter2close
        println("\nPress 'Enter' to close the plot:");
        x = readline();
    end
end

function update!(net::Network, batchsize::Int64, learningrate::Float64)

    # update weights in each layer based on the error terms dC_dweights, dC_dbias
    for i = 1:length(net.layers)
        layer = net.layers[i];

        layer.weights -= learningrate / batchsize * layer.dC_dweights;
        layer.bias    -= learningrate / batchsize * layer.dC_dbias;

        # reset the error terms for next batch 
        nrows, ncols = size(layer.weights);

        layer.dC_dlinear  = zeros(nrows, 1);
        layer.dC_dweights = zeros(nrows, ncols);
        layer.dC_dbias    = zeros(nrows, 1);
    end
end

function calcpartials!(net::Network, x::Array{Float64,2}, truth::Array{Float64,2})

    # calculate last layer partials (i.e. deltas)
    # wrt the linear transformation z = Ax + b
    # thus, 
    # dc/dzL = dloss(output) * dactivation(z)
    lastlayer = net.layers[end];

    lastlayer.dC_dlinear   = net.dcost(lastlayer.activated_out, truth) .* lastlayer.dactivation(lastlayer.linear_out);
    lastlayer.dC_dweights += lastlayer.dC_dlinear * net.layers[end - 1].activated_out';
    lastlayer.dC_dbias    += lastlayer.dC_dlinear;

    # iterate through previous layer partials (i.e. deltas)
    # note arrays are indexed at 1, not 0
    for i = 1:(length(net.layers) - 1)
        layer     = net.layers[end - i];      # layer "l"
        nextlayer = net.layers[end - i + 1];  # nextlayer "l + 1"

        if (i + 1 < length(net.layers))
            prevout = net.layers[end - i - 1].activated_out;
        elseif (i + 1 == length(net.layers))
            prevout = x;
        else
            throw(DomainError(i, "counter i is out of bounds"))
        end

        layer.dC_dlinear   = ( nextlayer.weights' * nextlayer.dC_dlinear ) .* layer.dactivation(layer.linear_out);
        layer.dC_dweights += layer.dC_dlinear * prevout';
        layer.dC_dbias    += layer.dC_dlinear; 
    end
end

function backpropagate!(net::Network, x::Array{Float64,2}, truth::Array{Float64,2})

    # use chain rule to calculate partial derivatives     
    calcpartials!(net, x, truth);
end

function feedforward!(net::Network, input::Array{Float64,2})::Array{Float64,2}
    nlayers = length(net.layers)

    lastoutput = input; 
    for i = 1:nlayers
        layer = net.layers[i]

        # z = Ax + b 
        layer.linear_out = layer.weights * lastoutput + layer.bias;

        # output = activation(Ax + b)
        layer.activated_out = layer.activation(layer.linear_out);

        # update for the next input
        lastoutput = layer.activated_out; 
    end
    return lastoutput
end

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

# desired interface 
using LinearAlgebra, Random, Statistics, Logging, Plots

# data 
#
# zero and one outcome var 
n = 10000; 

c  = ones(n, 1);
x1 = [i / 1000 for i = 1:n];
x2 = round.(randn(n, 1) * 10);

u = randn(n, 1);

# dgp 
# 
# yhat = 4 + 2 * x1 - 3 * x2
# y    = yhat > mean(yhat) 
yhat = 4 * c + 2 * x1 - 10 * x2 + u;
y    = yhat .> 0;

Y = Float64.(y);
X = hcat(c, x1, x2);

# setup network 
net = Network(inputdim=3, cost=loss, dcost=dloss);
# addlayer!(net, 16, relu, drelu); 
# addlayer!(net, 32, relu, drelu); 
addlayer!(net, 8, sigmoid, dsigmoid); 
addlayer!(net, 16, sigmoid, dsigmoid); 
addlayer!(net, 4, sigmoid, dsigmoid); 
addlayer!(net, 1, sigmoid, dsigmoid); 

# fit network - stochastic gradient descent 
#
# 1- get minibatches 
# 2- for each batch in minibatch 
#    >> feedforward 
#    >> backpropagate 
fit!(net, X, Y, batchsize=8, epochs=8, learningrate=0.05)

# check predictions
# predict(net, X)
