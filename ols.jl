using LinearAlgebra, Random, Statistics, Logging


# ----------------------------------------------------------------------------
# STORAGE 

# data 
#
# zero and one outcome var 

n = 10; 

c  = ones(n, 1);
x1 = 1:n;
x2 = round.(randn(n, 1) * 10);

u = randn(10, 1);

# dgp 
# 
# yhat = 4 + 2 * x1 - 3 * x2
# y    = yhat > 0 
Y = ( 4 * c + 2 * x1 - 3 * x2 + u) .> 0;
X = hcat(c, x1, x2);

# ----------------------------------------------------------------------------
# OLS 

# point estimates 
beta = inv(X' * X) * X' * Y;

# residuals 
resid = vec(Y - (X * beta));
omega = diagm(resid);                           # diagonal matrix of residuals 

# degrees of freedom 
ncoeffs = size(X)[2];
df = n - ncoeffs;

# variance-covariance matrix of the error terms
var   = inv(X' * X) * (X' * omega * X) * inv(X' * X) / df; 

# standard error of the coefficients 
se    = sqrt.(diag(var));    

