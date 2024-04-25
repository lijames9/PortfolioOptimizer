function  [mu, Q] = RidgeRegression(B, returns, factRet)
    
    % FUNCTION DESCRIPTION: RidgeRegression will take in the datasets of
    % assets and factor returns and factor coefficients. It will return the parameter
    % estimations of expected return and the covariance matrix.
    %
    % FUNCTION INPUTS
    % B: a vector of coefficients calculated using penalized Ridge Regression
    % returns: a matrix of size T x n containing the historical returns of n assets over T periods
    % factRet: a matrix of size T x p containing factor returns over T
    % periods.
    %
    % FUNCTION OUTPUTS:
    % mu: a column vector of expected asset returns
    % Q: a symmetric matrix of covariances between each of the 20 assets.
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet); 

    % Data matrix
    X = [ones(T,1) factRet];
    
    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    
    D        = diag(sigma_ep);
    
    % Factor expected returns and covariance matrix
    f_bar = mean(factRet,1)';
    F     = cov(factRet);
    
    % Calculate the asset expected returns and covariance matrix
    mu = a + V' * f_bar;
    Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
    %----------------------------------------------------------------------
end






