function B = RidgeCoeffs(returns, factRet, lambda)

    % FUNCTION DESCRIPTION: RidgeCoefficients will take in the datasets of
    % assets and factor returns and return a set of factor coefficients
    % using penalized Ridge Regression. The penalty term is a given
    % parameter determined with k-fold CV.
    %
    % FUNCTION INPUTS
    % returns: a matrix of size T x n containing the historical returns of n assets over T periods
    % factRet: a matrix of size T x p containing factor returns over T
    % periods.
    % lambda: a penalty term that gives the least Sharpe Ratio estimation
    % error.
    %
    % FUNCTION OUTPUTS:
    % B: a matrix of factor coefficients at each time period T

    % Number of observations and factors
    [T, p] = size(factRet);

    % Data matrix
    X = [ones(T,1) factRet];
    
    % Regression coefficients
    penalty = lambda*(eye(p+1));
    penalty(1,1) = 0;
    B = (X' * X + penalty) \ X' * returns;
end
