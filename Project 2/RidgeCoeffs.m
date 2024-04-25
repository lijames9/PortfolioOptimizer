function B = RidgeCoeffs(returns, factRet, lambda)

    % Number of observations and factors
    [T, p] = size(factRet);

    % Data matrix
    X = [ones(T,1) factRet];
    
    % Calculation coefficients with ridge regression
    B = (X' * X + lambda*(eye(p+1))) \ X' * returns;
end
