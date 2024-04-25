function x = Project1_Function(periodReturns, periodFactRet, x0)

    % Use this function to implement your algorithmic asset management
    % strategy. You can modify this function, but you must keep the inputs
    % and outputs consistent.
    %
    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights)
    % OUTPUTS: x (optimal portfolio)
    %
    % An example of an MVO implementation with OLS regression is given
    % below. Please be sure to include comments in your code.
    %
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------

    % Subset the data to estimate the parameters using the first 5 years of
    % data.
    returns = periodReturns(end-59:end,:);
    factRet = periodFactRet(end-59:end,:);
    
    % Determine the best lambda for each rebalancing period.
    lambda = best_lambda(returns, factRet, x0);

    % Obtain the best factor coefficients using the best lambda
    B = RidgeCoeffs(returns, factRet, lambda);

    % Obtain the best parameter estimates using the best factor
    % coefficients.
    [mu, Q] = RidgeRegression(B, returns, factRet);
    
    % Use the optimizer to find the best asset weights for the portfolio
    x = MVO(mu, Q, returns, factRet, x0);

    %----------------------------------------------------------------------
end