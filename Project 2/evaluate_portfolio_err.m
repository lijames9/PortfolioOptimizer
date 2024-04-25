function Sharpe_Ratio_Error = evaluate_portfolio_err(x, returns)

    % FUNCTION DESCRIPTION:
    % This code represents the loss function being used for the
    % cross-validation, where it will determine the error between the
    % Sharpe Ratio obtained using the training dataset and the testing
    % dataset.
    
    % FUNCTION INPUTS: 
    % x: a vector of asset weights.
    % returns: a dataset in the form of a matrix that represents the returns
    % of assets.
    
    % FUNCTION OUTPUTS:
    % Sharpe_Ratio_Error: the "score" to evaluate the quality of each penalty
    % term being tested.

   portfExRets = returns*x;

   Sharpe_Ratio_Error = -1*(geomean(portfExRets + 1) - 1) / std(portfExRets);
end