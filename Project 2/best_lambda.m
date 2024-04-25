function lambda = best_lambda(returns, factRet, x0)

% FUNCTION DESCRIPTION:
% This code performs k-fold cross-validation where k is the number of folds
% that the "Returns" dataset is being split and tested upon. It is tested 
% using ridge regression by comparing the training regression coefficients
% with the testing results set.

% FUNCTION INPUTS: 
% 
% returns: a dataset in the form of a matrix that represents the returns
% of assets.
% factRet: a dataset in the form of a matrix that represents the factor
% returns of assets.

% FUNCTION OUTPUTS:
% lambda: the penalty parameter we will implement in our final ridge
% regression that minimizes the error.

% Values of lambdas to use:
    lambda_vals = logspace(-6, 2, 100);
    best_lambda = 0;
    best_error = 100000000;
    lambda_errors = zeros(size(lambda_vals,1));
    format long
% Perform k-fold cross-validation with 10 folds:
    for i = 1:length(lambda_vals)
        lambda_error = 0;
        % Want to take the kth fold
        for j = 1:10
            % add a +1 in the first index because MATLAB starts at 1
            ret_test_idx = (j-1)*(floor(size(returns, 1)/10))+1:j*(floor(size(returns,1)/10));
            ret_test_set = returns(ret_test_idx, :);
            ret_train_idx = setdiff(1:size(returns,1), ret_test_idx);
            ret_train_set = returns(ret_train_idx, :);

            % also split factor data into folds
            factor_test_idx = (j-1)*(floor(size(factRet, 1)/10))+1:j*(floor(size(factRet,1)/10));
            factor_test_set = factRet(factor_test_idx, :);
            factor_train_idx = setdiff(1:size(factRet,1), factor_test_idx);
            factor_train_set = factRet(factor_train_idx, :);

            % Run ridge regression with our lambda value array
            B_train = RidgeCoeffs(ret_train_set, factor_train_set, lambda_vals(i));
            [mu_train, Q_train] = RidgeRegression(B_train, ret_train_set, factor_train_set);
            x_train = MVO(mu_train, Q_train, returns, factRet, x0); 

            % Determine the error of the training set
            temp_error = evaluate_portfolio_err(x_train, ret_test_set);
            lambda_error = lambda_error + temp_error;
        end
        lambda_errors(i) = lambda_error;
        % iteratively replace the worst error
        if (best_error > lambda_error)
            best_lambda = lambda_vals(i);
            best_error = lambda_error;
        end
    end

    fig_lambda = figure(3);
    semilogx(lambda_vals, lambda_errors)
    xlabel('Log-Lambda')
    ylabel('Lambda Error')
    title('Sharpe Ratio Loss by Lambda')
    lambda = best_lambda;
end