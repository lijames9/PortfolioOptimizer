function  [mu, Q] = RidgeRegression(B, returns, factRet)
    
    % Use this function to perform a basic OLS regression with all factors. 
    % You can modify this function (inputs, outputs and code) as much as
    % you need to.
 
    % *************** WRITE YOUR CODE HERE ***************
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
    
    % Regression coefficients
    % B = RidgeCoeffs(X, p, returns, lambda);
    %split = .5;

    % NEW CODE 
%     [~, Z] = size(returns);
%     training_set = returns(:, split*Z);
%     test_set = returns(:, (split*Z + 1):end);
%     lambdavec = [10^-4, 10^-3, 10^-2, 10^-1, 1];
% 
%     min_error = 1;
%     best_lambda = 0;
%     for i = 1:5
%         Btmin = ridge(training_set, X, lambdavec(i));
%         error = test_set - X*Btmin;
%         ERROR_RETURN = norm(error)^2;
%         if ERROR_RETURN < min_error
%            best_lambda = lambdavec(i);
%            min_error = ERROR_RETURN;
%         end
%     end
    
  
    %B = (X' * X + (1)*eye(p+1)) \ X' * returns;
    %inv(X' * X - lambdavec(i)*eye(size(X))) \ X' * training_set 
%     test_set - X*Btmin(:, 1);
%     dates x (assets - splitassets) =  x m+1 m+1 x 5 
    %END OF NEW CODE
end






