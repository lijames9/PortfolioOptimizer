function error = evaluate_err(B, returns, factors)
    [T,p] = size(factors); 

    X = [ones(T,1) factors];
    error = 0;
    % Calculating beta estimation error:
    for i = 1:(size(returns, 2))
        error_vec = returns(:,i) - X * B(:,i);
        L2Error = norm(error_vec);
        error = error + L2Error;
    end
end