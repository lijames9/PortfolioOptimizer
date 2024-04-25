function  x = MVO(mu, Q, returns, factRet, x0)
    
    % Use this function to construct an example of a MVO portfolio.
    %
    % An example of an MVO implementation is given below. You can use this
    % version of MVO if you like, but feel free to modify this code as much
    % as you need to. You can also change the inputs and outputs to suit
    % your needs. 
    
    % You may use quadprog, Gurobi, or any other optimizer you are familiar
    % with. Just be sure to include comments in your code.

    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    % Number of assets and number of historical scenarios
    [S, n] = size(returns);

    % Set the target as the average expected return of all assets
    targetRet = mean(mu);

    % Retrieve the optimal portfolio weights
    %----------------------------------------------------------------------
    % Disallow short sales
    lb = zeros(n+1,1);

    % Add the expected return constraint
    A = [[(x0 - 0.1*ones(n,1)), -1*eye(n)]; [(-1*x0 - 0.1*ones(n,1)), eye(n)]];
    b = [zeros(n, 1); zeros(n, 1)];

    %constrain weights to sum to 1
    Aeq = [[0 mu']; [-1 ones(1, n)]];
    beq = [1; 0];
    q = [[0 zeros(1, n)]; [zeros(n, 1) Q]];

    % Set the quadprog options 
    options = optimoptions( 'quadprog', 'TolFun', 1e-9, 'Display','off');

    % Optimal asset weights
    [z,fval,exitflag,output,lambda] = quadprog( 2 * q, 0.0001*[], A, b, Aeq, beq, lb, [], [], options)
    kappa = z(1);
    oi = z(2:end);
    x = oi./kappa;
end