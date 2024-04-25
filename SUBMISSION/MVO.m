function  x = MVO(mu, Q)
    % --------------------- SHARPE RATIO MAXIMIZATION FUNCTION ---------------------
    %
    % FUNCTION DESCRIPTION: This function uses the provided covariance
    % matrix Q and the asset returns matrix mu to find the ideal maximized
    % sharpe ratio.
    %
    % FUNCTION INPUTS
    % 'mu': a column vector of estimated expected returns for each asset
    % 'Q': a covariance matrix of asset returns
    %
    % FUNCTION OUTPUTS: 
    % 'x' is the resulting matrix from the quadratic programming of the
    % ideal weights of all the assets for the maximum sharpe ratio
    %----------------------------------------------------------------------
    
    % Find the total number of assets
    n = size(Q,1); 

    % Set the target as the average expected return of all assets
    targetRet = mean(mu);

    % Disallow short sales
    lb = zeros(n+1,1);

    % Add the expected return constraint
    A = [];
    b = [];

    % Constrain to create the following constraints:
    % mu'*y = 1
    % -kappa + y = 0 -> -1+x=0
    Aeq = [[0 mu']; [-1 ones(1, n)]];
    beq = [1; 0];

    % q is the matrix Q restructured to fit the z where z = [kappa; y^T]
    q = [[0 zeros(1, n)]; [zeros(n, 1) Q]];

    % Set the quadprog options 
    options = optimoptions( 'quadprog', 'TolFun', 1e-9, 'Display','off');

    % Optimal asset weights
    % We use z to account for kappa and y: so z optimizes kappa and y
    z = quadprog( 2 * q, [], A, b, Aeq, beq, lb, [], [], options);

    % We index the z to get kappa and y
    kappa = z(1);
    y = z(2:end);
    % Using the definition of y=kappa*x, we solve for x
    x = y./kappa;
    %----------------------------------------------------------------------

end