function [Wx, Wy, SparWx, SparWy, corr_list, time] = CCABI(X, Y, options) 

% CCA: (Sparse) Canonical Correlation Analysis
%
% Usage:
%     [Wx, Wy, SparWx, SparWy,corr_list,time] = CCA(X, Y, options)
%     [Wx, Wy, SparWx, SparWy,corr_list,time] = CCA(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        Y       - Data matrix Y. Each column of Y is a data point.
%       
%        options.flag             = 1: Projections Wx and Wy and sparse projections SparWx and SparWy are required;
%                                 = 0: Only return projections Wx and Wy.
%                                      The defult value is 1
%        options.DeltaL           -- A integer which denotes the difference (L - m). The default value is 0.
%        options.L                -- The number of columns in Wx and Wy.
%        options.mu               -- Parameter mu in the Linearized Bregman method. The default value is mu = 5.
%        options.delta            -- Parameter delta in the Linearized Bregman method, delta must satisfy 0 < delta <1.
%                                    The default value is delta = 0.9.
%        options.epsilon1         -- Tolerance for computing SparWx. The default value is 1e-6.
%        options.epsilon2         -- Tolerance for computing SparWy. The default value is 1e-6. 
%
%    Output:
%              W_x: Projection vectors for X. 
%              W_y: Projection vectors for Y. 
%          SparW_x: Sparse projection vectors for X. If the sparse projection for X is not required, an empty matrix is returned.
%          SparW_y: Sparse projection vectors for X. If the sparse projection for X is not required, an empty matrix is returned.
%        corr_list: the list of correlation coefficients in the projected spaces of X and Y. 
%             time: a 2x1 vector storing CPU time with the first componet storing the cpu time of computing nonsparse Wx and Wy and 
%                   the second storing the cpu time of computing (Wx,Wy) and (SparWx,SparWy)
%
%====================================================================================================================================

t = cputime;
% Preprocess the input 
if size(X, 2) ~= size(Y, 2)
    error('The numbers of samples in X and Y are not equal!');
end
if size(X,1)<2 || size(Y,1)<2
    error('Need at least two features in datasets X and Y!')
end

% Extract the input parameters, and set the values of parameters.
if (~exist('options','var'))
   options = [];
end

flag = 0;
if isfield(options, 'flag')
    flag = options.flag;
end

DeltaL = 0;
if isfield(options, 'DeltaL')
    DeltaL = options.DeltaL;
end

mu = 5;
if isfield(options, 'mu')
    mu = options.mu;
end

delta = 0.9;
if isfield(options, 'delta')
    delta = options.delta;
end

epsilon1 = 1e-6;
if isfield(options, 'epsilon1')
    epsilon1 = options.epsilon1;
end

epsilon2 = 1e-6;
if isfield(options, 'epsilon2')
    epsilon2 = options.epsilon2;
end

% Wx = [];
% Wy = [];
% SparWx = [];
% SparWy = [];

% Compute economic QR of X with column pivoting
[U, R, E1] = qr(X,0);
X_rank = rank(R);
U1 = U(:,1:X_rank);
[Sort_R index_R] = sort(E1);
R1 = R(1:X_rank,index_R);

% Compute economic QR of Y with column pivoting
[V, RR, E2] = qr(Y,0);
Y_rank = rank(RR);
V1 = V(:,1:Y_rank);
[Sort_RR index_RR] = sort(E2);
R2 = RR(1:Y_rank,index_RR);

% Compute economic SVD of R1
[P1, Sigma1, Q1] = svd(R1,'econ');

% Compute economic SVD of R1
[P2, Sigma2, Q2] = svd(R2,'econ');

% Compute SVD of Q1'*Q2
[Pro_P1,Sigma,Pro_P2] = svd(Q1'*Q2);
m = rank(Sigma);  % m is the rank of Q1'*Q2;

% The number of columns in Wx and Wy
if isfield(options,'L')
    L = options.L;
else
    L = m + DeltaL;
end

% Compute the canonical correlations
if L <=m
    corr_list = diag( Sigma(1:L,1:L) );
else
    corr_list = [diag( Sigma(1:m,1:m) );zeros(L-m,1)];
end


if L > min(X_rank,Y_rank)
    error('Your l has exceeded min{rank(X), rank(Y)}!');
else
    % Compute Wx
    d1 = diag(Sigma1);
    d1 = 1./d1;
    Sigma1_inv = diag(d1);
    Wx = U1 * P1 * Sigma1_inv * Pro_P1(:,1:L);
    
    % Compute Wy
    d2 = diag(Sigma2);
    d2 = 1./d2;
    Sigma2_inv = diag(d2);
    Wy = V1 * P2 * Sigma2_inv * Pro_P2(:,1:L);
end
time(1) = cputime - t;
% Compute Sparse Wx and Sparse Wy if flag == 1;

if flag == 1
    % Initial data
    dimen1 = size(X,1);
    dimen2 = size(Y,1);
    
    SparWx = zeros(dimen1,L);
    Vx = zeros(dimen1,L);
    
    SparWy = zeros(dimen2,L);
    Vy = zeros(dimen2,L);    
    
    W = eye(L);
    Ax = (U1 * P1)'; Bx = Sigma1_inv * Pro_P1(:,1:L) * W;
    Ay = (V1 * P2)'; By = Sigma2_inv * Pro_P2(:,1:L) * W;   

    % Linearized Bregman Iteration for SparWx
    error_x = 1;
    Numit_x = 0;
    Tempx_1 = Ax'*Bx; Tempx_2 = U1*U1';
    while error_x > epsilon1 
        Vx = Vx + Tempx_1 - Tempx_2 * SparWx;
        SparWx = delta * sign(Vx) .* max(abs(Vx)-mu,0);
        error_x = norm(Ax * SparWx - Bx,'fro')/norm(Bx,'fro');
        Numit_x = Numit_x + 1;
        if mod(Numit_x,200)==0
            fprintf('error_x = %10.5e \n',error_x);
        end
    end
    
    % Linearized Bregman Iteration for SparWy    
    error_y = 1;    
    Numit_y = 0;    
    Tempy_1 = Ay'*By; Tempy_2 = V1*V1';
    while error_y > epsilon2 
        Vy = Vy + Tempy_1 - Tempy_2 * SparWy;
        SparWy = delta * sign(Vy) .* max(abs(Vy)-mu,0);
        error_y = norm(Ay * SparWy - By,'fro')/norm(By,'fro');
        Numit_y = Numit_y + 1;
        if mod(Numit_y,200)==0
            fprintf('error_y = %10.5e \n',error_y);
        end
    end
end

time(2) = cputime - t;
