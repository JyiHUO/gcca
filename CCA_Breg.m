function [Wx, Wy, SWx, SWy, corr, time] = CCA_Breg(X, Y, opts) 

% CCA_ALBI: Sparse Canonical Correlation Analysis Based on Accelerated
% Linearized Bregman Iteration
% 
% Usage:
%     [Wx, Wy, SWx, SWy, corr, time] = CCA_Breg(X, Y, opts)
%     [Wx, Wy, SWx, SWy, corr, time] = CCA_Breg(X, Y)
% 
% Input:
%   X       - Data matrix X. Each column of X is a data point. Centerized
%   Y       - Data matrix Y. Each column of Y is a data point. Centerized
%       
%   opts.deltaL    -- A integer which denotes the difference (L - m). 
%                     The default value is 0.
%   opts.L         -- The number of columns in Wx and Wy.
%   opts.mu_x      -- Parameter mu_x in the Linearized Bregman method. 
%   opts.mu_y      -- Parameter mu_y in the Linearized Bregman method.
%                     The default value is mu = 5.
%   opts.delta     -- Parameter delta in the Linearized Bregman method. delta must satisfy 0 < delta <1.
%                     The default value is 0.9.
%   opts.epsilon1  -- Tolerance for computing SWx. 
%   opts.epsilon2  -- Tolerance for computing SWy. 
%                     The default values are 1e-6. 
%
% Output:
%   W_x   -- Nonsparse projection vectors for X. 
%   W_y   -- Nonsparse Projection vectors for Y. 
%   SW_x  -- Sparse projection vectors for X.
%   SW_y  -- Sparse projection vectors for X. 
%   corr  -- the list of correlation coefficients between training data in the projected spaces of X and Y. 
%   time  -- a 2x1 vector storing CPU time with the first componet storing the cpu time of computing (Wx,Wy) 
%            the second storing the cpu time of computing (SWx,SWy)
%
%
% Reference:
%
%   1. Delin Chu, Li-Zhi Liao, Michael K. Ng, Xiaowei Zhang,
%   Sparse Canonical Correlation Analysis: New Formulation and Algorithm. 
%   IEEE Trans. Pattern Anal. Mach. Intell. 35(12): 3050-3065 (2013)
%
%   2. Xiaowei Zhang,
%   Sparse Dimensionality Reduction Methods: Algorithms and Applications,
%   PhD Thesis, National University of Singapore, July 2013.
%   http://scholarbank.nus.edu.sg/bitstream/handle/10635/48696/ZhangXW.pdf?sequence=1
%
%   Written by Xiaowei Zhang (zxwtroy87 AT gmail.com)
%
%============================================================================================================
t = cputime;

% Extract the input parameters, and set the values of parameters.
if (~exist('opts','var'))
   opts = [];
end

deltaL = 0;
if isfield(opts, 'deltaL')
    deltaL = opts.deltaL;
end

mu_x = 5;
if isfield(opts, 'mu_x')
    mu_x = opts.mu_x;
end

mu_y = 5;
if isfield(opts, 'mu_y')
    mu_y = opts.mu_y;
end

tau = 1;
if isfield(opts, 'tau')
    tau = opts.tau;
end

delta = 0.9;
if isfield(opts, 'delta')
    delta = opts.delta;
end

epsilon1 = 1e-6;
if isfield(opts, 'epsilon1')
    epsilon1 = opts.epsilon1;
end

epsilon2 = 1e-6;
if isfield(opts, 'epsilon2')
    epsilon2 = opts.epsilon2;
end

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
if isfield(opts,'L')
    L = opts.L;
else
    L = m + deltaL;
end

% Compute the canonical correlations
if L <=m
    corr = diag( Sigma(1:L,1:L) );
else
    corr = [diag( Sigma(1:m,1:m) );zeros(L-m,1)];
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

t = cputime;
% Compute Sparse Wx and Sparse Wy
W = eye(L);
Ax = (U1 * P1)'; Bx = Sigma1_inv * Pro_P1(:,1:L) * W;
Ay = (V1 * P2)'; By = Sigma2_inv * Pro_P2(:,1:L) * W; 

% Initial data
dimen1 = size(X,1);
dimen2 = size(Y,1);
    
SWx = zeros(dimen1,L);
Vx_tilde = Ax' * Bx; Vx_old = Vx_tilde;
    
SWy = zeros(dimen2,L);
Vy_tilde = Ay' * By; Vy_old = Vy_tilde;    
  

% Accelerated Linearized Bregman Iteration for SWx
error_x = 1;
Numit_x = 0;

%%% Some note
error_x = 1
epsilon1 = 1e-6
delta = Parameter delta in the Linearized Bregman method. delta must satisfy 0 < delta <1.
mu_x = Parameter mu_x in the Linearized Bregman method. 
tau = 1
sign(vx) = 1 if vx > 0 else 0 if vx == 0 else -1 if vx < 0 
Numit_x = 0
%%%


while error_x > epsilon1 
    SWx = delta * sign(Vx_tilde) .* max(tau*abs(Vx_tilde)-mu_x,0);
    
    Vx_new = Vx_tilde - Ax' * (Ax * SWx - Bx);
    
    alpha = (2*Numit_x+3)/(Numit_x+3);
    Vx_tilde = alpha * Vx_new + (1-alpha)*Vx_old;
    
    Vx_old = Vx_new; % update Vx^{k}
    
    error_x = norm(Ax * SWx - Bx,'fro')/norm(Bx,'fro');
    Numit_x = Numit_x + 1;
    
     if mod(Numit_x,200)==0
         fprintf('error_x = %10.5e \n',error_x);
     end
end
    
% Accelerated Linearized Bregman Iteration for SWy    
error_y = 1;    
Numit_y = 0;    

while error_y > epsilon2 
    SWy = delta * sign(Vy_tilde) .* max(tau*abs(Vy_tilde)-mu_y,0);
    
    Vy_new = Vy_tilde - Ay' * (Ay * SWy - By);
    
    beta = (2*Numit_y+3)/(Numit_y+3);
    Vy_tilde = beta * Vy_new + (1-beta)*Vy_old;
    
    Vy_old = Vy_new; % update Vy^{k}
    
    error_y = norm(Ay * SWy - By,'fro')/norm(By,'fro');
    Numit_y = Numit_y + 1;
    
    if mod(Numit_y,200)==0
        fprintf('error_y = %10.5e \n',error_y);
    end
end

time(2) = cputime - t;
