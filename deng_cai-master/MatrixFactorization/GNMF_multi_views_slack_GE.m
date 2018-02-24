function [U_final, V_final, Vx_final,nIter_final, objhistory_final,coefficient] = GNMF_multi_views_slack_GE(X, k, Wr, Wi, options, U, V)
% Graph regularized Non-negative Matrix Factorization (GNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%               options.alpha ... the regularization parameter. 
%                                 [default: 100]
%                                 alpha = 0, GNMF boils down to the ordinary NMF. 
%                                 
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
% Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
% (ICDM'08), Pisa, Italy, Dec. 2008. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  
%
%
%   version 2.0 --April/2009 
%   version 1.0 --April/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
numOfviews = size(X,1);
for i=1:numOfviews
    if min(min(X{i})) < 0
        error('Input should be nonnegative!');
    end
end

if ~isfield(options,'error')
    options.error = 1e-5;
end

if ~isfield(options,'coefficient')
    options.coefficient = (ones(numOfviews,1)/numOfviews);
end

if ~isfield(options, 'maxIter')
    options.maxIter = [];
end

if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end

if ~isfield(options,'minIter')
    options.minIter = 30;
end

if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end

if ~isfield(options,'alpha')
    options.alpha = 100;
end

if ~isfield(options,'lambda')
    options.lambda = 0;
end

if ~isfield(options,'beta')
    options.beta = options.alpha;
end

if ~isfield(options,'delta')
    options.delta = 1;
end

if ~isfield(options,'gamma')
    options.gamma = 10;
end

nSmp = size(X,2);

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    options.alpha = options.alpha*nSmp;    
end

if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    feaSum = full(sum(X,2));
    D_half = X'*feaSum;
    X = X*spdiags(D_half.^-.5,0,nSmp,nSmp);
end

if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end

if ~exist('U','var')
    U = [];
    V = [];
end

switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, V_final,Vx_final, nIter_final, objhistory_final,coefficient] = GNMF_Multi_GE_slack_Hs_multi_views(X, k, Wr, Wi, options, U, V);
    otherwise
        error('optimization method does not exist!');
end


    
        