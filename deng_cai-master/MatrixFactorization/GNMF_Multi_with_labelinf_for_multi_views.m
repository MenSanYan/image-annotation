function [U_final, V_final, nIter_final, objhistory_final] = GNMF_Multi_with_labelinf_for_multi_views(X, k, Wr, Wi, options, U, V)
% Graph regularized Non-negative Matrix Factorization (GNMF) with
%          multiplicative update
% semi-supervised for the label information from the train set on V.
% where
%   X is a cell array, and the element X{i} is feature matrix from ith views.

% Notation:
% x ... (mFea x nSmp) data matrix from the train and set.
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
% Wr is the label information from the train set for the label similarity.
% Wi is the label information from the train set for the label unsimilarity.
% U is a cell array, and the element U{i} is basis matrix for X{i}.
% V is the coefficient matrix.

% options ... Structure holding all settings
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
%   version 2.1 --Dec./2011 
%   version 2.0 --April/2009 
%   version 1.0 --April/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
coefficient = options.coefficient;
differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
lambda = options.lambda;
epsion = options.epsion;
% nRepeat = 1;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;
beta = options.beta;
gamma = options.gamma;
Norm = 2;
NormV = 1;

numOfviews = size(X,1);
nSmp = size(X{1},2);

[numOftrain,~]=size(Wr);

if alpha > 0
    Wr = alpha*Wr;
    DColr = full(sum(Wr,2));
    Dr = spdiags(DColr,0,numOftrain,numOftrain);
    Lr = Dr - Wr;
    if isfield(options,'NormW') && options.NormW
        D_mhalfr = spdiags(DColr.^-.5,0,numOftrain,numOftrain) ;
        Lr = D_mhalfr*Lr*D_mhalfr;
    end
else
    Lr = [];
end

if beta > 0
    Wi = beta*Wi;
    DColi = full(sum(Wi,2));
    Di = spdiags(DColi,0,numOftrain,numOftrain);
    Li = Di - Wi;
    if isfield(options,'NormW') && options.NormW
        D_mhalfi = spdiags(DColi.^-.5,0,numOftrain,numOftrain) ;
        Li = D_mhalfi*Li*D_mhalfi;
    end
else
    Li = [];
end

if epsion > 0
    Wm = options.Wm;
    Wm = epsion*Wm;
    DColr = full(sum(Wm,2));
    Dm = spdiags(DColr,0,nSmp,nSmp);
    Lm = Dm - Wm;
    if isfield(options,'NormW') && options.NormW
        D_mhalfr = spdiags(DColr.^-.5,0,nSmp,nSmp) ;
        Lm = D_mhalfr*Lm*D_mhalfr;
    end
else
    Lm = [];
end
options.Lm = Lm;


selectInit = 1;
if isempty(U)
    U = cell(numOfviews,1);
    for i=1:numOfviews
        [mFea,~]=size(X{i});
        U{i} =  abs(rand(mFea,k));
    end
    V =  abs(rand(nSmp,k));
else
    nRepeat = 1;
end

[U,V] = NormalizeUV(U, V, NormV, Norm);

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Lr, Li, options, coefficient);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Lr, Li, options, coefficient);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

Xl = cell(numOfviews);
Xu = cell(numOfviews);
for i=1:numOfviews
    Xl{i} = X{i}(:,1:numOftrain);
    Xu{i} = X{i}(:,numOftrain+1:nSmp);
end

tryNo = 0;
nIter = 0;
while tryNo < nRepeat 
    disp(['tryNo:',num2str(tryNo)]);
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
% ii = 10;
%     while(ii > 0)
%         ii = ii-1;
        if selectInit == 0
            disp(['maxErr:',num2str(maxErr),',differror:',num2str(differror)]);
        end
        % ===================== update V ========================
        % here we will update the V separately for labeled and the
        % unlabeld.
        Vl = V(1:numOftrain,:);
        Vu = V(numOftrain+1:nSmp,:);
        XlUs = 0;
        VlUUs = 0;
        XuUs = 0;
        VuUUs = 0;
        for i=1:numOfviews
            UU = U{i}'*U{i};  % mk^2
            XlUs = XlUs+coefficient(i)*Xl{i}'*U{i};  % mnk or pk (p<<mn)
            VlUUs = VlUUs+coefficient(i)*Vl*UU; % nk^2
            XuUs = XuUs+coefficient(i)*Xu{i}'*U{i};  % mnk or pk (p<<mn)
            VuUUs = VuUUs+coefficient(i)*Vu*UU; % nk^2
        end
        
        %here is the update rule for the labeled sample.
        %1,push the semantic relevant vector in the new feature space.
        if alpha > 0
            WrVl = Wr*Vl;
            DrVl = Dr*Vl;
            
            XlUs = XlUs + WrVl;
            VlUUs = VlUUs + DrVl;
        end
        
        %2,pull the semantic irrelevant vector in the new feature space.
        if beta > 0
            WiVl = Wi*Vl;
            DiVl = Di*Vl;
%             traceVLiV = sum(sum());
            %here is the normal one.
            XlUs = XlUs + DiVl;
            VlUUs = VlUUs + WiVl;
            %here is the minus version
%             XlUs = XlUs - WiVl;
%             VlUUs = VlUUs - DiVl;
            %here is the new style
            
        end
        
        if gamma > 0
            VlUUs = VlUUs + gamma*Vl;
            VuUUs = VuUUs + gamma*Vu;
        end
        
%         if epsion > 0
%             WmVl = Wm*Vl;
%             DmVl = Dm*Vl;
%             
%             XlUs = XlUs + WmVl;
%             VlUUs = VlUUs + DmVl;
%             
%             
%             WmVu = Wm*Vu;
%             DmVu = Dm*Vu;
%             
%             XuUs = XuUs + WmVu;
%             VuUUs = VuUUs + DmVu;
%         end
        %update the Vl under the constrait of the semantic embedding.
        Vl = Vl.*(XlUs./max(VlUUs,1e-10));
        
        %there is no label information for the test set feature.
        Vu = Vu.*(XuUs./max(VuUUs,1e-10));
        
        V = [Vl;Vu];
        % ===================== update U ========================
        for i=1:numOfviews
            XV = X{i}*V;   % mnk or pk (p<<mn)
            VV = V'*V;  % nk^2
            UVV = U{i}*VV; % mk^2
            UVV = UVV+lambda*U{i}; %sparse the U
            
            U{i} = U{i}.*(XV./max(UVV,1e-10)); % 3mk
        end
        
        nIter = nIter + 1;
%                 objhistory = CalculateObj(X, U, V, Lr, Li, options, coefficient)
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Lr, Li, options, coefficient);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Lr, Li, options, coefficient);
                    objhistory = [objhistory newobj] ;%#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Lr, Li, options, coefficient);
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
%             U = abs(rand(mFea,k));
            
            U = cell(numOfviews,1);
            for i=1:numOfviews
                [mFea,~]=size(X{i});
                U{i} =  abs(rand(mFea,k));
            end
            V = abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(U, V, NormV, Norm);
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);

end
%==========================================================================

function [obj, dV] = CalculateObj(Xs, Us, V, Lr, Li, options, coefficient, deltaVU, dVordU)
    MAXARRAY = 1000*1024*1024/8; % 1G. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    numOfviews = size(Xs,1);
    obj_NMF = 0;
    for j=1:numOfviews
        X = Xs{j};
        U = Us{j};
        dV = [];
        nSmp = size(X,2);
        mn = numel(X);
        nBlock = ceil(mn/MAXARRAY);

        if mn < MAXARRAY
            dX = U*V'-X;
            obj_NMF = obj_NMF+coefficient(j)*sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV = dX'*U + L*V;
                else
                    dV = dX*V;
                end
            end
        else
            if deltaVU
                if dVordU
                    dV = zeros(size(V));
                else
                    dV = zeros(size(U));
                end
            end
            PatchSize = ceil(nSmp/nBlock);
            for i = 1:nBlock
                if i*PatchSize > nSmp
                    smpIdx = (i-1)*PatchSize+1:nSmp;
                else
                    smpIdx = (i-1)*PatchSize+1:i*PatchSize;
                end
                dX = U*V(smpIdx,:)'-X(:,smpIdx);
                obj_NMF = obj_NMF + coefficient(j)*sum(sum(dX.^2));
                if deltaVU
                    if dVordU
                        dV(smpIdx,:) = dX'*U;
                    else
                        dV = dU+dX*V(smpIdx,:);
                    end
                end
            end
            if deltaVU
                if dVordU
                    dV = dV + L*V;
                end
            end
        end
    end
    
    %the lapace obj for the push.
    
    if isempty(Lr)
        [numOftrain,~]=size(Li);
    else
        [numOftrain,~]=size(Lr);
    end
    Vl = V(1:numOftrain,:);
    
    %the laplace obj for the pull.
    if isempty(Lr)
        obj_Lap_r = 0;
    else
        obj_Lap_r = sum(sum((Vl'*Lr).*Vl'));%trace
    end
    
    %the laplace obj for the push.
    if isempty(Li)
        obj_Lap_i = 0;
    else
        obj_Lap_i = sum(sum((Vl'*Li).*Vl'));%trace
    end
    
    %sparse for the U
    if options.lambda == 0
        obj_sparse_U = 0;
    else
        obj_sparse_U = 0;
        for j=1:numOfviews
            obj_sparse_U = obj_sparse_U+sum(sum(Us{j}.^2));
        end
    end
    obj_sparse_U = options.lambda*obj_sparse_U;
    
    %sparse for the V
    if options.gamma == 0
        obj_sparse_V = 0;
    else
        obj_sparse_V = sum(sum(V.^2));
    end
    obj_sparse_V = options.gamma*obj_sparse_V;
    
    %manifold
    if options.epsion == 0
        obj_Lap_m = 0;
    else
        Lm = options.Lm;
        obj_Lap_m = sum(sum((V'*Lm).*V'));%trace
    end
    
    
    obj = obj_NMF+obj_Lap_r-obj_Lap_i+obj_sparse_U+obj_sparse_V+obj_Lap_m;
    
end



function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U{1},2);
    numOfviews = size(U,1);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            for i=1:numOfviews
                U{i} = U{i}*spdiags(norms,0,K,K);
            end
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end
end
        