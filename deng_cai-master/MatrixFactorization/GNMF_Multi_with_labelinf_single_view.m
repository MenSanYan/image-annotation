function [U_final, V_final, nIter_final, objhistory_final] = GNMF_Multi_with_labelinf_single_view(X, k, Wr, Wi, options, U, V)
% Graph regularized Non-negative Matrix Factorization (GNMF) with
%          multiplicative update
% semi-supervised for the label information from the train set on V.
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix from the train and set.
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
% Wr is the label information from the train set for the label similarity.
% Wi is the label information from the train set for the label similarity.

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

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;
beta = options.beta;

Norm = 2;
NormV = 0;

[mFea,nSmp]=size(X);

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

selectInit = 1;
if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end

[U,V] = NormalizeUV(U, V, NormV, Norm);
if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Lr, Li);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Lr, Li);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end


Xl = X(:,1:numOftrain);
Xu = X(:,numOftrain+1:nSmp);

tryNo = 0;
nIter = 0;
while tryNo < nRepeat
    disp(['iter',num2str(tryNo)]);
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
%         if selectInit == 0
%             disp(['maxErr',num2str(maxErr),',differror',num2str(differror)]);
%         end
        % ===================== update V ========================
        % here we will update the V separately for labeled and the
        % unlabeld.
        Vl = V(1:numOftrain,:);
        Vu = V(numOftrain+1:nSmp,:);
        
        UU = U'*U;  % mk^2
        XlU = Xl'*U;  % mnk or pk (p<<mn)
        VlUU = Vl*UU; % nk^2
        
         
        XuU = Xu'*U;  % mnk or pk (p<<mn)
        VuUU = Vu*UU; % nk^2
        
        %here is the update rule for the labeled sample.
        %1,push the semantic relevant vector in the new feature space.
        if alpha > 0
            WrVl = Wr*Vl;
            DrVl = Dr*Vl;
            
            XlU = XlU + WrVl;
            VlUU = VlUU + DrVl;
        end
        
        %2,pull the semantic irrelevant vector in the new feature space.
        if beta > 0
            WiVl = Wi*Vl;
            DiVl = Di*Vl;
            
            XlU = XlU + DiVl;
            VlUU = VlUU + WiVl;
        end
        %update the Vl under the constrait of the semantic embedding.
        Vl = Vl.*(XlU./max(VlUU,1e-10));
        
        %there is no label information for the test set feature.
        Vu = Vu.*(XuU./max(VuUU,1e-10));
        
        V = [Vl;Vu];
        % ===================== update U ========================
        XV = X*V;   % mnk or pk (p<<mn)
        VV = V'*V;  % nk^2
        UVV = U*VV; % mk^2
        
        U = U.*(XV./max(UVV,1e-10)); % 3mk
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Lr, Li);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Lr, Li);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Lr, Li);
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
            U = abs(rand(mFea,k));
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


%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, Lr, Li, deltaVU, dVordU)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    numOftrain = size(Lr,1);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
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
            obj_NMF = obj_NMF + sum(sum(dX.^2));
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
    
    %the lapace obj for the push.
    
    Vl = V(1:numOftrain,:);
    if isempty(Lr)
        obj_Lap_r = 0;
    else
        obj_Lap_r = sum(sum((Vl'*Lr).*Vl'));%trace
    end
    
    %the laplace obj for the pull.
    if isempty(Li)
        obj_Lap_i = 0;
    else
        obj_Lap_i = sum(sum((Vl'*Li).*Vl'));%trace
    end
    obj = obj_NMF+obj_Lap_r-obj_Lap_i;
    




function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
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

        