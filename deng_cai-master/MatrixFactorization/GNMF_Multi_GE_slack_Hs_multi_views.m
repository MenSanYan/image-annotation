function [U_final, V_final, Vx_final, nIter_final, objhistory_final,coefficient] = GNMF_Multi_GE_slack_Hs_multi_views(X, k, Wr, Wi, options, U, V)
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
delta = options.delta;
gamma = options.gamma;
% nRepeat = 1;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;
beta = options.beta;

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

selectInit = 1;
if isempty(U)
    U = cell(numOfviews,1);
    for i=1:numOfviews
        [mFea,~]=size(X{i});
        U{i} =  abs(rand(mFea,k));
    end
    V = cell(numOfviews,1);
    for i=1:numOfviews
        V{i} =  abs(rand(nSmp,k));
    end
    Vx =  abs(rand(nSmp,k));
else
    nRepeat = 1;
end

[U,V] = NormalizeUV(U, V, Vx, NormV, Norm);

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Vx, Lr, Li, options, coefficient);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Vx, Lr, Li, options, coefficient);
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
        
       
        
        % ===================== update V{i} ========================
        for i=1:numOfviews
            UU = U{i}'*U{i};  % mk^2
            XU = X{i}'*U{i}+delta*coefficient(i)*Vx;  % mnk or pk (p<<mn)
            VUU = V{i}*UU+delta*coefficient(i)*V{i}; % nk^2
            V{i} = V{i}.*(XU./max(VUU,1e-10));
        end
        
        % ===================== update U ========================
        for i=1:numOfviews
            XV = X{i}*V{i};   % mnk or pk (p<<mn)
            VV = V{i}'*V{i};  % nk^2
            UVV = U{i}*VV; % mk^2
            UVV = UVV+lambda*U{i}; %sparse the U
            
            U{i} = U{i}.*(XV./max(UVV,1e-10)); % 3mk
        end
         % ===================== update Vx ========================
        Vs = 0;
        for i=1:numOfviews
            Vs = Vs+coefficient(i)*V{i};
        end
        Vls = delta*Vs;
        Vx = Vls/delta;
%         Vls = delta*Vs(1:numOftrain,:);
%         Vus = Vs(numOftrain+1:nSmp,:);
%         
%         Vxls = delta*sum(coefficient)*eye(numOftrain);%*numOfviews
%         
%         %here is the update rule for the labeled sample.
%         %1,push the semantic relevant vector in the new feature space.
%         if alpha > 0
%             Vxls = Vxls+Lr;
%         end
%         
%         %2,pull the semantic irrelevant vector in the new feature space.
%         if beta > 0
%             Vxls = Vxls-Li;
%         end
%         
%         %here we deal with equally for the weight of the different views.
%         Vxl = Vxls^-1*Vls;
%         Vxu = Vus/(delta*sum(coefficient));
%         Vx = [Vxl;Vxu];
        
       

        % ===================== update coefficient ========================
        dis = zeros(numOfviews,1);
        for i=1:numOfviews
            dV = V{i}-Vx;
            dis(i,1) = sum(sum(dV.^2));
        end
        dis = gamma*dis;
        options.Display = 'off';
        [coefficient]=quadprog(eye(numOfviews),dis,[],[],ones(1,numOfviews),1,zeros(numOfviews,1));
          
        disp([num2str(coefficient(1)),num2str(coefficient(2)),num2str(coefficient(3))]);
        disp([num2str(dis(1)),'  ',num2str(dis(2)),'  ',num2str(dis(3))]);
%         here consider if should be put after the normalized step;

        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Vx, Lr, Li, options, coefficient);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Vx, Lr, Li, options, coefficient);
                    objhistory = [objhistory newobj]; %#ok<AGROW>
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Vx, Lr, Li, options, coefficient);
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
        Vx_final = Vx;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           Vx_final = Vx;
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
            
            V = cell(numOfviews,1);
            for i=1:numOfviews
                V{i} =  abs(rand(nSmp,k));
            end
            Vx =  abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(U, V, Vx, NormV, Norm);
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

[U_final,V_final] = NormalizeUV(U_final, V_final, Vx_final, NormV, Norm);


%==========================================================================

function [obj, dV] = CalculateObj(Xs, Us, Vs, Vx, Lr, Li, options, coefficient, deltaVU, dVordU)
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
        V = Vs{j};
        dV = [];
        nSmp = size(X,2);
        mn = numel(X);
        nBlock = ceil(mn/MAXARRAY);

        if mn < MAXARRAY
            dX = U*V'-X;
            dV = V-Vx;
            obj_NMF = obj_NMF+sum(sum(dX.^2));
            obj_NMF = obj_NMF+coefficient(j)*sum(sum(dV.^2));
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
                dV = V(smpIdx,:)-Vx(smpIdx,:);
                obj_NMF = obj_NMF + sum(sum(dX.^2));
                obj_NMF = obj_NMF + coefficient(j)*sum(sum(dV.^2));
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
    Vxl = Vx(1:numOftrain,:);
    if isempty(Lr)
        obj_Lap_r = 0;
    else
        obj_Lap_r = sum(sum((Vxl'*Lr).*Vxl'));%trace
    end
    
    %the laplace obj for the pull.
    if isempty(Li)
        obj_Lap_i = 0;
    else
        obj_Lap_i = sum(sum((Vxl'*Li).*Vxl'));%trace
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
    
    obj = obj_NMF+obj_Lap_r-obj_Lap_i+obj_sparse_U;
    




function [U, V] = NormalizeUV(U, V, Vx, NormV, Norm)
    K = size(U{1},2);
    numOfviews = size(U,1);
    if Norm == 2
        if NormV
            for i=1:numOfviews
                norms = max(1e-15,sqrt(sum(V{i}.^2,1)))';
                V{i} = V{i}*spdiags(norms.^-1,0,K,K);
                U{i} = U{i}*spdiags(norms,0,K,K);
                
                normsVx = max(1e-15,sqrt(sum(Vx.^2,1)))';
                Vx = Vx*spdiags(normsVx.^-1,0,K,K);
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

        