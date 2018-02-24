function perf = twopassknn(distance_vI_vJ,trainAnnotations,testAnnotations,K1,w,annotLabels)

% distance_vI_vJ =   mxn matrix 
%                    m = no. of all training images 
%                    n = no. of all test images
%                    contains pair-wise distance between all 
%                    training and test images
%		     the distance value should be normalized 
%		     between [0,1]
% trainAnnotations = mxl binary matrix
%                    l = no. of labels
%                    matrix of training images' annotations
%                    (i,j)th entry = 1 means image `i' is tagged with 
%                    label `j'; and 0 otherwise.
% testAnnotations =  nxl binary matrix
%                    matrix of test images' annotations
% K1 =               no. of nearest neighbours per label to be 
%                    considered (tested for 1-5)
% w =                bandwidth parameter (tested for 1-30)
% annotLabels =      no. of labels to be assigned to each test image (usually = 5)



numOfTrainImages = size(trainAnnotations,1);
numOfTestImages = size(testAnnotations,1);
numOfLabels = size(trainAnnotations,2);

labelTableSize = zeros(numOfLabels,1);
for i = 1:numOfLabels
	labelTableSize(i) = sum(trainAnnotations(:,i));
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2pknn begin %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%disp(['K1 = ' num2str(K1) ' , w = ' num2str(w)]);

Score_w_I = zeros(numOfLabels,numOfTestImages);
for i = 1:numOfTestImages
	if( mod(i,500)==0 )
		disp([num2str(i)]);
	end;
	subsetTrain = zeros(numOfLabels*K1,1);
	count = 0;
	for j = 1:numOfLabels
		currTrainSet = find(trainAnnotations(:,j)==1);
		currTrainIndxDist = [currTrainSet distance_vI_vJ(currTrainSet,i)];
		currTrainIndxDist = sortrows(currTrainIndxDist,2);    
		for k = 1:K1
			if( k <= labelTableSize(j) )
				count = count + 1;
				subsetTrain(count) = currTrainIndxDist(k,1);
			else
				break;
			end;
		end;
	end;
	subsetTrain = subsetTrain(1:count);
	subsetTrain = unique(subsetTrain);
	temp = [subsetTrain distance_vI_vJ(subsetTrain,i)];

	minDist = min(temp(:,2));
	currMeanDist = mean(temp(:,2));

	temp = sortrows(temp,2);
	subsetTrain = temp(:,1);
	currScores = zeros(numOfLabels,1);
	labelFreq = zeros(numOfLabels,1);
	pickNbrs = length(subsetTrain);
	for j = 1:pickNbrs
		trainIndx = subsetTrain(j);
		currLabels = find(trainAnnotations(trainIndx,:)==1);
		dist1 = distance_vI_vJ(trainIndx,i);
		distVal = dist1;
		distVal = ((dist1-minDist)/(currMeanDist));
		val = exp(-w*distVal);
		for l = 1:length(currLabels)
			currScores(currLabels(l)) = currScores(currLabels(l)) + val;
			labelFreq(currLabels(l)) = labelFreq(currLabels(l)) + 1;
		end;
	end;
	currScores = currScores/sum(currScores);
	Score_w_I(:,i) = currScores;
end;

% Normalize 
for i = 1:numOfLabels
	maxx = max(Score_w_I(i,:));
	if( maxx>0 )
		Score_w_I(i,:) = (Score_w_I(i,:))/(maxx);
	end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2pknn end %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save('Score_w_I','Score_w_I');
perf = [];
for i = annotLabels
	perf = [perf; computePerf(testAnnotations,Score_w_I,i)];
end;
perf


