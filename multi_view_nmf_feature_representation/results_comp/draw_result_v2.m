function draw_result_v2(n,avg_best,avg_bests,std_bests,titles,options)
    %here is show at best f1
        %draw the precision   
        tiles = {'presision','recall','F1','N+'};
        figure('NumberTitle','off','Name',titles);
    colors = {'go','bo','r*','c+','mo','g*','b*','r*','c*','m*','g','b','r','c','m'};
    numOftype = size(avg_bests);
    for i=1:4
        subplot(2,2,i);
        %original
%         plot([0:n-1],avg_best(i)*ones(1,n),'g');
%         hold on;
        
        %normal
        for j=1:numOftype
        n1 = size(avg_bests{j},1);
            errorbar([0:n1-1],avg_bests{j}(:,i),std_bests{j}(:,i),['-',colors{j+1}]);
            hold on;
        end
        set(gca,'xticklabel',[-500:500:500*n]);
        xlabel('dimension');
        ylabel(tiles{i});
        title(['fig-',num2str(i)]);
        grid;
    end
            
    l1 = legend('original',options.labels,'Orientation','horizontal');%
    set(l1,'Position',[.13,.94,.4,.05]);
end