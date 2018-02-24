function draw_result(n,avg_best,avg_best1,std_best1,avg_best2,std_best2,avg_best3,std_best3,title)
    %here is show at best f1
        %draw the precision   
        tiles = {'presision','recall','F1','Fp'};
        figure('NumberTitle','off','Name',title);
        n1 = size(avg_best1,1);
        n2 = size(avg_best2,1);
        n3 = size(avg_best3,1);
    for i=1:4
        subplot(2,2,i);
        %original
        plot(avg_best(i)*ones(1,n),'g');
        hold on;
        %without GE
        errorbar([1:1:n1],avg_best1(:,i),std_best1(:,i),'-ro');
        hold on;
        %with GE
        errorbar([1:1:n2],avg_best2(:,i),std_best2(:,i),'-bo');
        hold on;
        %concatente
        errorbar([1:1:n3],avg_best3(:,i),std_best3(:,i),'-mo');
        
        
        l1 = legend('original','without GE','with GE','concatente','Orientation','horizontal');%
        set(l1,'Position',[.13,.94,.4,.05]);
        set(gca,'xticklabel',[0:100:100*n]);
        xlabel('dimension');
        ylabel(tiles{i});
        grid;
    end
end