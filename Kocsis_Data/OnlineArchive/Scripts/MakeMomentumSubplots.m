figure('paperposition', [0 0 3.0 8.0]);

% q=subplot(2,2,1);
% set(q,'Position',[0.09 0.6 0.34 0.32])
% MakeMomentumPlot(fdata,1,16,q)
% q=subplot(2,2,2);
% set(q,'Position',[0.57 0.6 0.34 0.32])
% MakeMomentumPlot(fdata,1,29,q)
% q=subplot(2,2,3);
% set(q,'Position',[0.09 0.1 0.34 0.32])
% MakeMomentumPlot(fdata,1,37,q)
% q=subplot(2,2,4);
% set(q,'Position',[0.57 0.1 0.34 0.32])
% MakeMomentumPlot(fdata,1,48,q)

l = 0.16;
w = 0.66;
h = 0.22;

q=subplot(4,1,1);
set(q,'Position',[l 0.75 w h])
MakeMomentumPlot(fdata,1,16,q,0)
text(-8.5,0.00,'(A)','FontSize',14) 
q=subplot(4,1,2);
set(q,'Position',[l 0.52 w h])
MakeMomentumPlot(fdata,1,29,q,0)
text(-8.5,0.00,'(B)','FontSize',14) 
q=subplot(4,1,3);
set(q,'Position',[l 0.29 w h])
MakeMomentumPlot(fdata,1,37,q,0)
text(-8.5,0.00,'(C)','FontSize',14) 
q=subplot(4,1,4);
set(q,'Position',[l 0.06 w h])
MakeMomentumPlot(fdata,1,48,q,1)
text(-8.5,0.00,'(D)','FontSize',14) 