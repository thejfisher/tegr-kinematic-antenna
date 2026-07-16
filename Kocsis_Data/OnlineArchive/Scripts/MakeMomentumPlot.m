function MakeMomentumPlot(fdata,i,j,axhand,xaxflag)

ax1 = axhand;

set(ax1,'Color','none');

ax2 = axes('Position',get(ax1,'Position'),...
'XAxisLocation','bottom',...
'YAxisLocation','right',...
'Color','none',...
'XColor','k','YColor','k');
set(ax1,'YAxisLocation','left');

a1pos = get(ax1,'Position');

newa2pos = a1pos;
newa2pos(4) = a1pos(4)*0.7;
newa1pos = a1pos;
newa1pos(4) = a1pos(4)*0.3;
newa1pos(2) = a1pos(2)+a1pos(4)*0.7;



set(ax1,'Position',newa1pos);
set(ax2,'Position',newa2pos);

set(ax1,'XTickLabel',[]);

x = GetAll(fdata(i,1).imdata,'xreal');
kxkWeakDelta = GetAll(fdata(i,1).imdata,'kxkWeakDelta');
kxkWiseman = GetAll(fdata(i,1).imdata,'kxkWiseman');
kxkWeak = GetAll(fdata(i,1).imdata,'kxkWeak');
CDFxrealset = GetAll(fdata(i,1).imdata,'CDFxrealset');
z = GetAll(fdata(i,1).imdata,'z');

subh = GetAll(fdata(i,1).imdata,'subh');
subv = GetAll(fdata(i,1).imdata,'subv');

hold(ax1,'on');
hold(ax2,'on');

grid(ax1,'on');
grid(ax2,'on');


plot(ax1,x(j,:),subh(j,:),'r','linewidth',1)
plot(ax1,x(j,:),subv(j,:),'b','linewidth',1)

not0 = (kxkWeak(j,:)~=0);

plot(ax2,CDFxrealset(j,:),kxkWiseman(j,:),'m.','MarkerSize',10)

ehand = errorbar(ax2,x(j,not0),kxkWeak(j,not0),kxkWeakDelta(j,not0),'.k','MarkerSize',8,'linewidth',0.25);

errorbar_tick(ehand,250);

%plot(ax2,x(j,not0),kxkWeak(j,not0),'.k','MarkerSize',8);
%plot(ax2,x(j,not0),kxkWeak(j,not0)+kxkWeakDelta(j,not0),'-k','LineWidth',0.5);
%plot(ax2,x(j,not0),kxkWeak(j,not0)-kxkWeakDelta(j,not0),'-k','LineWidth',0.5);


xli = 6;

xlim(ax1,[-xli xli]);
xlim(ax2,[-xli xli]);

ylim(ax1,[-20 400]);
ylim(ax2,[-1.2e-3 1.2e-3]);

tihand = title(ax1,sprintf('Z coordinate: %.1f m',z(j)/1000),'fontsize',10);
set(tihand,'position',[0 300]);

xlabel(ax2,'Transverse coordinate [mm]','fontsize',14);
ylabel(ax1,'Photon counts','fontsize',10);
ylabel(ax2,'k_x/k','fontsize',10);

if (xaxflag ==0)
    xlabel(ax2,[]);
    set(ax2,'XTickLabel',[]);

end

set(ax1,'fontsize',10);
set(ax2,'fontsize',10);

%set(ax2,'Position',get(ax1,'Position'));

end