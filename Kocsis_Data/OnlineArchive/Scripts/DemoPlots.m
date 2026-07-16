%% DemoPlots: illustrating our data
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%%
figure
hold on
title(fdata(1,1).imdata(90,1).filename)
plot(fdata(1,1).imdata(90,1).raw,'.b')
plot(fdata(1,1).imdata(90,1).raws,'-r')
xlabel('data point number')
ylabel('photon counts')
legend('raw counts','after spike removal')
print(gcf,'-dpng','-r300','../Figures/SpikeRemoval.png');
close
%%
figure
hold on
title(fdata(1,1).imdata(30,1).filename)
plot(fdata(1,1).imdata(30,1).rawh,'-b')
plot(fdata(1,1).imdata(30,1).rawv,'-r')
xlabel('pixel number')
ylabel('photon counts')
legend('H','V')
print(gcf,'-dpng','-r300','../Figures/H and V.png');
close
%%
figure
hold on
title(fdata(1,1).imdata(30,1).filename)
plot(fdata(1,1).imdata(30,1).x,fdata(1,1).imdata(30,1).subh,'-b')
plot(fdata(1,1).imdata(30,1).x,fdata(1,1).imdata(30,1).subv,'-r')
xlabel('position on sensor [mm]')
ylabel('photon counts')
legend('H','V')
print(gcf,'-dpng','-r300','../Figures/H and V - background subtracted.png');
close
%%
figure
hold on
title('Comparison of z coordinate obtained from Gaussian propagation and from 2-slit fitting')
transl = GetAll(fdata(1,1).imdata,'transl');
zfit = GetAll(fdata(1,1).imdata,'twopatternz',1:2:91);
z = GetAll(fdata(1,1).imdata,'z');
plot(transl,z,'-b')
plot(transl(1:2:91),zfit,'.r')
xlabel('translation stage position')
ylabel('z coordinate [mm]')
legend('Gaussian propagation','fitting of interference pattern');
print(gcf,'-dpng','-r300','../Figures/z comparison.png');
close
%%
figure
hold on
title(fdata(1,1).imdata(30,1).filename)
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).probh,'.-b')
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).probv,'.-r')
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('probability of landing in given pixel')
legend('H','V')
print(gcf,'-dpng','-r300','../Figures/Probabilities.png');
close
%%
figure
hold on
title(fdata(1,1).imdata(30,1).filename)
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).probh,'.-b')
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).probv,'.-r')
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).kxkWeak*5,'.-k')
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('Probabilities, Weak momentum x 5 [k_x/k]')
legend('H','V','momentum')
print(gcf,'-dpng','-r300','../Figures/Momentum and Probabilities plot.png');
close
%%
figure
hold on
title(fdata(1,1).imdata(30,1).filename)
plot(fdata(1,1).imdata(30,1).xreal,fdata(1,1).imdata(30,1).kxkWeak,'.-b')
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('Weak momentum [k_x/k]')
print(gcf,'-dpng','-r300','../Figures/Momentumplot.png');
close
%%
figure
hold on
cdfs = GetAll(fdata(1,1).imdata,'CDFset');
cdfx = GetAll(fdata(1,1).imdata,'CDFxpropagate',11:51);
title('CDF positions at different planes, propagated using weak momentum values')
plot(cdfx(1,:),cdfs(11,:),'.-b')
plot(cdfx(11,:),cdfs(21,:),'.-r')
plot(cdfx(21,:),cdfs(31,:),'.-g')
plot(cdfx(31,:),cdfs(41,:),'.-m')
plot(cdfx(41,:),cdfs(51,:),'.-k')

xlim([-8 8])
legend(num2str(z([11 21 31 41 51])))
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('probability of landing to left of given position (H+V)')
print(gcf,'-dpng','-r300','../Figures/CDF-propagation.png');
close
%%
figure
hold on
cdfs = GetAll(fdata(1,1).imdata,'CDFset');
cdfxreal = GetAll(fdata(1,1).imdata,'CDFxrealset',11:51);
title(['CDF using weak momentum vs. actual cdf, z = ' num2str(z(41)) ' mm'])
plot(cdfx(31,:),cdfs(41,:),'.-b')
plot(cdfxreal(31,:),cdfs(41,:),'.-r')

legend('using weak momentum','actual')

xlim([-8 8])
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('probability of landing to left of given position (H+V)')
print(gcf,'-dpng','-r300','../Figures/CDF-Comparison1.png');
close
%%
figure
hold on
cdfs = GetAll(fdata(1,1).imdata,'CDFset');
cdfxreal = GetAll(fdata(1,1).imdata,'CDFxrealset',11:51);
title(['CDF using weak momentum vs. actual cdf, z = ' num2str(z(21)) ' mm'])
plot(cdfx(11,:),cdfs(21,:),'.-b')
plot(cdfxreal(11,:),cdfs(21,:),'.-r')

legend('using weak momentum','actual')

xlim([-8 8])
xlabel('position on sensor relative to axis of imaging system [mm]')
ylabel('probability of landing to left of given position (H+V)')
print(gcf,'-dpng','-r300','../Figures/CDF-Comparison2.png');
close
%%
MakeMomentumSubplots
print(gcf,'-dpng','-r300','../Figures/Figure 2 - Momentum comparison.png');
close
%%
plotrange = 10:50;
trajset = 21:12:979;
display(length(trajset));

z = GetAll(fdata(1,1).imdata,'z',plotrange);
CDFxpropagate = GetAll(fdata(1,1).imdata,'CDFxpropagate',plotrange);

figure

hold on

plot(z,CDFxpropagate(:,trajset),'k-','LineWidth',1)
grid on
set(gca,'LineWidth',1);
xlabel('Propagation distance[mm]','fontsize',14)
ylabel('Transverse coordinate[mm]','fontsize',14)
ylim([-6.7 5.2])
xlim([2500 8500]);
set(gca,'fontsize',14);

print(gcf,'-dpng','-r300','../Figures/Figure 3 - Trajectories.png');
close
%%
plotrange = 10:50;
trajset = 21:12:979;
display(length(trajset));

z = GetAll(fdata(1,1).imdata,'z',plotrange);
CDFxpropagate = GetAll(fdata(1,1).imdata,'CDFxpropagateWiseman',plotrange);

figure

hold on

plot(z,CDFxpropagate(:,trajset),'k-','LineWidth',1)
grid on
set(gca,'LineWidth',1);
xlabel('Propagation distance[mm]','fontsize',14)
ylabel('Transverse coordinate[mm]','fontsize',14)
ylim([-6.7 5.2])
xlim([2500 8500]);
set(gca,'fontsize',14);

print(gcf,'-dpng','-r300','../Figures/Probability conservation trajectories.png');
close
%%

plotrange = 10:50;
trajset = 15:4:985;

z = GetAll(fdata(1,1).imdata,'z',plotrange);
prob = GetAll(fdata(1,1).imdata,'prob',plotrange);
probden = GetAll(fdata(1,1).imdata,'probden',plotrange);
x = GetAll(fdata(1,1).imdata,'xreal',plotrange);
CDFxpropagate = GetAll(fdata(1,1).imdata,'CDFxpropagate',plotrange);

zmesh1 = z*ones(1,size(probden,2));
zmesh2 = z*ones(1,size(CDFxpropagate,2));

for i = 1:length(z)
    
    probdeninter(i,:) = interp1(x(i,:),probden(i,:),CDFxpropagate(i,:),'linear',0);
end

probdeninter = probdeninter + 0.007;

%probdeninter = interp2(zmesh1,x,probden,zmesh2,CDFxpropagate,'cubic',0);

figure

hold on

surfrange = 31:220;

surf(zmesh1(:,surfrange),x(:,surfrange),probden(:,surfrange),'edgealpha',0.1)
alpha(1);
grid on
xlim([2500 8500]);
ylim([-12 9]);
zlim([0 0.3]);
caxis([-0.15 0.2]);
plot3(zmesh2(:,trajset),CDFxpropagate(:,trajset),probdeninter(:,trajset),'-k','linewidth',0.25,'MarkerSize',2)

xlabel('Transverse coordinate (x) [mm]','position',[9400 3.2 0],'rotation',3.5,'FontSize',10);
ylabel('Propagation distance (z) [mm]','position',[4000 -14.7 0],'rotation',-77,'FontSize',10);
zlabel('Probability density [mm^{-1}]','position',[1000 -12.7 0.01],'FontSize',10);

%colormap jet;
%colormap summer;
colormap (1-0.8*(1-hot));
%colormap (flipdim(gray,1));
%camlight headlight; 
lighting phong;

view(84,66);

print(gcf,'-dpng','-r300','../Figures/Figure 4 - 3D Trajectories.png');

hold off
close 