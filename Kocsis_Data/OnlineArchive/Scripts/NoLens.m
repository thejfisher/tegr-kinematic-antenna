%% NoLens: data extraction from lens-free images
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function [nolensparams] = NoLens(data)

lambda = 0.000943;
pixelpitch = 0.026;

xvals = (1:1024)';

leftbg = [475:500 625:650]';
rightbg = [650:675 800:825]';
leftx = (475:650)';
rightx = (650:825)';

%data.rawh;
%data.rawv;

sublh = linbgsub(xvals,data.rawh,leftbg);
subrh = linbgsub(xvals,data.rawh,rightbg);

sublv = linbgsub(xvals,data.rawv,leftbg);
subrv = linbgsub(xvals,data.rawv,rightbg);

[lhp,r,funlhp] = GaussianFit(leftx,sublh(leftx));
[rhp,r,funrhp] = GaussianFit(rightx,subrh(rightx));
[lvp,r,funlvp] = GaussianFit(leftx,sublv(leftx));
[rvp,r,funrvp] = GaussianFit(rightx,subrv(rightx));

lhp = lhp*pixelpitch;
lvp = lvp*pixelpitch;
rhp = rhp*pixelpitch;
rvp = rvp*pixelpitch;

nolensparams.lhp = lhp;
nolensparams.rhp = rhp;

nolensparams.lvp = lvp;
nolensparams.rvp = rvp;


nolensparams.leftsigma = 0.5*(lhp(1)+lvp(1));
nolensparams.rightsigma = 0.5*(rhp(1)+rvp(1));

nolensparams.leftmean = 0.5*(lhp(2)+lvp(2));
nolensparams.rightmean = 0.5*(rhp(2)+rvp(2));

nolensparams.leftamp = 0.5*(lhp(3)+lvp(3));
nolensparams.rightamp = 0.5*(rhp(3)+rvp(3));



end