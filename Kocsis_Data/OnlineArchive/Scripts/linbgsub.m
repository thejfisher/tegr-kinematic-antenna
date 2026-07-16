%% linbgsub: a linear background subtraction tool
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This script 

function yn = linbgsub(x,y,bgregion)

npts = length(bgregion);

matrix = [ones(npts,1)];

matrix = [x(bgregion),matrix];

p = lscov(matrix,y(bgregion),ones(1,npts));
   
yn = y - polyval(p,x);

end