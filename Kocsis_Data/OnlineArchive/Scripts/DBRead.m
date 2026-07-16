%% DBRead: data loader and preprocessor
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This script loads the data from the folders indicated in the fdata struct
%array using the auxiliary script "Bohmdataread", and stores them in the
%imdata field of the struct. 

function [fdata] = DBRead(fdata)

for i = 1:size(fdata,1)
    froot = fdata(i,1).folder;
    display(['Now reading/preprocessing data from ',froot]);
    opts.range = fdata(i,1).range;
    fdata(i,1).imdata = Bohmdataread(froot,opts);
end

end