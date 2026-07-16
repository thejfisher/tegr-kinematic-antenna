%% spikeremove 3: 3 applications of spikeremove2
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function [yn,prms] = spikeremove3(y)

prms.numfixed = 0;

z = y;

for i =1:3
    [z,prm] = spikeremove2(z);
    prms.numfixed = prm.numfixed+prms.numfixed;
end

yn = z;


end