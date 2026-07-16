%% spikeremove2: a spike removal script
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%In some of the data files we acquired, some of the pixels had abnormally
%large photon counts. This script removes these spikes by looking for
%tall spikes, preventing the removal of actual data by comparing with the
%CCD data for the other polarization component.

function [yn,prms] = spikeremove2(y)

yl = length(y);

y1=y(1:yl/2);
y2=y(yl/2+1:yl);

bg1 = min(y1);
bg2 = min(y2);

y1 = y1-bg1;
y2 = y2-bg2;

rat12 = y1./y2;
rat21 = y2./y1;

prms.numfixed = 0;

cutoff = 2;
cutoffl = 1.5;
cutofflplus = 5;
win = 10;

for i=win+1:yl/2-win
    m1 = mean(y1(i-win:i+win));
    m2 = mean(y2(i-win:i+win));
    if(rat12(i)>cutoff && (y1(i) > cutoffl*m2+cutofflplus))
        y1(i) = m2;
        prms.numfixed = prms.numfixed + 1;
    end
    if(rat21(i)>cutoff && (y2(i) > cutoffl*m1))
        y2(i) = m1;
        prms.numfixed = prms.numfixed + 1;
    end

end

yn = [y1+bg1;y2+bg2];

end