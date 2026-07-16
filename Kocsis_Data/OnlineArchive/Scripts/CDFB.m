%% CDFB: a cumulative distribution function calculation tool
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function cdf = CDFB(prob)

cdf = zeros(length(prob),1);

cdf(1) = prob(1)/2;

for i = 2:length(prob)
    cdf(i) = cdf(i-1) + (prob(i)+prob(i-1))/2;
end

end