%% GetAll: a struct array data retrieval tool
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function [v] = GetAll(CB,varname,interval)

if(~exist('interval','var'))
    interval = 1:size(CB,1);
end

for i = 1:length(interval)
    v(i,:) = eval(strcat('CB(interval(i)).',varname));
end


end