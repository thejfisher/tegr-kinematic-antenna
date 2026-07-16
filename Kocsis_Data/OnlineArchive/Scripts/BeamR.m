%% BeamR
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This function simply calculates the 1/e^2 radius for a Gaussian beam of
%known waist w0 and wavelength lambda, after propagating a distance z away
%from its waist.

function R = BeamR(w0,z,lambda)

zR = pi*w0^2/lambda;

R = w0*sqrt(1+(z/zR).^2);

end