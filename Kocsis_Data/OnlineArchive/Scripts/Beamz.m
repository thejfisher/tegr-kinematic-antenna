%% Beamz
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This function calculates the propagation distance travelled by a Gaussian
%beam of waist w0 and wavelength lambda, if at the present plane, it has a
%width (1/e^2 radius) of R.

function z = Beamz(w0,R,lambda)

zR = pi*w0^2/lambda;

z = zR*sqrt((R/w0).^2-1);


end