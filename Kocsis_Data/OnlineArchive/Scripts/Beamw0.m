%% Beamw0
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This function finds the waist width of a Gaussian beam, given its 1/e^2
%radius a distance R away from the waist. Note that finding this width
%requires solving a quadratic, and hence often has two roots; we select the
%larger root, because this proves to be the consistent choice with our data
%set, in particular, generating agreement between all the different
%interference images acquired.

function w0 = Beamw0(z,R,lambda)

m = R^2 - sqrt(R^4 - 4 * (lambda*z/pi)^2);

w0 = sqrt(m/2);

end