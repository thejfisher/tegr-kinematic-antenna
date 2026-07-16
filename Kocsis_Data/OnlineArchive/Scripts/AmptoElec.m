%% AmptoElec
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This script takes an amplitude for a Gaussian beam propagated by some
%distance z, and returns the electric field in the current plane, using the
%formula from http://www.rp-photonics.com/gaussian_beams.html (standard
%Gaussian beam propagation formula)

function E = AmptoElec(x,A,params)

lambda = 943e-6;
ii = sqrt(-1);

k = 2*pi/lambda;

%w0 is the 1/e^2 radius; x_0 is the middle of the Gaussian in the pattern.

z = params.z;
w = params.w;
x0 = params.x0;

w0 = Beamw0(z,w,lambda);

if imag(w0) ~= 0
    E = NaN*ones(size(A));
    return;
end

E = sqrt(A);
zr = pi*w0^2/lambda;

R = z*(1+(zr/z)^2);

phi = (k*z - atan(z/zr) + k*(x-x0).^2/(2*R));

E = E.*exp(-ii*phi);

end