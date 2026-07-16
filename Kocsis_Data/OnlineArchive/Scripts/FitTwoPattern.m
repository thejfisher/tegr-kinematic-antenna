%% FitTwoPattern
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This function takes two amplitude functions A1 and A2 assumed to be
%cross-sections through Gaussian beams, and then attempts to determine the
%propagation direction and the relative phase between the two beams. The
%amplitudes of the two beams are the remaining fitting parameters.

function [paramsb,R,At] = FitTwoPattern(A,A1,A2,zi)

ii = sqrt(-1);
lambda = 0.000943; %wavelength of our light

wmin = min(A1.w,A2.w);

zmax = Beamz(wmin/sqrt(2),wmin,lambda);

    %We model the expected amplitude of the interference pattern
    function [At] = Ath(fitprm) 
        params.z = fitprm(1)*1000; 
        params.w = A1.w;
        params.x0 = A1.x0;
        
        E1 = AmptoElec(A1.x,A1.A,params);
        
        params.w = A2.w;
        params.x0 = A2.x0;
        
        E2 = AmptoElec(A2.x,A2.A,params);
        
        E1 = interp1(A1.x,E1,A.x,'linear',0);
        E2 = interp1(A2.x,E2,A.x,'linear',0);
        
        E = E1*sqrt(fitprm(3))+E2*sqrt(fitprm(4))*exp(-ii*fitprm(2));
        
        At = E.*conj(E);
    end

    function [d] = dist(fitprm)
        d = (Ath(fitprm)-A.A);
    end

%Parameters are z,phi, and amplitudes of the two slits.

paramsb = [zi/1000,1,0.5,0.5];

opts = optimset('Display','iter','TolX',1e-11,'TolFun',1e-11,'MaxFunEvals',1e3,'Maxiter',1e2);

[paramsb,R] = lsqnonlin(@dist,paramsb,[0 -2*pi 0 0],[zmax/1000 2*pi 1 1],opts);

At = Ath(paramsb);

paramsb(1) = paramsb(1)*1000;

% figure
% hold on
% plot(A.x,A.A,'.k');
% plot(A.x,At,'-r');
% hold off
% pause

end