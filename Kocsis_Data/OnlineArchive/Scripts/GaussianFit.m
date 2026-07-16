%% GaussianFit: a Gaussian fitter
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function [paramsg,R,fitfun] = GaussianFit(x,y)

sy = sum(y)*(max(x)-min(x))/length(x);
mx = sum(abs(y).*x)/sum(abs(y));

expectxsq = sum(abs(y).*(x.^2))/sum(abs(y));
sd = sqrt(expectxsq - mx^2)*sqrt(2);

%sigma,mean,amp

    function [g] = g(params,r)
        g = params(3)*exp(-2*((r-params(2))/params(1)).^2)/sqrt(0.5*pi*params(1)^2);
    end

    function [d] = dist(params)
        d = (g(params,x)-y);
    end

paramsg = [sd,mx,sy];

opts = optimset('Display','none','TolX',1e-6,'TolFun',1e-11,'MaxFunEvals',1e4,'Maxiter',1e3);

[paramsg,R] = lsqnonlin(@dist,paramsg,[],[],opts);

% figure
% hold on
% plot(x,y,'.r');
fitfun = g(paramsg,x);
% plot(x,fitfun,'-k');
% hold off

end    