%% LOESS
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This function is a simple implementation of the locally weighted
%scatterplot smoothing technique. This code is never used in the analysis
%of the weak momentum values, only for finding the photon momentum values
%from probability conservation.
%
function [yloess] = LOESS(data,deg,window,sigma)

if(~exist('deg','var'))
    deg = 2;
end

if(~exist('window','var'))
    window = 3;
end

%data = sortrows(data);


width = max(3,deg+1);
width = max(width,window);

for i = 1:size(data,1)
    
   interval = [max(1,i-width):min(size(data,1),i+width)];
   
   npts = length(interval);
   
   xs = data(interval,1);
   Currx = data(i,1);
   xoff = mean(xs);
   
   xs = xs - xoff;
   Currx = Currx - xoff;
   
   varvals = data(interval,2);
   
   matrix = [ones(npts,1)];
   for j = 1:deg
        matrix = [xs.^j,matrix];
   end
   
    xdiff = xs - Currx;
%    ourind = find(xdiff==0);
%    
%    xdiff(ourind) = Inf;
%    
%    wts = sqrt(abs(1./xdiff));
%    
%    TotalWt = sum(wts);
%    wts(ourind) = 0.3*TotalWt;

    wts = exp(-((xdiff/sigma).^2));
    
  
   p = lscov(matrix,varvals,wts);
   
   
   for j = 1:deg+1
        yloess(i,j) = polyval(p,Currx);
        p = polyder(p);
   end   
end




end