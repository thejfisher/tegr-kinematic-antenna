%% Bohmdataread: the basic data reading function
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function Imgs = Bohmdataread(froot,opts)

pixelpitch = 0.026; %size of a camera pixel (in mm)

tmpfiles = dir(froot);
tmpfiles = tmpfiles(find(~GetAll(tmpfiles,'isdir')))';

for i = 1:length(tmpfiles)
    fname = strcat(froot,tmpfiles(i).name);
    
    Imgs(i,1).transl = NaN;
    namefindimage = strfind(tmpfiles(i).name,'Image');
    display(['File name:' tmpfiles(i).name]);
    if(~isempty(namefindimage))
        num = sscanf(tmpfiles(i).name(namefindimage+5:length(tmpfiles(i).name)),'%d');
       Imgs(i,1).transl = opts.range(1)+(opts.range(2)-opts.range(1))*num/(length(tmpfiles)-1);
    end
    
    tmp = load(fname);
    
    tmp = tmp';
    
    Imgs(i,1).raw = tmp;
    
    tmp = spikeremove3(tmp);
    
    Imgs(i,1).raws = tmp;
    
    Imgs(i,1).filename = fname;
    Imgs(i,1).rawh = tmp(1:1024);
    Imgs(i,1).rawv = tmp(1025:2048);
    
    xvals = pixelpitch*(1:1024)';
    bgregion = [400:475 750:900]';
    
    Imgs(i,1).subh = linbgsub(xvals,Imgs(i,1).rawh,bgregion);
    Imgs(i,1).subv = linbgsub(xvals,Imgs(i,1).rawv,bgregion);
        
    Imgs(i,1).subh = round(Imgs(i,1).subh);
    Imgs(i,1).subv = round(Imgs(i,1).subv);
    
    %Pixel counts cannot be negative...
    Imgs(i,1).subh(find(Imgs(i,1).subh<=0)) = 1e-11;%some small, non-0 number
    Imgs(i,1).subv(find(Imgs(i,1).subv<=0)) = 1e-11;
    
    Imgs(i,1).roi = (475:725)';
    Imgs(i,1).x = xvals(Imgs(i,1).roi);
    
    Imgs(i,1).subh = Imgs(i,1).subh(Imgs(i,1).roi);
    Imgs(i,1).subv = Imgs(i,1).subv(Imgs(i,1).roi);
    
    %Smooth the data - this bit was implemented but *never* used in the
    %data analysis presented in the paper
    
    %t = LOESS([Imgs(i,1).roi Imgs(i,1).subh],2,8,3);
    %Imgs(i,1).subh = t(:,1);
    
    %t = LOESS([Imgs(i,1).roi Imgs(i,1).subv],2,8,3);
    %Imgs(i,1).subv = t(:,1);
        
    Imgs(i,1).probh = Imgs(i,1).subh / sum(Imgs(i,1).subh);
    Imgs(i,1).probv = Imgs(i,1).subv / sum(Imgs(i,1).subv);
    
    Imgs(i,1).prob = 0.5*(Imgs(i,1).probh+Imgs(i,1).probv);
    
    Imgs(i,1).CDF = CDFB(Imgs(i,1).prob);
end

%sorting the images based on their translation stage positions
translvals = GetAll(Imgs,'transl');
[A,IX] = sort(translvals);
Imgs = Imgs(IX);
    
end