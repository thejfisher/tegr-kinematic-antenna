%% DBAnalyze: Photon trajectory data analysis script
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
function [fdata,slitparams] = DBAnalyze(fdata)
%%
%Inital parameters:
lambda = 943e-6; %wavelength of light [mm]
pixelpitch = 0.026; %size of each pixel [mm]

lenspresent = GetAll(fdata,'lenspresent');
leftpresent = GetAll(fdata,'leftpresent');
rightpresent = GetAll(fdata,'rightpresent');

%selecting those image sets that have only one slit open or both slits
onlyrightind = find(~leftpresent & rightpresent );
onlyleftind = find(leftpresent & ~rightpresent );
bothind = find(leftpresent & rightpresent ); 

%%
%Begin by looking at the no slit data and extracting the parameters of the
%two slits.
display (['Extracting no-lens parameters from files found in ' fdata((lenspresent==0),1).folder]);
nolensparams = NoLens(fdata((lenspresent==0),1).imdata);

%The obtained parameters are reasonable, although we get a slit width
%difference of 10% for the left slit between the top and right images. At
%least the centre separations are relatively consistent between the two
%polarizations. The values I currently obtain are: 
%     leftsigma: 0.7595
%     rightsigma: 0.7518
%       leftmean: 14.4354
%      rightmean: 19.1227
%       leftamp: 497.5494
%      rightamp: 697.2405       ->Note the right slit is considerably
%                               brighter
%By sigma, we mean the 1/e^2 intensity radius of the beam (since our camera
%records intensity, our Gaussian fitter just returns this as the width
%value. http://www.rp-photonics.com/beam_radius.html
%All units in mm.

%finding the slit separation: because the beams from both slits were
%aligned bvery carefully, we assume this separation to be constant in all
%images.
slitparams.sep = nolensparams.rightmean - nolensparams.leftmean;
slitparams.slitradiusl = nolensparams.leftsigma;
slitparams.slitradiusr = nolensparams.rightsigma;
slitparams.nolensparams = nolensparams;

%%
%next step is to read in the different single-slit images, fit them with
%Gaussians, and then extract the magnification factor as a function of the
%arbitrary translation stage coordinate.

slitparams.lefth = [];
slitparams.leftv = [];
slitparams.left = [];

idx = 1;

for i = 1:length(onlyleftind)
    for j = 1:length(fdata(onlyleftind(i),1).imdata)
        display(['Now fitting ' fdata(onlyleftind(i),1).imdata(j,1).filename]);
        
        transl = fdata(onlyleftind(i),1).imdata(j,1).transl;
        x = fdata(onlyleftind(i),1).imdata(j,1).x;
        
        p = fdata(onlyleftind(i),1).imdata(j,1).subh;
        fitparams = GaussianFit(x,p);
        
        slitparams.lefth.transl(idx,1) = transl;
        slitparams.lefth.snum(idx,1) = onlyleftind(i);
        slitparams.lefth.imnum(idx,1) = j;
        slitparams.lefth.w(idx,1) = fitparams(1);
        slitparams.lefth.x0(idx,1) = fitparams(2);
        slitparams.lefth.A(idx,1) = fitparams(3);
        
        p = fdata(onlyleftind(i),1).imdata(j,1).subv;
        fitparams = GaussianFit(x,p);
        
        slitparams.leftv.transl(idx,1) = transl;
        slitparams.leftv.snum(idx,1) = onlyleftind(i);
        slitparams.leftv.imnum(idx,1) = j;
        slitparams.leftv.w(idx,1) = fitparams(1);
        slitparams.leftv.x0(idx,1) = fitparams(2);
        slitparams.leftv.A(idx,1) = fitparams(3);
        
        p = fdata(onlyleftind(i),1).imdata(j,1).prob;
        fitparams = GaussianFit(x,p);
        
        slitparams.left.transl(idx,1) = transl;
        slitparams.left.snum(idx,1) = onlyleftind(i);
        slitparams.left.imnum(idx,1) = j;
        slitparams.left.w(idx,1) = fitparams(1);
        slitparams.left.x0(idx,1) = fitparams(2);
        slitparams.left.A(idx,1) = fitparams(3);
        
        idx = idx+1;
        
    end
end
%% 
%now the same for the right slit

slitparams.righth = [];
slitparams.rightv = [];
slitparams.right = [];


idx = 1;

for i = 1:length(onlyrightind)
    for j = 1:length(fdata(onlyrightind(i),1).imdata)
        display(['Now fitting ' fdata(onlyrightind(i),1).imdata(j,1).filename]);
        
        transl = fdata(onlyrightind(i),1).imdata(j,1).transl;
        x = fdata(onlyrightind(i),1).imdata(j,1).x;
        
        p = fdata(onlyrightind(i),1).imdata(j,1).subh;
        fitparams = GaussianFit(x,p);
        
        slitparams.righth.transl(idx,1) = transl;
        slitparams.righth.snum(idx,1) = onlyrightind(i);
        slitparams.righth.imnum(idx,1) = j;
        slitparams.righth.w(idx,1) = fitparams(1);
        slitparams.righth.x0(idx,1) = fitparams(2);
        slitparams.righth.A(idx,1) = fitparams(3);
        
        p = fdata(onlyrightind(i),1).imdata(j,1).subv;
        fitparams = GaussianFit(x,p);
        
        slitparams.rightv.transl(idx,1) = transl;
        slitparams.rightv.snum(idx,1) = onlyrightind(i);
        slitparams.rightv.imnum(idx,1) = j;
        slitparams.rightv.w(idx,1) = fitparams(1);
        slitparams.rightv.x0(idx,1) = fitparams(2);
        slitparams.rightv.A(idx,1) = fitparams(3);
        
        p = fdata(onlyrightind(i),1).imdata(j,1).prob;
        fitparams = GaussianFit(x,p);
        
        slitparams.right.transl(idx,1) = transl;
        slitparams.right.snum(idx,1) = onlyrightind(i);
        slitparams.right.imnum(idx,1) = j;
        slitparams.right.w(idx,1) = fitparams(1);
        slitparams.right.x0(idx,1) = fitparams(2);
        slitparams.right.A(idx,1) = fitparams(3);
        
        idx = idx+1;
        
    end
end
%%
%The magnification factor just equals the real slit separation divided by
%the imaged slit separation. Note that we assume that the left and right
%slit images were acquired at the same translation stage positions.

slitparams.magnification = slitparams.sep ./ (slitparams.right.x0-slitparams.left.x0);
slitparams.transl = slitparams.right.transl;


%%

%Next we calculate the invariant point of the imaging system by considering
%the asymptotic point of the two single slit images - which point on the
%ccd they converge to. We simply minimize the variance in the ratio between
%the shifted positions of the left and right slits - e.g. how proportional
%are the two positions for the different image acquisitions?

ImagingCentreProportionality = @(magic) var((slitparams.right.x0-magic)./(slitparams.left.x0-magic));

magicx = 15.8;
opts = optimset('Display','iter','TolX',1e-6,'TolFun',1e-11,'MaxFunEvals',1e4,'Maxiter',1e3,'LargeScale','on');
slitparams.magicx = fminsearch(ImagingCentreProportionality,magicx,opts);

%%
%Next we evaluate the real sizes and positions of the slits as a function 
%of translation stage position
mag = slitparams.magnification;
magicx = slitparams.magicx;

slitparams.realright = slitparams.right;
slitparams.realright.w = slitparams.realright.w.*mag;
slitparams.realright.x0 = (slitparams.realright.x0-magicx).*mag;

slitparams.realrighth = slitparams.righth;
slitparams.realrighth.w = slitparams.realrighth.w.*mag;
slitparams.realrighth.x0 = (slitparams.realrighth.x0-magicx).*mag;

slitparams.realrightv = slitparams.rightv;
slitparams.realrightv.w = slitparams.realrightv.w.*mag;
slitparams.realrightv.x0 = (slitparams.realrightv.x0-magicx).*mag;

slitparams.realleft = slitparams.left;
slitparams.realleft.w = slitparams.realleft.w.*mag;
slitparams.realleft.x0 = (slitparams.realleft.x0-magicx).*mag;

slitparams.reallefth = slitparams.lefth;
slitparams.reallefth.w = slitparams.reallefth.w.*mag;
slitparams.reallefth.x0 = (slitparams.reallefth.x0-magicx).*mag;

slitparams.realleftv = slitparams.leftv;
slitparams.realleftv.w = slitparams.realleftv.w.*mag;
slitparams.realleftv.x0 = (slitparams.realleftv.x0-magicx).*mag;

%%
%Next step is to shift and rescale all the x-coordinates in all the image
%data. The centre of the shift is the invariant imaging point found above.

for i = 1:length(fdata)
    for j = 1:length(fdata(i,1).imdata)
        %This interpolation will fail if the translation stage during
        %two-slit imaging was outside of the one-slit imaging range
        fdata(i,1).imdata(j,1).mag = interp1(slitparams.transl,slitparams.magnification,fdata(i,1).imdata(j,1).transl,'cubic');
        %Rescaling the pixel data
        fdata(i,1).imdata(j,1).xreal = fdata(i,1).imdata(j,1).x  - slitparams.magicx;
        fdata(i,1).imdata(j,1).xreal = fdata(i,1).imdata(j,1).xreal * fdata(i,1).imdata(j,1).mag;
        %Calculating the per mm probability density
        fdata(i,1).imdata(j,1).probden = fdata(i,1).imdata(j,1).prob/(fdata(i,1).imdata(j,1).mag*pixelpitch);
    end
end


%%

%Now use the two-slit fitting to find the distance and slit widths that
%could have corresponded to the observed interference pattern. For this, we
%take the real positions of each interference image, and see if we can find
%corresponding single-slit images. If so, then we fit the interference
%pattern to extract propagation distance and relative phase for the image.
%We then fit a Gaussian propagation curve to the reliable images (which is
%the second half of the double-slit interference pattern images).

for i = 1:length(bothind)
    for j = 1:length(fdata(bothind(i),1).imdata)
        currtrans = fdata(bothind(i),1).imdata(j,1).transl;
        
        leftidx = find(slitparams.left.transl==currtrans,1);
        rightidx = find(slitparams.right.transl==currtrans,1);
        
        if (~isempty(leftidx) && ~isempty(rightidx))
            
            lefts = slitparams.realleft.snum(leftidx);
            lefti = slitparams.realleft.imnum(leftidx);
            
            Aleft.x = fdata(lefts,1).imdata(lefti,1).xreal;
            Aleft.A = fdata(lefts,1).imdata(lefti,1).prob;
            Aleft.w = slitparams.realleft.w(leftidx);
            Aleft.x0 = slitparams.realleft.x0(leftidx);
            
            rights = slitparams.realright.snum(rightidx);
            righti = slitparams.realright.imnum(rightidx);
            
            Aright.x = fdata(rights,1).imdata(righti,1).xreal;
            Aright.A = fdata(rights,1).imdata(righti,1).prob;
            Aright.w = slitparams.realright.w(rightidx);
            Aright.x0 = slitparams.realright.x0(rightidx);
            
            A.x = fdata(bothind(i),1).imdata(j,1).xreal;
            A.A = fdata(bothind(i),1).imdata(j,1).prob;
            
            zi = Beamz(0.6,slitparams.realright.w(rightidx),lambda);
            
            [paramsb,R,At] = FitTwoPattern(A,Aleft,Aright,zi);
            
            fdata(bothind(i),1).imdata(j,1).twopatternz = paramsb(1);
            fdata(bothind(i),1).imdata(j,1).twopatternphi = paramsb(2);
            slitparams.realright.twopatternz(rightidx,1) = paramsb(1);
            slitparams.realright.twopatternphi(rightidx,1) = paramsb(2);
            slitparams.realleft.twopatternz(leftidx,1) = paramsb(1);
            slitparams.realleft.twopatternphi(leftidx,1) = paramsb(2);
            
        end
    end
end

%%
%Now we determine the slit waists by fitting the widths of the two beams as
%a function of their propagation direction.
fitset = 14:46;
leftw = slitparams.realleft.w(fitset);
leftz = slitparams.realleft.twopatternz(fitset);
rightw = slitparams.realright.w(fitset);
rightz = slitparams.realright.twopatternz(fitset);

GaussianSpreadingLeft = @(width0) sum((BeamR(width0,leftz,lambda)-leftw).^2);
GaussianSpreadingRight = @(width0) sum((BeamR(width0,rightz,lambda)-rightw).^2);


width0left = 0.6;
width0right = 0.6;
opts = optimset('Display','iter','TolX',1e-6,'TolFun',1e-11,'MaxFunEvals',1e4,'Maxiter',1e3,'LargeScale','on');
slitparams.realslitradiusl = fminsearch(GaussianSpreadingLeft,width0left,opts);
slitparams.realslitradiusr = fminsearch(GaussianSpreadingRight,width0right,opts);



%%

%and calculate the imaging plane z-positions corresponding to the different
%translation stage positions

slitparams.zleft = Beamz(slitparams.realslitradiusl,slitparams.realleft.w,lambda);
slitparams.zright = Beamz(slitparams.realslitradiusr,slitparams.realright.w,lambda);

slitparams.z = 0.5*(slitparams.zleft+slitparams.zright);

%%
%Now go through and add z values:
for i = 1:length(fdata)
    for j = 1:length(fdata(i,1).imdata)
        %This interpolation will fail if the translation stage during
        %two-slit imaging was outside of the one-slit imaging range
        fdata(i,1).imdata(j,1).z = interp1(slitparams.transl,slitparams.z,fdata(i,1).imdata(j,1).transl,'cubic');
    end
end

%%
%Recall that k_x(weak)/k = tan(sin^-1((V-H)/(H+V)*coeff)) where coeff is
%the measurement coefficient, in units of rad/rad. 

%Measurement coefficient found by Sacha:
MeasCoeff = 1/373;

k = 2*pi/lambda;

for i = 1:length(fdata)
    for j = 1:length(fdata(i,1).imdata)
        h = fdata(i,1).imdata(j,1).probh;
        v = fdata(i,1).imdata(j,1).probv;
        subsum = fdata(i,1).imdata(j,1).subh+fdata(i,1).imdata(j,1).subv;
        
        h1 = fdata(i,1).imdata(j,1).subh;
        v1 = fdata(i,1).imdata(j,1).subv;
        
        deltasum = sqrt(h1+v1);
        deltadiff = sqrt(h1+v1);

        deltarat = sqrt((deltadiff./(h1+v1)).^2+(deltasum.*(h1-v1)./((h1+v1).^2)).^2);
        deltarat = abs(deltarat);
        
        rmimag = min(0.5,max(-0.5,MeasCoeff*(v-h)./(h+v)));
        
        
        fdata(i,1).imdata(j,1).kxkWeak = tan(asin(rmimag));
        fdata(i,1).imdata(j,1).kxkWeak = fdata(i,1).imdata(j,1).kxkWeak.*(subsum>5);
        fdata(i,1).imdata(j,1).kxkWeakDelta = MeasCoeff*deltarat.*(subsum>5); 
    end
end

%%
%Here's a crude way to approximate Wiseman momentum : create 1000 pdf
%points, calculate their positions in all the planes, interpret each curve
%as a function, do some LOESS, extract derivative - and that's kx/kz. Then,
%kx/k is just a simple function of that.

cdfset =(0.0005:0.001:0.9995)';

for i = 1:length(fdata)
    i
    cdfxtable = [];
    ztable = [];
    for j = 1:length(fdata(i,1).imdata)
        if(~any(isnan(fdata(i,1).imdata(j,1).xreal)))
            fdata(i,1).imdata(j,1).CDFset = cdfset;
            fdata(i,1).imdata(j,1).CDFxrealset = interp1(fdata(i,1).imdata(j,1).CDF,fdata(i,1).imdata(j,1).xreal,cdfset,'cubic');
            cdfxtable(j,:) = fdata(i,1).imdata(j,1).CDFxrealset;
            ztable(j,1) = fdata(i,1).imdata(j,1).z;
            fdata(i,1).imdata(j,1).kxkWiseman = [];
        end
    end
    
    for k = 1:size(cdfxtable,2)
        Ltemp = LOESS([ztable cdfxtable(:,k)],1,1,1e4);
        k
        for j = 1:length(fdata(i,1).imdata)
            
            fdata(i,1).imdata(j,1).kxkWiseman(k,1) = tan(asin(Ltemp(j,2)));  
        end
    end
    
end


%%
%Now let's make some trajectories. To do so, we just take the cdf from the
%first plane, then propagate it appropriately to the "weak momentum"
%measured for each of the current positions of the pdf.


%find the "good" series for trajectory reconstruction
good = find( (leftpresent | rightpresent)&lenspresent );

for n = 1:length(good)
    i = good(n);
    
    initplane = 10;
    
    initcdfx = fdata(i,1).imdata(initplane,1).CDFxrealset;
    
    cdfx = initcdfx;
    cdfxWise = initcdfx;

    i
    for j = initplane:length(fdata(i,1).imdata)-1
        fdata(i,1).imdata(j,1).CDFxpropagate = cdfx;
        fdata(i,1).imdata(j,1).CDFxpropagateWiseman = cdfxWise;
        fdata(i,1).imdata(j,1).PDFpropagate = PDFB(cdfx,cdfset);
        fdata(i,1).imdata(j,1).PDFpropagateWiseman = PDFB(cdfx,cdfset);
        
        cdfx = cdfx + (fdata(i,1).imdata(j+1,1).z-fdata(i,1).imdata(j,1).z)*(interp1(fdata(i,1).imdata(j,1).xreal,fdata(i,1).imdata(j,1).kxkWeak,cdfx,'cubic',0));
        cdfxWise = cdfxWise + (fdata(i,1).imdata(j+1,1).z-fdata(i,1).imdata(j,1).z)*(interp1(fdata(i,1).imdata(j,1).CDFxrealset,fdata(i,1).imdata(j,1).kxkWiseman,cdfx,'cubic',0));
        
        j
    end
    fdata(i,1).imdata(length(fdata(i,1).imdata),1).CDFxpropagate = cdfx;
    fdata(i,1).imdata(length(fdata(i,1).imdata),1).CDFxpropagateWiseman = cdfxWise;
    
    
    fdata(i,1).imdata(length(fdata(i,1).imdata),1).PDFpropagate = PDFB(cdfx,cdfset);
    fdata(i,1).imdata(length(fdata(i,1).imdata),1).PDFpropagateWiseman = PDFB(cdfxWise,cdfset);
    
    
end

end