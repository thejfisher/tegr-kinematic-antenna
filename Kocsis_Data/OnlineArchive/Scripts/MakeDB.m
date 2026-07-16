%% MakeDB: a database creation function
%Boris Braverman
%Script used for analyzing the data presented in *Sacha Kocsis,
%Boris Braverman, Sylvain Ravets, Martin J. Stevens, Richard P. Mirin, L.
%Krister Shalm, and Aephraim M. Steinberg, “Observing the Average
%Trajectories of Single Photons in a Two-Slit Interferometer”*, published in
%Science on June 4, 2011. 
%
%This script creates a bare-bones database of folders used in the
%subsequent data analysis.
%
%Inputs: 
%*xlfn* is a path to an Excel spreadsheet (.xls) with the following columns:
%[File name] [Useful?] [Range start] [Range stop] [Left] [Right] [Lens],
%which describe the experimental parameters corresponding to the folder
%whose path is given in [File name]. [Range start] is the position (in
%a.u.) of the translation stage corresponding to the first image in the
%folder (i.e. image 0.txt in each data folder); [Range stop] is similarly
%the translation stage position corresponding to the last image in the
%folder; [Left] and [Right] indicate whether the left/right slits were
%imaged in the particular data run; [Lens] indicates whether the imaging
%system was in place during the experimental run.
%*toread* is a 1-dimensional vector of integers indicating which rows of
%the .xls spreadsheet should be loaded into the struct fdata. If this input
%is missing, all rows will be loaded.
%
%Output:
%*fdata* is a struct array containing the same information as the .xls
%file, with only the elements indicated in "toread" loaded in.
function [fdata] = MakeDB(xlfn,toread)

[numdata,stringdata] = xlsread(xlfn);

stringdata(1,:) = [];

numfolders = size(numdata,1);

if(~exist('toread','var'))
    toread = 1:numfolders;
    
end

for i = 1:length(toread)
    fdata(i,1).folder = stringdata{toread(i),1};
    fdata(i,1).range = [numdata(toread(i),1) numdata(toread(i),2)];
    fdata(i,1).leftpresent = numdata(toread(i),3);
    fdata(i,1).rightpresent = numdata(toread(i),4);
    fdata(i,1).lenspresent = numdata(toread(i),5);
end

end