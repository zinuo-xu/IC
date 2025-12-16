% This file is used to align the captured image to CCF atlas, and transform
% the corrdination of the neuron positions

% original by Yeyi Cai
% 
% updated/modified by fxr in 20250121

close all;
clear,clc;

%% Directories

atlas = importdata('C:\Users\xuzinuo\Desktop\map\CCFmaps\utility\atlas_reduce0.8\atlas_top_projection.mat'); 

%%%%%%%%%% change: choose the registered Intrinsic_to_CCF data
load('C:\Users\xuzinuo\Desktop\map\M79CCFalign_res_try1.mat')

%%%%%%%%% change : all directories
intMapDir='C:\Users\xuzinuo\Desktop\map\M79\output'; % path of intrinsic map
intDir='C:\Users\xuzinuo\Desktop\map\M79\output\test.TIF'; % path of intrinsic image
int_ifDir='C:\Users\xuzinuo\Desktop\map\M79\output\infocus.TIF'; % path of in focused intrinsic image  

%%%%%%%% change: your directories
FDir = 'C:\Users\xuzinuo\Desktop\map\';
% path of the folder containing AVG_whole_brain_3d.tif and wholebrain_output.mat 


%% Parameters
ag=30;

% Load the needed data
% Load intrinsic map

Aalt=loadtiff([intMapDir,'\Aalt.tiff']);
Aazi=loadtiff([intMapDir,'\Azi.tiff']);
A=(Aalt+Aazi)/2;
dsRate=2; % spatial downsample rate of the intrinsic result


% Load intrinsic image
Intr=loadtiff(intDir);
[Y,X]=size(Intr);
XX=round(X/dsRate);
YY=round(Y/dsRate);
Intr=imresize(Intr,[YY,XX]);

% Debackground
Intr=-double(Intr);
se = strel('disk',30);
background = imopen(Intr,se);
Intr=Intr-background;
Intr=-Intr;

Intr_infocus=loadtiff(int_ifDir);
Intr_infocus=imresize(Intr_infocus,[YY,XX]);
Intr_infocus=-double(Intr_infocus);
se = strel('disk',30);
background = imopen(Intr_infocus,se);
Intr_infocus=Intr_infocus-background;
Intr_infocus=-Intr_infocus;



%% Step1, map the fluorescence image to the in-focus intrinsic image
% get the transform

global_img = loadtiff([FDir,'\AVG_whole_brain_3d.tif']);
global_img = double(global_img);
img_scaled = mat2gray(global_img);
F = imadjust(img_scaled);
F = fliplr(F);
[imgH,imgW,~] = size(F);

Intr_infocus = mat2gray(Intr_infocus);

%%
[selectedMovingPoints,selectedFixedPoints] = cpselect(F,Intr_infocus,'Wait',true);

save([FDir,'mappoints.mat'],...
            'selectedMovingPoints','selectedFixedPoints'); 

%% apply the transform
tt=estimateGeometricTransform2D(selectedMovingPoints,selectedFixedPoints,'projective');
T5=tt.T;
Fr=imwarp(F,tt,'OutputView',imref2d(size(Intr_infocus)));
imshowpair(Fr,Intr_infocus);

%% Show the overlay of the ccf and the fluorescence image
figure();
F2ccf=imwarp(F,projective2d(T5*T_infocus2CCF),'OutputView',imref2d(size(CCF)));
imshow(F2ccf,[]); hold on;
scatter(r,n,2,'filled');
xlim([1,size(CCF,2)]);
ylim([1,size(CCF,1)]);

%% Step2, for each neural footprint, calculate its position on the atlas

% load data and apply the transform
load([FDir,'wholebrain_output.mat']); 

tform = projective2d(T5*T_infocus2CCF);

xx = whole_center(:,2);
yy = whole_center(:,1);

xx_n = imgW -xx +1; % the image was flipped in step3, so xx should be adjusted

[xx_tform,yy_tform] = transformPointsForward(tform,xx_n,yy);


figure,imshow(CCF,[]);hold on; scatter(xx_tform,yy_tform); % check the position

%% define neurons region 

valid_neuron_x = xx_tform; 
valid_neuron_y = yy_tform; 

brain_region = {};
brain_region_id = [];


for neuroni =1:size(whole_center,1)

    curr_x = round(valid_neuron_x(neuroni));
    curr_y = round(valid_neuron_y(neuroni));

    brain_region_idx = Pg(curr_y, curr_x); % Pg is 456*528 where 456 is the y direction

    if brain_region_idx == 0 % not in cortical area
        brain_region_k=  'None';
        k = 0;
    else
        for k = 1 : length(atlas.ids)
            if strcmp(sprintf('%d', brain_region_idx), atlas.ids{k})
                break
            end
        end
        brain_region_k=  atlas.names{k};
    end

    brain_region{neuroni} = brain_region_k;
    brain_region_id = [brain_region_id,k];
end

save([FDir,'brain_results.mat'],...
            'brain_region','brain_region_id', ...
            'valid_neuron_x', 'valid_neuron_y');

%% e.g., find all primary visual cortex
conss = cellfun(@(x) ischar(x) && contains(x,'Primary visual'), brain_region); 
positions = find (conss); %  the number of primary visual neurons
area = brain_region(positions);

%% check selected neurons
imshow(imadjust(img_scaled),[])
hold on
scatter(whole_center(positions,2),whole_center(positions,1))

%% Functions
function [CCF,Pg1,n,r]=getCCFatlas(ag)
%     CCFdir=['CCFmaps\Pg',num2str(ag),'.tiff'];
%     Spdir=['CCFmaps\Sp',num2str(ag),'.tiff'];
%     CCF=loadtiff(CCFdir);
%     CCF=flipud(CCF); 
    load(['/all_data/Registration/CYY_map/CCFmaps/CCF_',num2str(ag),'.mat']); % to be changed
    CCF=Pg;
    Pg1=Sp;
    CCF=flipud(CCF); 
    Pg1=flipud(Pg1);
    [n,r]=find(CCF~=0); % Edges of ccf
end


function oimg = loadtiff(path)
    s = warning('off', 'all'); % To ignore unknown TIFF tag.
    % Frame number
    tiff = Tiff(path, 'r');
    frame = 0;
    while true
        frame = frame + 1;
        if tiff.lastDirectory(), break; end;
        tiff.nextDirectory();
    end
    k_struct = 0;
    tiff.setDirectory(1);
    for kf = 1:frame
        if kf == 1
            n1 = tiff.getTag('ImageWidth');
            m1 = tiff.getTag('ImageLength');
            spp1 = tiff.getTag('SamplesPerPixel');
            sf1 = tiff.getTag('SampleFormat');
            bpp1 = tiff.getTag('BitsPerSample');
            if kf ~= frame
                tiff.nextDirectory();
            end
            continue;
        end  
        n2 = tiff.getTag('ImageWidth');
        m2 = tiff.getTag('ImageLength');
        spp2 = tiff.getTag('SamplesPerPixel');
        sf2 = tiff.getTag('SampleFormat');
        bpp2 = tiff.getTag('BitsPerSample');     
        if n1 ~= n2 || m1 ~= m2 || spp1 ~= spp2 || sf1 ~= sf2 || bpp1 ~= bpp2
            k_struct = k_struct + 1;
            tifstruct(k_struct).m = m1;
            tifstruct(k_struct).n = n1;
            tifstruct(k_struct).spp = spp1;
            tifstruct(k_struct).frame = kf-1;
            tifstruct(k_struct).data_type = DataType(sf1, bpp1);
        end
        if kf ~= frame
            tiff.nextDirectory();
        else
            if k_struct > 0
                k_struct = k_struct + 1;
                tifstruct(k_struct).m = m2;
                tifstruct(k_struct).n = n2;
                tifstruct(k_struct).spp = spp2;
                tifstruct(k_struct).frame = kf;
                tifstruct(k_struct).data_type = DataType(sf2, bpp2);
            end
        end
        n1 = n2; m1 = m2; spp1 = spp2; sf1 = sf2; bpp1 = bpp2;
    end  
    if k_struct == 0
        if spp1 == 1
            oimg = zeros(m1, n1, frame, DataType(sf1, bpp1)); % grayscle
            for kf = 1:frame
                tiff.setDirectory(kf);
                oimg(:, :, kf) = tiff.read();
            end
        else
            oimg = zeros(m1, n1, spp1, frame, DataType(sf1, bpp1)); % color
            for kf = 1:frame
                tiff.setDirectory(kf);
                oimg(:, :, :, kf) = tiff.read();
            end
        end
    else
        k_cell = 1;
        kf_start = 1;
        for kc=1:k_struct
            if tifstruct(kc).spp == 1
                temp = zeros(tifstruct(kc).m, tifstruct(kc).n, tifstruct(kc).frame-kf_start+1, tifstruct(kc).data_type);
                for kf=1:tifstruct(kc).frame-kf_start+1
                    tiff.setDirectory(kf+kf_start-1);
                    temp(:, :, kf) = tiff.read();
                end
                oimg{k_cell} = temp; k_cell = k_cell+1;
            else
                temp = zeros(tifstruct(kc).m, tifstruct(kc).n, 3, tifstruct(kc).frame-kf_start+1, tifstruct(kc).data_type);
                for kf=1:tifstruct(kc).frame-kf_start+1
                    tiff.setDirectory(kf+kf_start-1);
                    temp(:, :, :, kf) = tiff.read();
                end
                oimg{k_cell} = temp; k_cell = k_cell+1;
            end
            kf_start = tifstruct(kc).frame + 1;
        end
    end    
    tiff.close();    
    warning(s);
end

function out = DataType(sf, bpp)
    switch sf
        case 1
            switch bpp
                case 8
                    out = 'uint8';
                case 16
                    out = 'uint16';
                case 32
                    out = 'uint32';
            end
        case 2
            switch bpp
                case 8
                    out = 'int8';
                case 16
                    out = 'int16';
                case 32
                    out = 'int32';
            end
        case 3
            switch bpp
                case 32
                    out = 'single';
                case 64
                    out = 'double';
            end
    end
end

