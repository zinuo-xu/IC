% This file is used to align the captured image to CCF atlas, and transform
% the corrdination of the neuron positions

% original by Yeyi Cai
% 
% updated/modified by fxr in 20250121

close all;
clear,clc;

%% Directories

atlas = importdata("C:\Users\xuzinuo\Desktop\IC实验课\M79\wholebrain_output.mat"); 

%%%%%%%%% change : all directories
intMapDir="C:\Users\xuzinuo\Desktop\IC实验课\M79\output"; % path of intrinsic map
intDir="C:\Users\xuzinuo\Desktop\IC实验课\M79\output\test.TIF"; % path of intrinsic image
int_ifDir="C:\Users\xuzinuo\Desktop\IC实验课\M79\output\infocus.TIF"; % path of in focused intrinsic image  

outputdir="C:\Users\xuzinuo\Desktop\IC实验课\M79\output"; % path of your own folder

%% Parameters
ag=30;
minAg=15;
maxAg=60;

dxyCCF=25; % The size of pixel on CCF atlas


regInfocusMode=1; % 0 for automatic,1 for manual
dxyInt=6.5/(2*0.63);

% Load the needed data
% Load intrinsic map
Pfinal=loadtiff([intMapDir,'Pfinal.tiff']);
Aalt=loadtiff([intMapDir,'Aalt.tiff']);
Aazi=loadtiff([intMapDir,'Azi.tiff']);
A=(Aalt+Aazi)/2;
dsRate=2; % spatial downsample rate of the intrinsic result
% Load ccf
P1ccf=[180,231]; % landmark1 of ccf
P2ccf=[330,247]; % landmark2 of ccf (might be not very accurate, just try)
[CCF,Pg,n,r]=getCCFatlas(ag);  % directory needed to be changed, please refer to the function at the end
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

%% Get the roi on the image, select the round window area
figure();
imshow(Intr_infocus,[]);
title('Please select the window area, double click to confirm');
h_poly=roipoly();
x_coords = h_poly(:, 1);
y_coords = h_poly(:, 2);
roimask=double(h_poly);
close all;

%% Get point of confidence from the user
imshow(Intr,[]);
title('请依次在大血管上选择一个前信点和一个后信点');

% Get the input from the user
[P1intr, P2intr] = ginput(2);

% Plot what the user selected
hold on;
plot(P1intr, P2intr, 'ro');
hold off;

%% Step2, map the intrinsic image to the in-focus intrinsic image
figure();
if regInfocusMode==0 % Automated register the infocus image to the     
    [tt,peakcorr]=imregcorr(imgaussfilt(Intr_infocus,[10,10]), Intr);
    T4=tt.T;
    InfocusB=imwarp(Intr_infocus,tt,'OutputView',imref2d(size(Intr_infocus)));
    imshowpair(InfocusB,Intr);
else % Manual mode
    rs=0.91;
    t_x=15;
    t_y=0;

    while true

        T_scale=[rs, 0, 0; 0, rs, 0; 0, 0, 1];
        T_translate=[1, 0, 0; 0, 1, 0; t_x, t_y, 1];

        % Display the adjusted image
        InfocusB=imwarp(Intr_infocus,affine2d(T_translate*T_scale),'OutputView', imref2d(size(Intr)));
    
        imshowpair(InfocusB,Intr);
        pause(0.05);
    
        % Get input
        userInput = input('w:up, s:down, a:left, d:right, x:bigger, c: smaller, q:quit）：', 's');
        if isempty(userInput)
            disp('无效输入，请按照提示输入。');
            continue;
        end
        ang=userInput(1);
        if length(userInput)==1 
            moveN=1;
        elseif length(userInput)==0
            continue;
        else
            moveN=str2double(userInput(2:end));
        end 
        if isnan(moveN)
            disp('无效输入，请按照提示输入。');
            continue
        end
        % Deal with input
        switch ang
            case 'w'
                disp('向上');
                t_y=t_y-moveN;
            case 's'
                disp('向下');
                t_y=t_y+moveN;
            case 'a'
                disp('向左');
                t_x=t_x-moveN;
            case 'd'
                disp('向右');
                t_x=t_x+moveN;
            case 'x'
                disp('放大');
                rs=rs*1.01^moveN;
            case 'c'
                disp('缩小');
                rs=rs/1.01^moveN;
            case 'q'
                disp('退出');
                break;  % quit
            otherwise
                disp('无效输入，请按照提示输入。');
        end
    T4=T_translate*T_scale;
    end

end


%% Based on the scale and the vessel, do the first step registration
% Resize so that the scale is the same between intrinsic imaging and CCF
rszRate=dxyInt*dsRate/dxyCCF/T_scale(1,1);
T1=[rszRate, 0, 0; 0, rszRate, 0; 0, 0, 1]; % The transformation matrix of resize
% Tranform the image and the confidence point
Intr_1 = imwarp(Intr, affine2d(T1));
P1intr_1=transformPointsForward(affine2d(T1), P1intr');
P2intr_1=transformPointsForward(affine2d(T1), P2intr');

% Rotate so that P1-P2 is parrellel to P1-P2 on CCF
P12ccf=P2ccf-P1ccf;
P12intr=P2intr_1-P1intr_1;
P12ccf=P12ccf/norm(P12ccf);
P12intr=P12intr/norm(P12intr);
rotation_angle = atan2(P12intr(2), P12intr(1)) - atan2(P12ccf(2), P12ccf(1));
T2 = [cos(rotation_angle), -sin(rotation_angle), 0; sin(rotation_angle), cos(rotation_angle), 0; 0, 0, 1];
tform = affine2d(T2);
% Tranform the image and the confidence point
Intr_2=imwarp(Intr_1,tform,'OutputView', imref2d(size(Intr_1)));
P1intr_2=transformPointsForward(tform, P1intr_1);
P2intr_2=transformPointsForward(tform, P2intr_1);
[H,W]=size(Intr_2);

% Translate so that P1-P2 overlap with P1-P2 on CCF
t_y=((P2ccf(2)-P2intr_2(2))+(P1ccf(2)-P1intr_2(2)))/2;
t_x=((P2ccf(1)-P2intr_2(1))+(P1ccf(1)-P1intr_2(1)))/2;
T3 = [1, 0, 0; 0, 1, 0; t_x, t_y, 1];
tform = affine2d(T3);
% Tranform the image and the confidence point
Intr_3=imwarp(Intr,affine2d(T1*T2*T3),'OutputView', imref2d(size(CCF)));
P1intr_3=transformPointsForward(affine2d(T3), P1intr_2);
P2intr_3=transformPointsForward(affine2d(T3), P2intr_2);

% Visualize
figure();iptsetpref('ImshowAxesVisible','on');
imshow(Intr_3,[]); hold on;
scatter(r,n,2,'filled');
xlim([1,size(CCF,2)]);
ylim([1,size(CCF,1)]);

%% Adjust the translation based on the user input
T3a=T3;
ra=rotation_angle;
disp('输入微调，输入 "q" 键退出。');
atran=[0,0]; % [dy,dx]
aro=0; % rotation
while true
    
    % Display the adjusted image
    T3a (3,1) = T3 (3,1) + atran(2);
    T3a(3,2)=T3(3,2)+atran(1);
    ra=rotation_angle+aro;
    T2a=[cos(ra), -sin(ra), 0; sin(ra), cos(ra), 0; 0, 0, 1];
    Intr_3=imwarp(Intr,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF)));
    P3=imwarp(Pfinal,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF)));
    A3=(imwarp(Aazi,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF)))+...
    imwarp(Aazi,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF))))/2;
    roi3=imwarp(roimask,affine2d(T4*T1*T2a*T3a),'OutputView', imref2d(size(CCF)));


    ax1=subplot(2,2,1);
    imshow(Intr_3,[]); hold on;
    scatter(r,n,2,'filled');hold on;
    scatter(229,221,'red','filled');
    xlim([1,size(CCF,2)]);
    ylim([1,size(CCF,1)]); hold off;

    ax2=subplot(2,2,2);
%     imagesc(P3,'AlphaData',A3,'AlphaDataMapping','scaled'); 
    imagesc(P3);
    caxis(ax2,[-1,1]);
    colormap(ax2,'turbo');
    hold on;
    scatter(r,n,2,'filled');
    xlim([1,size(CCF,2)]);
    ylim([1,size(CCF,1)]); hold off;
    axis equal;   

    ax3=subplot(2,2,4);
    linkaxes([ax1,ax2,ax3]);
    imagesc(roi3); hold on;
    scatter(r,n,2,'filled');
    xlim([1,size(CCF,2)]);
    ylim([1,size(CCF,1)]); hold off;
    axis equal;

    pause(0.05);

    % Get input
    userInput = input('w:up, s:down, a:left, d:right, x:rotate left, c: rotate right, +, next atlas, - previous atlas, q:quit）：', 's');
    if isempty(userInput)
        disp('无效输入，请按照提示输入。');
        continue;
    end
    ang=userInput(1);
    if length(userInput)==1 
        moveN=1;
    elseif length(userInput)==0
        continue;
    else
        moveN=str2double(userInput(2:end));
    end 
    if isnan(moveN)
        disp('无效输入，请按照提示输入。');
        continue
    end
    % Deal with input
    switch ang  
        case 'w'
            disp('向上');            
            atran(1)=atran(1)-moveN;
        case 's'
            disp('向下');
            atran(1)=atran(1)+moveN;
        case 'a'
            disp('向左');
            atran(2)=atran(2)-moveN;
        case 'd'
            disp('向右');
            atran(2)=atran(2)+moveN;
        case 'c'
            disp('右转');
            aro=aro-moveN/180*pi;
        case 'x'
            disp('左转');
            aro=aro+moveN/180*pi;
        case '+'
            disp('Next atlas');
            ag=ag+5;
            if ag<=maxAg
                [CCF,Pg,n,r]=getCCFatlas(ag);
            else
                disp('Last atlas reached');
                ag=ag-5;
            end
        case '-'
            disp('Previous atlas');
            ag=ag-5;
            if ag>=minAg
                [CCF,Pg,n,r]=getCCFatlas(ag);
            else
                disp('First atlas reached');
                ag=ag+5;
            end
        case 'g'
            disp('larger');
            T1(1,1)=T1(1,1)*1.01^moveN;
            T1(2,2)=T1(2,2)*1.01^moveN;
        case 'f'
            disp('smaller')
            T1(1,1)=T1(1,1)/1.01^moveN;
            T1(2,2)=T1(2,2)/1.01^moveN;
        case 'q'
            disp('退出');
            break;  % quit
                    
        otherwise
            disp('无效输入，请按照提示输入。');
    end

end

%% Show the overlay result
figure();
ax1=subplot(2,2,1);
in_T=imwarp(Intr_infocus,projective2d(T4*T1*T2a*T3a),'OutputView',imref2d(size(CCF)));
imshow(in_T,[]); hold on;
scatter(r,n,2,'blue','filled');
axis off;
ax2=subplot(2,2,2);
P3=imwarp(Pfinal,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF)));
A3=(imwarp(Aazi,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF)))+...
    imwarp(Aazi,affine2d(T1*T2a*T3a),'OutputView', imref2d(size(CCF))))/2;
imagesc(P3,'AlphaData',A3,'AlphaDataMapping','scaled'); 
colormap(ax2,'turbo');
hold on;
scatter(r,n,2,'filled');
xlim([1,size(CCF,2)]);
ylim([1,size(CCF,1)]); hold off;
axis equal;   
axis off;

ax3=subplot(2,2,4);
linkaxes([ax1,ax2,ax3]);
imagesc(roi3); hold on;
scatter(r,n,2,'filled');
xlim([1,size(CCF,2)]);
ylim([1,size(CCF,1)]); hold off;
axis equal;
axis off;

%%%%%%%%%%%%%
% rememnber to change the name of the stored file
savefig([outputdir,'alignres.fig']);

%% Save all the results from registration
T_decofusIntr2CCF=T1*T2a*T3a;
T_infocus2decofus=T4;
T_infocus2CCF = T4*T1*T2a*T3a;

%%
% rememnber to change the name of the stored file
save([outputdir,'CCFalign_res.mat'],'T_decofusIntr2CCF','T_infocus2decofus','T_infocus2CCF','ag','CCF','Pg','n','r');

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

