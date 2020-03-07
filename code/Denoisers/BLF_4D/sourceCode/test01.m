close all
clear variables
clc

%load a 3D image
load volume1
I=double(R);

%select filter parameters
sigmaS=5;
sigmaR=15;
samS=5;
samR=15;

%get the filter result
Ibf=bilateral3(I, sigmaS,sigmaS,sigmaR,samS,samR);

%visualize a few slices of the volume
i=[15 25 40 45];
for j=1:length(i)
    figure
    subplot 121
    imagesc(I(:,:,i(j)));
    axis image
    colormap gray
    set(gca,'xticklabel',[ ]);
    set(gca,'yticklabel',[ ]);
    subplot 122
    imagesc(Ibf(:,:,i(j)));
    axis image
    colormap gray
    set(gca,'xticklabel',[ ]);
    set(gca,'yticklabel',[ ]);
end
shg