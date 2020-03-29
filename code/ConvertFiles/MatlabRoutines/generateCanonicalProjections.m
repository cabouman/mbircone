function [ ] = generateCanonicalProjections( folderName, dispName, img, opts)



outFolder = folderName;


for i = 1:3

%%
    proj = squeeze(sum(img,i));
    
    if(i<3)
        proj = proj';
    end

    dom = max(proj(:)) - min(proj(:));
    if(dom>0)
        proj = proj/dom;
    end
    proj = proj - min(proj(:));

    
    imwrite(proj, [outFolder, '/', dispName, '_projection_', num2str(i), '.png']);

    
end

end
    