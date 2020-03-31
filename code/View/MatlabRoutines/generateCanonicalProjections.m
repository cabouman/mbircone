function [ ] = generateCanonicalProjections( folderName, dispName, img, opts)



outFolder = folderName;


for i = 1:3

%%
    proj = squeeze(sum(img,i));
    
    if(i<3)
        proj = proj';
    end
    
    lo = prctile(proj(:), 0.5);
    hi = prctile(proj(:), 99.5);
    dom = hi - lo;
    proj = proj - lo;
    if(dom>0)
        proj = proj/dom;
    end
    

    
    imwrite(proj, [outFolder, '/', dispName, '_projection_', num2str(i), '.png']);

    
end

end
    