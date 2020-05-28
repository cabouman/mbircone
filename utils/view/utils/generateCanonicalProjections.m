function [ ] = generateCanonicalProjections( folderName, dispName, img, opts)



outFolder = folderName;


for i = 1:3

%%
    proj = squeeze(sum(img,i));
    
    if(i<3)
        proj = proj';
    end
    
    if(~isempty(which('prctile')))  % check in case Stats toolbox isn't loaded
        lo = prctile(proj(:), 0.5);
        hi = prctile(proj(:), 99.5);
    else
        lo = my_prctile(proj(:), 0.5);
        hi = my_prctile(proj(:), 99.5);
    end
    dom = hi - lo;
    proj = proj - lo;
    if(dom>0)
        proj = proj/dom;
    end
    

    
    imwrite(proj, [outFolder, '/', dispName, '_projection_', num2str(i), '.png']);

    
end

end
