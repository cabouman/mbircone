function [new_img] = shepperd_interpolation( interpLocations, sampled_mask, img, p, win, ifEmptyNeighborhood)


% a pixels(i,j) interpolated value is affected only by pixels within a window of
% i-win:i+win,j-win:j+win

[len1,len2] = size(img);
new_img = img;

numInterpLocations = size(interpLocations,1);


for k=1:numInterpLocations
    i = interpLocations(k,1);
    j = interpLocations(k,2);

    if(sampled_mask(i,j)~=0)
        error('shepperd_interpolation: interpLocations and sampled_mask inconsistent');
    end

    id1_min = max(i-win,1);
    id1_max = min(i+win,len1);
    id2_min = max(j-win,1);
    id2_max = min(j+win,len2);
    window_id = sampled_mask(id1_min:id1_max,id2_min:id2_max);
    window_samples = img(id1_min:id1_max,id2_min:id2_max);

    winlen1 = id1_max-id1_min+1;
    winlen2 = id2_max-id2_min+1;

    id1 = repmat((id1_min:id1_max)',1,winlen2);
    id2 = repmat((id2_min:id2_max),winlen1,1);
    distVal = sqrt( (id1-i).^2 + (id2-j).^2 );
    wt = 1./(distVal.^p);
    new_img(i,j) = sum( window_samples(window_id==1).*wt(window_id==1) )/sum(wt(window_id==1));
    if isnan(new_img(i,j))

        switch ifEmptyNeighborhood

        case 'giveError'
            error('shepperd_interpolation: neighborhood empty');
            exit;

        case 'useFullImage'
            id1 = repmat((1:len1)',1,len2);
            id2 = repmat((1:len2),len1,1);
            distVal = sqrt( (id1-i).^2 + (id2-j).^2 );
            wt = 1./(distVal.^p);
            new_img(i,j) = sum( img(sampled_mask==1).*wt(sampled_mask==1) )/sum(wt(sampled_mask==1));

        end

    end

end


return