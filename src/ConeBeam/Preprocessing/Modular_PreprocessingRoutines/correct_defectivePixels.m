function [sino, wght, driftReference_sino, jigMeasurementsSino] = correct_defectivePixels( sino, wght, driftReference_sino, jigMeasurementsSino, defectivePixelMap, method_str );

switch method_str
    case 'interpolate'
        [rowVect, colVect] = find( defectivePixelMap);
        interpLocations = [ rowVect(:) colVect(:) ];
        sampled_mask = double( defectivePixelMap==0 );

        for i=1:size(sino,3)
            sino(:,:,i) = shepperd_interpolation( interpLocations, sampled_mask, sino(:,:,i), 1, 1, 'giveError');
        end

        for i=1:size(wght,3)
            wght(:,:,i) = shepperd_interpolation( interpLocations, sampled_mask, wght(:,:,i), 1, 1, 'giveError');
        end

        for i=1:size(driftReference_sino,3)
            driftReference_sino(:,:,i) = shepperd_interpolation( interpLocations, sampled_mask, driftReference_sino(:,:,i), 1, 1, 'giveError');
        end

        
        for i=1:size(jigMeasurementsSino,3)
            jigMeasurementsSino(:,:,i) = shepperd_interpolation( interpLocations, sampled_mask, jigMeasurementsSino(:,:,i), 1, 1, 'giveError');
        end
        

        
    otherwise
        error(['correct_defectivePixels: invalid method_str: ', method_str])
end

end

