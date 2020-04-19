function [ scans ] = flipScans( scans )

scans.darkmeanImg = scans.darkmeanImg(end:-1:1, end:-1:1);
scans.darkvarImg = scans.darkvarImg(end:-1:1, end:-1:1);
scans.blankmeanImg = scans.blankmeanImg(end:-1:1, end:-1:1);
scans.blankvarImg = scans.blankvarImg(end:-1:1, end:-1:1);

scans.object = scans.object(end:-1:1, end:-1:1, :);




end

