function [ ] = generateStatsFile( folderName, dispName, img, stats)



fName = [folderName, '/', dispName, '_stats.txt'];

%immse_ = immse(img,zeros(size(img), 'like', img));
% immse() is in Image toolbox.. removing dependency here
immse_ = mean(img(:).^2);
rmse_ = sqrt(immse_);
mean_ = mean(img(:));
min_ = min(img(:));
max_ = max(img(:));


i = 0;

i=i+1; C{i} = ['Stats of ', dispName];

i=i+1; C{i} = ['immse = ', num2str(immse_), ' = 1 / ', num2str(1/immse_)];
i=i+1; C{i} = ['rmse = ', num2str(rmse_), ' = 1 / ', num2str(1/rmse_)];
i=i+1; C{i} = ['mean = ', num2str(mean_)];
i=i+1; C{i} = '';

i=i+1; C{i} = ['min = ', num2str(min_)];
i=i+1; C{i} = ['max = ', num2str(max_)];
i=i+1; C{i} = '';

i=i+1; C{i} = ['min_norm = ', num2str(stats.min_norm)];
i=i+1; C{i} = ['max_norm = ', num2str(stats.max_norm)];
i=i+1; C{i} = '';

i=i+1; C{i} = ['range1 = ', num2str(stats.range1(1)), ':', num2str(stats.range1(end))];
i=i+1; C{i} = ['range2 = ', num2str(stats.range2(1)), ':', num2str(stats.range2(end))];
i=i+1; C{i} = ['range3 = ', num2str(stats.range3(1)), ':', num2str(stats.range3(end))];

fid = fopen(fName, 'w');
for i = 1:length(C)
	fprintf(fid, '%s\n', C{i});
end
fclose(fid);


end

