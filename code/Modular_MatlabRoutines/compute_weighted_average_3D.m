function res = compute_weighted_average_3D(fNameList, weightList, dataType)

if(isempty(weightList))
	weightList = ones(length(fNameList), 1);
end

res = read3D(fNameList{1}, dataType) * weightList(1);
normalizer = weightList(1);

for i = 2:length(fNameList)

	res = res + read3D(fNameList{i}, dataType) * weightList(i);
	normalizer = normalizer + weightList(i);

end

if(normalizer > 0)
	res = res / normalizer;
else
	error('Weights have to be positive')
end


end