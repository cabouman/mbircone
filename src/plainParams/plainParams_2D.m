function value_out = plainParams_2D(executablePath, get_set, masterFile_list, masterField, subField, value, resolveFlag)

value_out = cell(size(masterFile_list));

for i = 1:size(value_out,1)
for j = 1:size(value_out,2)

	value_out{i,j} = plainParams(executablePath, get_set, masterFile_list{i,j}, masterField, subField, value, resolveFlag);

end
end



end