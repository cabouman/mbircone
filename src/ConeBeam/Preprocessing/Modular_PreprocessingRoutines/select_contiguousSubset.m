function indexList_new = select_contiguousSubset(indexList, num_chunk, ind_chunk)

if ind_chunk >= num_chunk
	error('select_contiguousSubset: ind_chunk >= num_chunk');
end

len_list = zeros(1,num_chunk);

% split evenly
num_per_set = floor(length(indexList)/num_chunk);
for i=1:num_chunk
	len_list(i) = num_per_set;
end

% distribute the remaining
remaining_index_num = length(indexList)- num_per_set*num_chunk ;
for i=1:remaining_index_num
	len_list(i) = len_list(i) + 1;
end

start_id = sum(len_list(1:ind_chunk+1-1))+1 ;
end_id = sum(len_list(1:ind_chunk+1)) ;

indexList_new = indexList(start_id:end_id);

return


