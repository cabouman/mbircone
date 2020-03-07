function indexList_new = select_interlacedSubset(indexList, num_subsets, index_subsets)

if index_subsets >= num_subsets
	error('select_interlacedSubset: index_subsets >= num_subsets');
end

indexList_new = indexList(index_subsets+1:num_subsets:end);

return


