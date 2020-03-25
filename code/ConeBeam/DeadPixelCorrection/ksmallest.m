function [els, bin_arr]  = ksmallest(arr, k)




s = size(arr);
arr_lin = arr(:);

[~, I_sort] = sort(arr_lin);
I_unsort = zeros(length(I_sort), 1);
I_unsort(I_sort) = 1:length(I_unsort);

arr_lin = arr_lin(I_sort);

els = arr_lin(1:k-1);

arr_lin(1:k) = 1;
arr_lin(k+1:end) = 0;
arr_lin = arr_lin(I_unsort);
bin_arr = reshape(arr_lin, s);


end