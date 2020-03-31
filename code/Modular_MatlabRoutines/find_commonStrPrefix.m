function commonStr = find_commonStrPrefix(s1, s2)

commonLen = min(length(s1),length(s2));

n = 0;
while strncmpi(s1,s2,n)
	n = n+1;
end

commonStr = s1(1:n-1);

return
