function tf = startsWith_(s, pattern)

%STARTSWITH True if text starts with pattern.


k = strfind(s,pattern);

if isempty(k);
	tf = false;
else
	tf = k==1;
end