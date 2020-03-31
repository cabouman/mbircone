function s = makeCell(c)

if(~iscell(c))
    s = {};
    s{1} = c;
else
    s = c;
end

end

