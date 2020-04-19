function [ DirStructure ] = findBinaryFilesInFolder( folderName_binary, extensions )



D = dir(folderName_binary);



isBinaryFile = zeros(length(D), 1);
for i = length(D):-1:1
    
    [~, ~, extension] = fileparts([folderName_binary, '/', D(i).name]);
    
    for j = 1:length(extensions)
        
        if(    strcmp(extension, extensions{j}) ...
            && D(i).isdir == 0 ...
          )
            isBinaryFile(i) = 1;
        end
        
    end
    
    
end
DirStructure = D(isBinaryFile==1);



end

