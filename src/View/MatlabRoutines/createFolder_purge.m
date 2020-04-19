function [] = createFolder_purge(folderName)

if(exist(folderName, 'dir') == 7)
 	rmdir(folderName, 's');
end
mkdir(folderName);


end