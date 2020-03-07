function time = clear_BinaryFolder( control_4D , plainParamsFile )


tic

fName = control_4D.binaryFnames_C.x_fwd{1};
binaryFolder = fileparts(fName);

disp('Binary Folder:');
disp(binaryFolder);

system(['rm ' binaryFolder, '/*' ]);


time = toc;

return
