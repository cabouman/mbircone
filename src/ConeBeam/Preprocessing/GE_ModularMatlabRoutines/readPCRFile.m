function [ rawPCRData ] = readPCRFile( fName )


C = textread(fName, '%s','delimiter', '\n');

rawPCRData.N1 = 0;
rawPCRData.N2 = 0;
rawPCRData.N3 = 0;
rawPCRData.Minimum = 0;
rawPCRData.Maximum = 0;



for i = 1:length(C)
    line = C{i};
    
    if(startsWith_(line, '['))
        sectionName = line;
    end
    
    rawPCRData.N1 = readPC_Entry(rawPCRData.N1, sectionName, '[VolumeData]', 'Volume_SizeX', line);
    rawPCRData.N2 = readPC_Entry(rawPCRData.N2, sectionName, '[VolumeData]', 'Volume_SizeY', line);
    rawPCRData.N3 = readPC_Entry(rawPCRData.N3, sectionName, '[VolumeData]', 'Volume_SizeZ', line);
    
    rawPCRData.Minimum = readPC_Entry(rawPCRData.Minimum, sectionName, '[VolumeData]', 'Min', line);
    rawPCRData.Maximum = readPC_Entry(rawPCRData.Maximum, sectionName, '[VolumeData]', 'Max', line);
    
end


end

