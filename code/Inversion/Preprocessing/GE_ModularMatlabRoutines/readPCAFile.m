function [ rawPCAData ] = readPCAFile( fName )

C = textread(fName, '%s','delimiter', '\n');

rawPCAData.NumberImages = 0;
rawPCAData.FOD = 0;
rawPCAData.FDD = 0;
rawPCAData.Magnification = 0;
rawPCAData.PixelsizeX = 0;
rawPCAData.PixelsizeY = 0;

rawPCAData.cx = 0;
rawPCAData.cy = 0;
rawPCAData.DimX = 0;
rawPCAData.DimY = 0;

rawPCAData.FreeRay = 0;
rawPCAData.RotationSector = 0;

for i = 1:length(C)
    line = C{i};
    
    if(startsWith_(line, '['))
        sectionName = line;
    end
    
    rawPCAData.NumberImages = readPC_Entry(rawPCAData.NumberImages, sectionName, '[CT]', 'NumberImages', line);
    rawPCAData.FOD = readPC_Entry(rawPCAData.FOD, sectionName, '[Geometry]', 'FOD', line);
    rawPCAData.FDD = readPC_Entry(rawPCAData.FDD, sectionName, '[Geometry]', 'FDD', line);
    rawPCAData.Magnification = readPC_Entry(rawPCAData.Magnification, sectionName, '[Geometry]', 'Magnification', line);
    
    rawPCAData.PixelsizeX = readPC_Entry(rawPCAData.PixelsizeX, sectionName, '[Detector]', 'PixelsizeX', line);
    rawPCAData.PixelsizeY = readPC_Entry(rawPCAData.PixelsizeY, sectionName, '[Detector]', 'PixelsizeY', line);
    

    rawPCAData.cx = readPC_Entry(rawPCAData.cx, sectionName, '[Geometry]', 'cx', line);
    rawPCAData.cy = readPC_Entry(rawPCAData.cy, sectionName, '[Geometry]', 'cy', line);
    
    rawPCAData.DimX = readPC_Entry(rawPCAData.DimX, sectionName, '[Image]', 'DimX', line);
    rawPCAData.DimY = readPC_Entry(rawPCAData.DimY, sectionName, '[Image]', 'DimY', line);


    rawPCAData.FreeRay = readPC_Entry(rawPCAData.FreeRay, sectionName, '[Image]', 'FreeRay', line);
    rawPCAData.RotationSector = readPC_Entry(rawPCAData.RotationSector, sectionName, '[CT]', 'RotationSector', line);
    
end

rawPCAData.TIFFPathNames = readTIFFPathNamesFromPCAFile( fName );

if(length(rawPCAData.TIFFPathNames) < rawPCAData.NumberImages)
    error([num2str(rawPCAData.TIFFPathNames), ' = length(rawPCAData.TIFFPathNames) < rawPCAData.NumberImages = ', num2str(rawPCAData.NumberImages), ]);
end




end

