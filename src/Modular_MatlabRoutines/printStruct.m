function [] = printStruct(s, nameOfStruct, isInterpretable, identation, fileID)

% this can print out structs but does not truncate long sting fields
if(~exist('isInterpretable'))
    isInterpretable = 0;
end
if(~exist('identation'))
    identation = '';
end
if(~exist('fileID'))
    fileID = 1;
end

if isempty(identation)
    genericDispRoutine(' ', fileID);
end

if ~isempty(s)

    f = fields(s);
    if(isInterpretable~=1)
        genericDispRoutine([nameOfStruct, ' : '], fileID);
    end

    for i = 1:length(f)

    	content = s.(f{i});

        if isstruct(content)
            if(isInterpretable==1)
                new_name = [nameOfStruct, '.', f{i} ];
                new_identation = identation;
            else
                new_name = ['    ', f{i} ];
                new_identation = [ '    ', identation ];
            end
            printStruct(content, new_name, isInterpretable, new_identation, fileID);
        else
        	if(ischar(content))
        		content = ['''', content, ''''];
        	else
        		content = num2str(content);
            end

            if(isInterpretable==1)
                genericDispRoutine([ identation, nameOfStruct, '.', f{i}, ' = [', content, '];'], fileID);
            else
                genericDispRoutine([ identation, '    ', f{i}, ' : ', content], fileID);
            end
        end
      
    end

end


return


function genericDispRoutine(str, fileID)

% disp(str)
fprintf(fileID, [str, '\n']);

return


