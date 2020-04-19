function value = plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag)
%
%      if get_set = 'get' then
%          - contents of value will be ignored and
%          - read data will be returned
%
%      if get_set = 'set' then
%          - contents of value are used to set params file and
%          - returned data is not specified
%
%      Any argument = '' will be ignored.
%
% This file need to be in the same folder as the plainParams.sh executable!

% to test run:
% plainParams('plainParams.sh', 'get', '../../control/ConeBeam/master.txt', 'preprocessingParams', 'N_beta_all', '', '')   

command = ['bash ', executablePath];

if(~strcmp(get_set, ''))
	command = [command, ' -a ', '''', get_set, ''''];
	if(strcmp(get_set, 'get'))
		value = '';
	end
end

if(~strcmp(masterFile, ''))
	command = [command, ' -m ', '''', masterFile, ''''];
end

if(~strcmp(masterField, ''))
	command = [command, ' -F ', '''', masterField, ''''];
end

if(~strcmp(subField, ''))
	command = [command, ' -f ', '''', subField, ''''];
end

if(~strcmp(value, ''))
	command = [command, ' -v ', '''', value, ''''];
end

if(~strcmp(resolveFlag, ''))
	command = [command, ' ', resolveFlag];
end

% disp(command)
% [exit_status, commandLineOutput] = system(command);
[exit_status, commandLineOutput] = system_safe(command);

if(exit_status == 0)
	value = commandLineOutput(1:end-1);
else
	disp(commandLineOutput);
	error(['Error executing command ', '"', command, '"']);

end