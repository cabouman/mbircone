function [exit_status, commandLineOutput] = system_errcheck(command)

[exit_status, commandLineOutput] = system(command);

disp(commandLineOutput);
if exit_status ~= 0
	error(['Error executing command ', '"', command, '"']);
end

end