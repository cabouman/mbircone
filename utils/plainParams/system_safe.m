function [exit_status, commandLineOutput] = system_safe(command)
% This saves command output to tempfile to make sure it is not corrupted by user input
fName = tempname;

[exit_status, ~] = system(['(', command, ') > ', fName]);

commandLineOutput = fileread(fName);


end