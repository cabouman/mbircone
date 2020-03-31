function [cmd] = system_bulidMultiCommand(commands, isParallel)
% Examples:
% commands = {'sleep 3; touch a.txt', 'touch b.txt', 'touch c.txt'};
% 
% isParallel = true;
% >> cmd = '(sleep 3; touch a.txt & pids+=("$! ") ; touch b.txt & pids+=("$! ") ; touch c.txt & pids+=("$! ") ;  wait ${pids[0]} &&  wait ${pids[1]} &&  wait ${pids[2]} && :)'
% 
% isParallel = false; 
% >> cmd = '(fail="false"; sleep 3; touch a.txt || fail="true" ; touch b.txt || fail="true" ; touch c.txt || fail="true" ;  [ $fail == "false" ])'


if(isParallel)
    
    cmd = ''; 

    for i = 1:length(commands)
        cmd = [cmd, commands{i}, ' & pids+=("$! ") ; ']; % documenting process IDs 
    end

    for i = 1:length(commands) % Check all the pids for success
            cmd = [cmd, ' wait ${pids[', num2str(i-1), ']} && '];
    end
    
    cmd = [cmd, ':']; % This is just NOP for simplicity
    
else
    
    cmd = 'fail="false"; ';
    
    for i = 1:length(commands)
        cmd = [cmd, commands{i} ' || fail="true" ; ']; % documents fail if one command fails
    end                                                % still executes all commands
    
    cmd = [cmd, ' [ $fail == "false" ]'];
end

cmd = ['(', cmd, ')'];


end


