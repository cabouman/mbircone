function [ pid_lb, pid_ub ] = generateProcessIndexList(maxNumProcesses, numFiles)

numProcesses = min(maxNumProcesses, numFiles);

for i = 0:numProcesses-1
	lb(i+1) = floor(i     * numFiles / numProcesses) + 1;
	ub(i+1) = floor((i+1) * numFiles / numProcesses) - 1 + 1;
end

pid_lb = lb';
pid_ub = ub';


end