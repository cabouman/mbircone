function [ ] = consensus_iterations( masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(genpath(fullfile(mfilepath,'MatlabRoutines')));

% This always has to be added: ConeBeam read/write routines
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Handling args
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

control_4D = read_control_4D(masterFile, plainParamsFile);
crossCheck_control_4D(control_4D);

save('control_4D.mat','control_4D');
print_control_4D( control_4D );
forceStopFlag = control_4D.params_consensus.forceStopFlag;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if control_4D.params_consensus.clearBinaryFolder_start
    clear_BinaryFolder( control_4D, plainParamsFile );
end

if control_4D.params_consensus.isInitializeRecon
    init_command = [control_4D.inversionConfig.recon_Script_4D, ' ', masterFile, ' ', control_4D.inversionConfig.recon_4D_mode_init];
    fprintf('Starting Init Command: %s\n',init_command );
    tic
    system(init_command);
    stats.time.initRecon = toc
end

if control_4D.params_consensus.isInitializeVars
    fprintf('Initialize variables...\n' );
    stats.time.initVar = consensus_init_variables( control_4D );
    fprintf('Initialized variables.\n' );
end

if control_4D.params_consensus.isConsensusIters == 1
    disp(' === set_proxmapMode  ===');
    set_proxmapMode( control_4D, plainParamsFile );
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Set Sigma Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       sigma_lambda <- sigma_ref / sqrt(inversion_priorWeight)
%       sigma_denoisers = sigma_ref * sqrt(priorWeight_denoisers)
if control_4D.params_consensus.isConsensusIters == 1
    disp(' === set_sigma  ===');
    [~, control_4D] = set_sigma( control_4D, plainParamsFile );
    printStruct(control_4D.params_consensus, 'params_consensus');
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  ADMM Iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if control_4D.params_consensus.isConsensusIters == 1
    for itnumber = 0:control_4D.params_consensus.MaxIter

        params_consensus_temp = readParams_consensus( masterFile, plainParamsFile );
        forceStopFlag = params_consensus_temp.forceStopFlag;
        if forceStopFlag==1
            break;
        end

        control_4D = consensus_step(control_4D, itnumber);

        [control_4D, stop_condition] = consensus_stats(control_4D, itnumber);

        control_4D = consensus_visualization(control_4D, itnumber);

        if isfield(control_4D, 'stats')
            printStruct(control_4D.stats, 'stats', 1);
            fileID = fopen( control_4D.statsFile, 'w');
            printStruct(control_4D.stats, 'stats', 1, '', fileID);
            fclose(fileID);
        end

        if stop_condition==1
            break;
        end

    end

    save('control_4D.mat','control_4D');
    control_4D = consensus_convergance_stats(control_4D);    
end



if control_4D.params_consensus.clearBinaryFiles_end
    clear_BinaryFiles( control_4D, plainParamsFile );
end

if isfield(control_4D, 'stats')
    printStruct(control_4D.stats, 'stats', 1);
    fileID = fopen( control_4D.statsFile, 'w');
    printStruct(control_4D.stats, 'stats', 1, '', fileID);
    fclose(fileID);
end

save('control_4D.mat','control_4D');

return


