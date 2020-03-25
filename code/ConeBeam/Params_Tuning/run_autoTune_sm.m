clear all


plainParamsFile = '/scratch/rice/s/smajee/cone_beam_project/Results/code/plainParams/plainParams.sh'
masterFile = '/scratch/rice/s/smajee/cone_beam_project/Results/control/Inversion/QGGMRF/4D_replica_000029_master.txt'

verbosity = 0;

paramName = 'Delta_v_r';

% these need to have same length
searchRadius_list =     [10 5 1 0.1 ];
numPointsGrid_list =    [5 5 5 5];

% searchRadius_list =     [2  0.5   0.1];
% numPointsGrid_list =    [7  7   7   ];

run autoTune_iterative_run

