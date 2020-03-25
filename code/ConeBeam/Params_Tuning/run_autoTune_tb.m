clear all


plainParamsFile = '/scratch/snyder/t/tbalke/coneBeam/code/plainParams/plainParams.sh'
masterFile = '/scratch/snyder/t/tbalke/coneBeam/control/Inversion/QGGMRF/master.txt'

verbosity = 0;

paramName = 'Delta_v_d0';

% these need to have same length
searchRadius_list =     [2 0.5  ];
numPointsGrid_list =    [7  7];

% searchRadius_list =     [2  0.5   0.1];
% numPointsGrid_list =    [7  7   7   ];

run autoTune_iterative_run

