function [ indexFull ] = subIndex2FullIndex( indexSub, NSub, NFull )
% Used to compute the indices of the full set of views from an index of the
% smaller subset of views
% Choose 0 <= indexSub < NSub <= NFull
% Uses 0-indexing (0, 1, ..., N-1).

indexFull = floor( ...
    indexSub / NSub * NFull...
    );




end

