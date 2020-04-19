function scan_norm = normalize_scan(scan, dark, blank)

% v = (l - d) / (b - d)

scan_norm = 0*scan;

lambda_corr = scan - dark;
blank_corr = blank - dark;
    
scan_norm(lambda_corr<=0 | blank_corr<=0) = 0;
scan_norm(lambda_corr> 0 & blank_corr> 0) = lambda_corr(lambda_corr> 0 & blank_corr> 0) ./ blank_corr(lambda_corr> 0 & blank_corr> 0);
% scan_norm(scan_norm>1) = 1;