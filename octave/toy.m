ids   = {'S1','S1','S2','S2','S3','S3','S4','S4','S5','S5'}';
testIdx  = logical([1 0 0 0 0 0 1 0 0 0])';   % rows 1 (S1) and 7 (S4) are test
trainIdx = ~testIdx;
removeIdx = ismember(ids(trainIdx), ids(testIdx));  % relative to the 8 training rows
disp('removeIdx (length 8):'); disp(removeIdx')
t2 = trainIdx; t2(removeIdx) = [];
disp('trainIdx after deletion:'); disp(t2'); printf('length %d\n', numel(t2));
X = (1:10)'; disp('rows used for training:'); disp(X(t2)')
disp('intended rows:'); disp(find(trainIdx & ~ismember(ids, ids(testIdx)))')
