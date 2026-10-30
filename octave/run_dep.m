function out = run_dep(model, task, K, pcs, mode, seed)
% Faithful core of deposited pcasvm.m / pcalda_k.m (figures removed).
% mode: 'deposited' = original line trainIdx(removeIdx)=[]; 'fixed' = intended sample exclusion
load('prep_oct.bin'); data = Z; label = label_select';
fid=fopen('sid.txt'); C=textscan(fid,'%s'); fclose(fid); simple_id = C{1};
if strcmp(task,'binary'), label = 1 + (label>1); end
rand('seed',seed); randn('seed',seed); rand('state',seed); randn('state',seed);
idx = randperm(size(data,1));
shuffledData = data(idx,:); shuffledLabels = label(idx); shuffledSimpleID = simple_id(idx);
[COEFF, SCORE] = pca(shuffledData);
cv = cvpartition(shuffledLabels, 'KFold', K, 'Stratify', true);
acc = zeros(numel(pcs),1); leak = zeros(K,1); ntr = zeros(K,1); misal = zeros(K,1);
for i=1:K
  trainIdx = training(cv,i); testIdx = test(cv,i);
  testSimpleIDs = shuffledSimpleID(testIdx);
  removeIdx = ismember(shuffledSimpleID(trainIdx), testSimpleIDs);
  if strcmp(mode,'deposited'), trainIdx(removeIdx) = [];
  else, trainIdx = trainIdx & ~ismember(shuffledSimpleID, testSimpleIDs); end
  T{i}=trainIdx;
  sel = find(trainIdx); tst = find(testIdx);
  leak(i) = numel(intersect(sel, tst)); ntr(i)=numel(sel);
  misal(i) = sum(ismember(shuffledSimpleID(sel), testSimpleIDs));
end
for p=1:numel(pcs)
  nPCs = pcs(p); data_PCA = SCORE(:,1:nPCs); accSum=0;
  for i=1:K
    trainIdx = T{i}; testIdx = test(cv,i);
    if strcmp(model,'svm')
      Xtr = data_PCA(trainIdx,:); r = randperm(size(Xtr,1), min(300,size(Xtr,1)));
      D = sqrt(max(0, sum(Xtr(r,:).^2,2) + sum(Xtr(r,:).^2,2)' - 2*Xtr(r,:)*Xtr(r,:)')); ks = median(D(D>0));
      mdl = fitcecoc(Xtr, shuffledLabels(trainIdx), 'Learners', templateSVM('KernelFunction','rbf','BoxConstraint',10,'KernelScale',ks), 'Coding','onevsone');
    else
      mdl = fitcdiscr(data_PCA(trainIdx,:), shuffledLabels(trainIdx), 'DiscrimType','pseudoLinear');
    end
    pred = predict(mdl, data_PCA(testIdx,:));
    accSum = accSum + sum(pred == shuffledLabels(testIdx))/sum(testIdx);
  end
  acc(p) = accSum/K;
end
out.acc=acc; out.leak=leak; out.ntr=ntr; out.sameSampleInTrain=misal;
end
