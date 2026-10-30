function job_all(model,task,mode)
pkg load statistics;
pcs = unique([1:5:300 18 44 178 182]);
o = run_dep(model,task,20,pcs,mode,1);
csvwrite(sprintf('sweep_%s_%s_%s.csv',model,task,mode),[pcs' o.acc]);
csvwrite(sprintf('leak_%s_%s_%s.csv',model,task,mode),[o.leak o.ntr o.sameSampleInTrain]);
if strcmp(model,'svm') && strcmp(task,'binary'), k=44; elseif strcmp(model,'svm'), k=18; elseif strcmp(task,'binary'), k=182; else, k=178; end
A=[]; for sd=1:10, o=run_dep(model,task,20,k,mode,sd); A=[A; sd o.acc mean(o.leak) mean(o.ntr)]; csvwrite(sprintf('seeds_%s_%s_%s.csv',model,task,mode),A); end
end
