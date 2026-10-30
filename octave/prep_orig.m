% Preprocessing copied from main_code.mlx (plots removed)
M = csvread('raw.csv'); raman_shift = M(1,:); raman_data = M(2:end,:);
label_select = dlmread('label.csv')';
fid=fopen('sid.txt'); C=textscan(fid,'%s'); fclose(fid); sample_id = string(C{1});
shift_indices = find((raman_shift >= 800 & raman_shift <= 2000));
selected_raman_data_filtered = raman_data(:, shift_indices);
order=3; frame_size=11;
for i=1:size(selected_raman_data_filtered,1)
  raman_FP_smoothed(i,:) = sgolayfilt(selected_raman_data_filtered(i,:), order, frame_size);
end
polyOrder=5; remove_baseline=zeros(size(raman_FP_smoothed));
for i=1:size(raman_FP_smoothed,1)
  current_spectrum = raman_FP_smoothed(i,:);
  [p_initial,~,mu_initial] = polyfit(raman_shift(shift_indices), current_spectrum, polyOrder);
  baseline_initial = polyval(p_initial, raman_shift(shift_indices), [], mu_initial);
  residuals_initial = current_spectrum - baseline_initial;
  dev_initial = sqrt(sum((residuals_initial-mean(residuals_initial)).^2)/length(residuals_initial));
  indices_to_remove = current_spectrum <= baseline_initial;
  y_to_remove = current_spectrum(indices_to_remove);
  x_to_remove = raman_shift(shift_indices(indices_to_remove));
  j=1; continue_iteration=true; dev=[];
  while continue_iteration
    [p_iteration,~,mu_iteration] = polyfit(x_to_remove,y_to_remove,polyOrder);
    y_fit_iteration = polyval(p_iteration,x_to_remove,[],mu_iteration);
    residuals_iteration = y_to_remove - y_fit_iteration;
    dev(j) = sqrt(sum((residuals_iteration-mean(residuals_iteration)).^2)/length(residuals_iteration));
    if j==1, continue_iteration = abs(dev(j)-dev_initial)/dev(j) > 0.05;
    else, continue_iteration = abs((dev(j)-dev(j-1))/dev(j)) > 0.05; end
    index_to_replace = y_to_remove >= y_fit_iteration;
    y_to_remove(index_to_replace) = y_fit_iteration(index_to_replace);
    j=j+1;
  end
  baseline_final = polyval(p_iteration, raman_shift(shift_indices), [], mu_iteration);
  remove_baseline(i,:) = raman_FP_smoothed(i,:) - baseline_final;
end
Z = zscore(remove_baseline')'; 
save('-binary','prep_oct.bin','Z','label_select'); csvwrite('Z_oct.csv',Z);
