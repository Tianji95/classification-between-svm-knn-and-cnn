%% SVM
%for it is a over 2 classes problem, we use one-to-one methored to do
%classfication
t_start=clock;
nRep = 1000; % number of replicates

total_num = 0;
total_label_count = 10;
%  load data
%[X_validation, y_validation] = load_validation_data();

[X_train, y_train] = load_train_data();
[X_test, y_test] = load_test_data();
[P,N_train] = size(X_train);
[P,N_test] = size(X_test);

train_prediction = zeros(total_label_count, N_train);
test_prediction = zeros(total_label_count, N_test);

for label_num = 0:total_label_count - 2
    y_index_train = find(y_train == label_num);
    [y_P, y_N] = size(y_train(y_index_train));
    X_train_label = X_train(:,y_index_train);
    y_train_label = ones(y_P, y_N);
    
    y_index_test = find(y_test == label_num);
    [y_P, y_N] = size(y_test(y_index_test));
    X_test_label = X_test(:,y_index_test);
    y_test_label = ones(y_P, y_N);
    for comp_label_num = label_num + 1 : total_label_count - 1
        fprintf('now we are at %f, in the %f compare\n', label_num, comp_label_num);
        
        y_index_train_comp = find(y_train == comp_label_num);
        [y_P, y_N] = size(y_train(y_index_train_comp));
        X_train_label_comp = X_train(:,y_index_train_comp);
        y_train_label_comp = -ones(y_P, y_N);

        y_index_test_comp = find(y_test == comp_label_num);
        [y_P, y_N] = size(y_test(y_index_test_comp));
        X_test_label_comp = X_test(:,y_index_test_comp);
        y_test_label_comp = -ones(y_P, y_N);
        
        X_train_final_data = [X_train_label, X_train_label_comp];
        [w_g, num_sc] = svm(X_train_final_data, [y_train_label, y_train_label_comp]);
        
        fprintf('after svm\n');
        [P, N_train_label] = size(X_train_final_data);
        %train predict
        X_train_final_data = [ones(1,N_train_label);X_train_final_data];
        y_train_final_index = [y_index_train, y_index_train_comp];
        for col_index = 1:N_train_label
            thisPredict = int64(w_g'*X_train_final_data(:,col_index));
            if thisPredict == 1
                train_prediction(label_num+1, y_train_final_index(col_index)) = train_prediction(label_num+1, y_train_final_index(col_index)) + 1;
            else
                train_prediction(comp_label_num+1, y_train_final_index(col_index)) = train_prediction(comp_label_num+1, y_train_final_index(col_index)) + 1;
            end
        end
        
        
        %test predict
        y_test_final_index = [y_index_test, y_index_test_comp];
        X_test_final_data = [X_test_label, X_test_label_comp];
        [P, N_test_label] = size(X_test_final_data);
        X_test_final_data = [ones(1, N_test_label); X_test_final_data];
        for col_index = 1:N_test_label
            thisPredict = int64(w_g'*X_test_final_data(:,col_index));
            if thisPredict == 1
                test_prediction(label_num+1, y_test_final_index(col_index)) = test_prediction(label_num+1, y_test_final_index(col_index)) + 1;
            else
                test_prediction(comp_label_num+1, y_test_final_index(col_index)) = test_prediction(comp_label_num+1, y_test_final_index(col_index)) + 1;
            end
        end
        
    end 
end

%compute the final prediction and error profile is shutting down.

E_train = 0;
E_test = 0;
for col_index = 1:N_train
    [predict, predict_idx] = max(train_prediction);
     if y_train(col_index)~=predict_idx(col_index) - 1
        E_train = E_train + 1;
     end

end

for col_index = 1: N_test
    [predict, predict_idx] = max(test_prediction);
    if y_test(col_index)~= predict_idx(col_index) - 1
        E_test = E_test + 1;
    end 
end

E_test = E_test / N_test;
E_train = E_train / N_train;
% Compute training, testing error
% Sum up number of support vectors

t_end = clock;
fprintf('E_train is %f, E_test is %f., number of support vectors is %f, using time:%f\n', E_train, E_test, num_sc, etime(t_end, t_start));


%% KNN
t_start=clock;

K = 3;
[X_train, y_train] = load_train_data();
[X_test, y_test] = load_test_data();

y_predict = knn(X_test, X_train, y_train, K);
E_test = length(find((y_predict - y_test)~=0));

t_end = clock;
fprintf('KNN algorithm, K is %f, E_test is %f, using time:%f\n',K, E_test, etime(t_end, t_start));

