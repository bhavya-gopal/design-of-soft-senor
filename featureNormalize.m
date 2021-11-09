function [X_norm, mu, sigma] = featureNormalize(X_train)

X_norm = X_train;

 [m n] = size(X_train);
 mu = mean(X_train);
 sigma = std(X_train);
 X_norm = (X_train - mu(ones(m,1),:))./sigma(ones(m,1),:);



end
