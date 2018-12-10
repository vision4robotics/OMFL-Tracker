function out = get_feature_map(im_patch, w2c)

% Extracts the given features from the image patch. w2c is the
% Color Names matrix, if used.

% the names of the features that can be used


% the dimension of the valid features
        feature_levels = 10;


        num_valid_features = length(valid_features);
        used_features = false(num_valid_features, 1);

        out(:,:,level+(1:11)) = im2c(single(im_patch), w2c, -2);

        level = level + feature_levels(2);
    end
end