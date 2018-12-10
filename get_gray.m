function feature_image = get_gray( im, fparam, gparam )
%HOG features, from Piotr's Toolbox

if ~isfield(fparam, 'nOrients')
    fparam.nOrients = 9;
end

[im_height, im_width, num_im_chan, num_images] = size(im);
temp = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size));
feature_image = zeros(size(temp,1), size(temp,2), fparam.nDim, num_images);

for k = 1:num_images
    im_gray = rgb2gray(im(:,:,:,k));
    im_gray2= single(im_gray)/255;
    im_gray2 = im_gray2 - mean(im_gray2(:));
    feature_image(:,:,1,k) = imresize(im_gray2, [size(temp,1),size(temp,2)]);
end

end