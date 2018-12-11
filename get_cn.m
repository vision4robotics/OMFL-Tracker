

function feature_image = get_cn( im, fparam, gparam )
load('w2c.mat');
[im_height, im_width, num_im_chan, num_images] = size(im);
temp = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size));
feature_image = zeros(size(temp,1), size(temp,2), fparam.nDim, num_images);

for k = 1:num_images
    im_temp = imresize(im(:,:,:,k), [size(temp,1) size(temp,2)]); 
    cn_image = im2c(double(im_temp(:,:,:,k)),w2c,-2);
    feature_image(:,:,:,k) = cn_image;
end
end