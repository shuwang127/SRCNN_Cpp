im_gnd = imread('test_gnd.jpg');
up_scale = 2;
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
imwrite(im_l, 'test.jpg');