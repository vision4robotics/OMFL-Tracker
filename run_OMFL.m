%   This function runs the BACFIF tracker on the video specified in "seq".
%   This function is based on the BACF tracker. 
%   Modified by Fuling Lin (fuling.lin@outlook.com)

function results = run_OMFL(seq, video_path, lr)

params.video_path = video_path;

%   HOG feature parameters
hog_params.nDim   = 31;
%   CN feature parameters
cn_params.nDim = 11;
%   Gray feature parameters
gray_params.nDim = 1;
%   Saliency feature parameters
saliency_params.nDim = 3;
%   Global feature parameters 
params.t_features = {struct('getFeature_fhog',@get_fhog,...
                            'getFeature_cn',@get_cn,...
                            'getFeature_gray',@get_gray,...
                            'getFeature_saliency',@get_saliency,...
                            'hog_params',hog_params,...
                            'cn_params',cn_params,...
                            'gray_params',gray_params,...
                            'saliency_params',saliency_params)};
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       = lr;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.newton_iterations     = 50;       % number of Newton's iteration to maximize the detection scores

%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);

params.s_frames = seq.s_frames; 
params.no_fram  = seq.en_frame - seq.st_frame + 1; 
params.seq_st_frame = seq.st_frame;
params.seq_en_frame = seq.en_frame; 

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.admm_iterations = 2;
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;

%   Run the main function
results = OMFL_optimized(params);