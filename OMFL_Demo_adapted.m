%   This script runs the original implementation of Background Aware Correlation Filters (BACF) for visual tracking.
%   the code is tested for Mac, Windows and Linux- you may need to compile
%   some of the mex files.
%   Paper is published in ICCV 2017- Italy
%   Some functions are borrowed from other papers (SRDCF, CCOT, KCF, etc)- and
%   their copyright belongs to the paper's authors.
%   copyright- Hamed Kiani (CMU, RI, 2017)
%   contact me: hamedkg@gmail.com

%   This function runs the OMFL tracker on the video specified in "seq".
%   This function is based on the BACF tracker. 
%   Modified by Fuling Lin (fuling.lin@outlook.com)

function OMFL_Demo_adapted(~)
    close all;
    clear;
    clc;
    
%   Load video information    
    video_path_UAV123 = '.\UAV123_10fps\data_seq';
    ground_truth_path_UAV123 = '.\UAV123_10fps\anno';
    video_name = choose_video(ground_truth_path_UAV123);
    seq = load_video_info_UAV123(video_name, video_path_UAV123, ground_truth_path_UAV123);
    video_path = seq.video_path;
    ground_truth = seq.ground_truth;

    gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
    %   Run OMFL - main function
    learning_rate = 0.013;  %   you can use different learning rate for different benchmarks.
    results       = run_OMFL(seq, video_path, learning_rate);

    %   compute the OP
    pd_boxes = results.res;
    pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
    OP = zeros(size(gt_boxes,1),1);
    for i=1:size(gt_boxes,1)
        b_gt = gt_boxes(i,:);
        b_pd = pd_boxes(i,:);
        OP(i) = computePascalScore(b_gt,b_pd);
    end
    OP_vid = sum(OP >= 0.5) / numel(OP);
    FPS_vid = results.fps;
    display([video_name  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(OP_vid)]);
    
    result_name = video_name;
    OMFL = results;
    
    savedir = './results/';
    if ~exist(savedir,'dir')
        mkdir(savedir);
    end   
    save([savedir,result_name],'OMFL');
    precision_plot(results.res,ground_truth,video_name,savedir,1);
end
