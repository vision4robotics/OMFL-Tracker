%   Loads relevant information of UAV123 in the given path.
%   Fuling Lin, 2018

function seq = load_video_info_UAV123(video_name, video_path, ground_truth_path)

    seqs=configSeqs(video_path);
    
    i=1;
    while ~strcmpi(seqs{i}.name,video_name)
            i=i+1;
    end
    
    seq.VidName = seqs{i}.name;
    seq.video_path = seqs{i}.path;
    seq.st_frame = seqs{i}.startFrame;
    seq.en_frame = seqs{i}.endFrame;
    
    seq.ground_truth_fileName = seq.VidName;
    ground_truth = dlmread([ground_truth_path '\' seq.ground_truth_fileName '.txt']);
    
    seq.ground_truth = ground_truth;
    seq.len = seq.en_frame-seq.st_frame+1;
    seq.init_rect = ground_truth(1,:);
    
    img_path = seq.video_path;
    img_files = dir(fullfile(img_path, '*.jpg'));
    img_files = {img_files.name};
    seq.s_frames_temp = cellstr(img_files);
    seq.s_frames = seq.s_frames_temp(1, seq.st_frame : seq.en_frame);