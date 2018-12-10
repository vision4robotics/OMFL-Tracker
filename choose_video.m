function video_name = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (groundtruth.txt in the given path).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
%   
%   Modified by Fuling Lin (fuling.lin@outlook.com)
%
	%process path to make sure it's uniform
	if ispc(), base_path = strrep(base_path, '\', '/'); end
	if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders
    dirOutput = dir(fullfile(base_path, '*.txt'));
    contents = {dirOutput.name}';
	names = {};
    for k = 1:numel(contents),
        name = contents{k}(1:end-4);
        names{end+1} = name;
    end

%===================================================================
% uncomment following scripts if you test on the entire benchmark
%   names(strcmpi('Jogging', names)) = [];
% 	names(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
% 	
%     names(strcmpi('Skating2', names)) = [];
% 	names(end+1:end+2) = {'Skating2.1', 'Skating2.2'};
%===================================================================

	%no sub-folders found
	if isempty(names), video_name = []; return; end
	
	%choice GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');
	
	if isempty(choice),  %user cancelled
		video_name = [];
	else
		video_name = names{choice};
	end
	
end

