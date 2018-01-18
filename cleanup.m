if isempty(currentFigures), currentFigures = []; end;
close(setdiff(findall(0, 'type', 'figure'), currentFigures))
clear mex
delete *.mexw64
[~,~,~] = rmdir('C:\Sumpurn\gpucoderdemo_object_detection\codegen','s');
clear C:\Sumpurn\gpucoderdemo_object_detection\getYolo.m
delete C:\Sumpurn\gpucoderdemo_object_detection\getYolo.m
clear C:\Sumpurn\gpucoderdemo_object_detection\getYoloFromCaffe.m
delete C:\Sumpurn\gpucoderdemo_object_detection\getYoloFromCaffe.m
clear C:\Sumpurn\gpucoderdemo_object_detection\yolo_from_caffe.m
delete C:\Sumpurn\gpucoderdemo_object_detection\yolo_from_caffe.m
delete C:\Sumpurn\gpucoderdemo_object_detection\downtown_short.mp4
delete C:\Sumpurn\gpucoderdemo_object_detection\main.cpp
delete C:\Sumpurn\gpucoderdemo_object_detection\make_win.bat
delete C:\Sumpurn\gpucoderdemo_object_detection\Makefile
clear
load old_workspace
delete old_workspace.mat
delete C:\Sumpurn\gpucoderdemo_object_detection\cleanup.m
cd C:\Sumpurn
rmdir('C:\Sumpurn\gpucoderdemo_object_detection','s');
