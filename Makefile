# Copyright 2017 The MathWorks, Inc.
ifeq ($(OS),Windows_NT)
    CODEGEN_LIB := cnnbuild.lib
    TIFF_LIB :=
    CUDNN_LIB_PATH := $(NVIDIA_CUDNN)\lib\x64
    OPENCV_FLAG :=  -I"$(OPENCV_DIR)\include" \
                    -L"$(OPENCV_DIR)\x64\vc12\lib" \
                    -lopencv_imgproc249 -lopencv_core249 -lopencv_highgui249 \
                    -lopencv_video249 -lopencv_objdetect249
else
    CODEGEN_LIB := cnnbuild.a
    TIFF_LIB := -ltiff
    CUDNN_LIB_PATH := $(NVIDIA_CUDNN)/lib64
    OPENCV_FLAG := `pkg-config --cflags --libs opencv`
endif

all:	
	nvcc -o object_detection_exe main.cpp \
         -Icodegen codegen/${CODEGEN_LIB} \
         -lcublas $(TIFF_LIB) \
         -I"$(NVIDIA_CUDNN)/include" -L"$(CUDNN_LIB_PATH)" -lcudnn \
         $(OPENCV_FLAG) \
         -Wno-deprecated-gpu-targets
