################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../ImageHandle.cpp \
../LayerTransform.cpp \
../neuro_encode_decode.cpp \
../neuro_vision.cpp \
../neuro_vision_utility.cpp \
../simple_test_patterns.cpp \
../simple_test_weights.cpp 

CU_SRCS += \
../BPNeuralNetwork.cu 

CU_DEPS += \
./BPNeuralNetwork.d 

OBJS += \
./BPNeuralNetwork.o \
./ImageHandle.o \
./LayerTransform.o \
./neuro_encode_decode.o \
./neuro_vision.o \
./neuro_vision_utility.o \
./simple_test_patterns.o \
./simple_test_weights.o 

CPP_DEPS += \
./ImageHandle.d \
./LayerTransform.d \
./neuro_encode_decode.d \
./neuro_vision.d \
./neuro_vision_utility.d \
./simple_test_patterns.d \
./simple_test_weights.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -G -g -O0 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_11,code=compute_11 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -G -g -O0 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


