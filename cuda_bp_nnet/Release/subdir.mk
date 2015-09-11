################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BPNeuralNetwork.cu 

CPP_SRCS += \
../ImageHandle.cpp \
../LayerTransform.cpp \
../cublas_bp_nnet_main.cpp \
../cublas_bp_nnet_utility.cpp \
../neuro_encode_decode.cpp \
../simple_test_patterns.cpp \
../simple_test_weights.cpp 

OBJS += \
./BPNeuralNetwork.o \
./ImageHandle.o \
./LayerTransform.o \
./cublas_bp_nnet_main.o \
./cublas_bp_nnet_utility.o \
./neuro_encode_decode.o \
./simple_test_patterns.o \
./simple_test_weights.o 

CU_DEPS += \
./BPNeuralNetwork.d 

CPP_DEPS += \
./ImageHandle.d \
./LayerTransform.d \
./cublas_bp_nnet_main.d \
./cublas_bp_nnet_utility.d \
./neuro_encode_decode.d \
./simple_test_patterns.d \
./simple_test_weights.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


