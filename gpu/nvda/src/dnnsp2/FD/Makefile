## USER SPECIFIC DIRECTORIES ##
#
# ##CUDA directory:
#CUDA_ROOT_DIR=. #/usr/local/cuda
#
# ##########################################################
#
# ## CC COMPILER OPTIONS ##
#
# ## CC compiler options:
# CC=gcc
# CC_FLAGS=
# CC_LIBS=
#
# ##########################################################
#
# ## NVCC COMPILER OPTIONS ##
#
# # NVCC compiler options:
NVCC=nvcc
#NVCC_FLAGS= 
#NVCC_LIBS=

# # CUDA library directory:
CUDA_LIB_DIR= -Llib
# # CUDA include directory:
CUDA_INC_DIR= -Iinclude
# # CUDA linking libraries:
CUDA_LINK_LIBS= -lcublas -lcusolver 
#
# ##########################################################
#
# ## Project file structure ##
#
# # Source file directory:
SRC_DIR = src
#
# # Object file directory:
OBJ_DIR = bin
#
# # Include header file diretory:
INC_DIR = include
#
# ##########################################################
#
# ## Make variables ##
#
# # Target executable name:
EXE = finite_diff
#
# # Object files:
OBJS = $(OBJ_DIR)/*.o
# ##########################################################
#
# ## Compile ##
#
# # Link CUDA compiled object files to target executable:
finite_diff : lib/libtcore.a bin/sp2_dnn_tcore.o bin/main.o  
	nvcc -arch sm_70 bin/sp2_dnn_tcore.o bin/main.o -o finite_diff $(CUDA_LINK_LIBS) -Llib -ltcore

bin/main.o : src/main.cu 
	nvcc -arch sm_70 -c -std=c++11 src/main.cu -o bin/main.o -I$(INC_DIR)

bin/sp2_dnn_tcore.o : src/sp2_dnn_tcore.cu
	nvcc -arch sm_70 -c -std=c++11 src/sp2_dnn_tcore.cu -o bin/sp2_dnn_tcore.o -I$(INC_DIR)

lib/libtcore.a : bin/tcore_hp_emulator.o bin/linalg_tools.o 
	ar rcs lib/libtcore.a bin/tcore_hp_emulator.o bin/linalg_tools.o

#libs: lib/libtcore.a

bin/tcore_hp_emulator.o : src/tcore_hp_emulator.cu
	nvcc -arch sm_70 -c -std=c++11 src/tcore_hp_emulator.cu -o bin/tcore_hp_emulator.o -I$(INC_DIR) 

bin/linalg_tools.o : src/linalg_tools.cu
	nvcc -arch sm_70 -c -std=c++11 src/linalg_tools.cu -o bin/linalg_tools.o -I$(INC_DIR) 
 


#Clean objects in object directory.
clean:
	$(RM) bin/* *.o lib/*a
