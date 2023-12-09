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
CC=nvc++
CC_FLAGS= -O3 -DNDEBUG -cuda -gpu=sm_80 
CC_SO_FLAGS= -fpic
# #
INCLUDE_DIR= -Iinclude
# #
#
# # CUDA math library directory:
NVCOMP_LIB_DIR= -L${NVHPC_ROOT}/compilers/lib64
# # CUDA include directory:
NVCOMP_INC_DIR= -I${NVHPC_ROOT}/compilers/include
# # CUDA linking libraries:
NVCOMP_LINK_LIB= -lcudanvhpc
# #
#
# # CUDA math library directory:
CUDAMATH_LIB_DIR= -L${NVHPC_ROOT}/math_libs/lib64
# # CUDA include directory:
CUDAMATH_INC_DIR= -I${NVHPC_ROOT}/math_libs/include
# # CUDA linking libraries:
CUDAMATH_LINK_LIB= -lcublas 
# #
#
# # CUDA library directory:
CUDA_LIB_DIR= -L${NVHPC_ROOT}/cuda/lib64
# # CUDA include directory:
CUDA_INC_DIR= -I${NVHPC_ROOT}/cuda/include
# # CUDA linking libraries:
CUDA_LINK_LIB= -lcudart
# #
# 
# #MPI library directory:
MPI_LIB_DIR= -L${MPI_ROOT}/lib
# # MPI include directory:
MPI_INC_DIR= -I${MPI_ROOT}/include
# # MPI linking libraries:
MPI_LINK_LIB= -lmpi
#
# # Thrust directory:
THRUST_INC_DIR= -I${NVHPC_ROOT}/compilers/include-stdpar/thrust
# #
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
SO = diag.so
#   
# # Object files:
OBJS = $(OBJ_DIR)/sp2_dnn_tcore.o $(OBJ_DIR)/tcore_hp_emulator.o \
       $(OBJ_DIR)/linalg_tools.o
# ##########################################################
#
# ## Compile ##
#
# # 
# Link compiled object files to target so:
so: $(OBJS)
	 $(CC) -shared $(CC_SO_FLAGS) $(CC_FLAGS) $(OBJS) -o $(SO) $(MPI_LIB_DIR) $(MPI_LINK_LIB) $(CUDA_LIB_DIR) \
                        $(CUDA_LINK_LIB) $(CUDAMATH_LIB_DIR) $(CUDAMATH_LINK_LIB) $(NVCOMP_LIB_DIR) $(NVCOMP_LINK_LIB)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	 $(CC) $(CC_SO_FLAGS) $(CC_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR) \
                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)


#Clean objects in object directory.
clean:
	$(RM) bin/* *.o




