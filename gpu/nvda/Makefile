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
CXX:=${CXX}
CXX_FLAGS:=${CXX_FLAGS}
CXX_SO_FLAGS= -fpic
EXTRA_CXX_FLAGS= --diag_suppress=bad_macro_redef
$(info $$CXX is [${CXX}])
# #
#
# # CUDA math library directory:
NVCOMP_LIB_DIR= -L${NVHPC_ROOT}/compilers/lib64
# # CUDA include directory:
NVCOMP_INC_DIR= -I${NVHPC_ROOT}/compilers/include
# # CUDA linking libraries:
NVCOMP_LINK_LIB= -lcudanvhpc
# #

INCLUDE_DIR= -Iinclude
#
# # CUDA math library directory:
CUDAMATH_LIB_DIR= -L${NVHPC_ROOT}/math_libs/lib64
# # CUDA include directory:
CUDAMATH_INC_DIR= -I${NVHPC_ROOT}/math_libs/include
# # CUDA linking libraries:
CUDAMATH_LINK_LIB= -lcublas -lcusolver 
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
#INCLUDE_DIR = include
#
# ##########################################################
#
# ## Make variables ##
#
# # Target executable name:
SO = libnvda.so
#   

# # Make object directories:
OBJ_DIRS= $(OBJ_DIR)/lib $(OBJ_DIR)/dnnsp2 $(OBJ_DIR)/diag $(OBJ_DIR)/nlist\
		 $(OBJ_DIR)/fastcheby $(OBJ_DIR)/mlsp2 $(OBJ_DIR)/test
$(shell   mkdir -p $(OBJ_DIRS)) 

# # Object files:

OBJDIAG     =  $(OBJ_DIR)/diag/diag.o\

#OBJNLIST   =  $(OBJ_DIR)/nlist/*.o\

OBJCHEBY   =  $(OBJ_DIR)/fastcheby/*.o\

#OBJINVOVLP =  $(OBJ_DIR)/invovlp/*.o\

#OBJMLSP2  =  $(OBJ_DIR)/mlsp2/*.o\

#OBJSP2      =    #$(OBJ_DIR)/dnnsp2/linalg_tools.o\
	         
OBJSP2      =    $(OBJ_DIR)/dnnsp2/dnnsp2.o\
                 $(OBJ_DIR)/dnnsp2/tcore_hp_emulator.o\

#OBJNLIST=  $(OBJ_DIR)/nlist/utils.o\
	   $(OBJ_DIR)/nlist/nlist.o\

OBJMAIN=   $(OBJ_DIR)/lib/libnvda.o

OBJS = $(OBJSP2) $(OBJDIAG) $(OBJNLIST) $(OBJCHEBY) \
	$(OBJMLSP2) $(OBJMAIN)


# ##########################################################
#
# ## Compile ##
#
# # 


# Link compiled object files to target so:
so: $(OBJS)
	$(CXX) -shared $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) $(OBJS) -o $(SO) $(MPI_LIB_DIR) $(MPI_LINK_LIB) $(CUDA_LIB_DIR)\
                $(CUDA_LINK_LIB) $(CUDAMATH_LIB_DIR) $(CUDAMATH_LINK_LIB) $(NVCOMP_LIB_DIR) $(NVCOMP_LINK_LIB)

$(OBJ_DIR)/diag/%.o: $(SRC_DIR)/diag/%.cu
	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR) $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)

$(OBJ_DIR)/dnnsp2/%.o: $(SRC_DIR)/dnnsp2/%.cu
	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR) $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB) 

#$(OBJ_DIR)/nlist/%.o: $(SRC_DIR)/nlist/%.cu
#	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR)\
                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)

$(OBJ_DIR)/fastcheby/%.o: $(SRC_DIR)/fastcheby/%.cu
	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR)\
                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)

#$(OBJ_DIR)/invovlp/%.o: $(SRC_DIR)/invovlp/%.cu
#	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR)\
#                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)

#$(OBJ_DIR)/mlsp2/%.o: $(SRC_DIR)/mlsp2/%.cu
#	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR)\
                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)

$(OBJ_DIR)/lib/%.o: $(SRC_DIR)/lib/%.cu
	$(CXX) $(CXX_SO_FLAGS) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) -c $< $(INCLUDE_DIR) $(MPI_INC_DIR) $(CUDA_INC_DIR)\
                      $(CUDAMATH_INC_DIR) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIB)



#Clean objects in object directory.
clean:
	$(RM) -r bin
	#$(RM) bin/* *.o




