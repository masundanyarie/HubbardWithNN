src1=overlap.cpp
src2=bosehubbard1.cpp
src3=bosehubbard2.cpp
hed=src/*.hpp
cmd=icpc
lib=-std=c++0x -fno-alias -openmp -O3 -ipo -no-prec-div -xHost -mkl
#lib=-std=c++0x -fno-alias -qopenmp -O3 -ipo -no-prec-div -xHost -mkl	
all: target1 target2 target3
target1:$(src1) $(hed)
	$(cmd) $(lib) $(src1) -o overlap.o
target2:$(src2) $(hed)
	$(cmd) $(lib) $(src2) -o bose_hubbard1.o
target3:$(src3) $(hed)
	$(cmd) $(lib) $(src3) -o bose_hubbard2.o
