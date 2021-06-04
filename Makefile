CC   := gcc
CXX  := g++
NVCC := nvcc
LD   := nvcc

CFLAGS   := -v -g -O2
CXXFLAGS := -v -g -O2
NVCFLAGS := -v -g -G -O2
LDFLAGS  := -v -g -G


.PHONY : all clean

all : prog

clean :
	rm -rfv prog *.o


prog : main.o kern.o kern_impl.o kern_gpu.o
	$(LD) $(LDFLAGS) -o $@ $^

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^
%.o : %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $^
