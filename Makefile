CC   := gcc
CXX  := g++
NVCC := nvcc
LD   := nvcc

CFLAGS   := -v -g -O2
CXXFLAGS := -v -g -O2
NVCFLAGS := -v -g -G -O2
LDFLAGS  := -v -g -G

INCLUDE := -Iinclude/

EXECNAME := cuda-magnetic-pendulum
OBJFILES := src/main.o src/kern.o src/kern_impl.o src/kern_gpu.o


.PHONY : all clean

all : $(EXECNAME)

clean :
	rm -rfv $(EXECNAME) $(OBJFILES)


$(EXECNAME) : $(OBJFILES)
	$(LD) $(LDFLAGS) -o $@ $^

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $@ $^
%.o : %.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -c -o $@ $^
