CC   := gcc
CXX  := g++
NVCC := nvcc
LD   := nvcc

CFLAGS   := -v -O2 -Wall -Werror
CXXFLAGS := -v -O2 -Wall -Werror
NVCFLAGS := -v -O2 -Werror all-warnings
LDFLAGS  := -v

INCLUDE := -Iinclude/

EXECNAME := cuda-magnetic-pendulum
OBJFILES := src/main.o \
	src/kern.o src/kern_state.o src/kern_impl.o src/kern_gpu.o \
	src/img_color.o src/img_write.o


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
