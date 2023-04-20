OBJECTS = main.o utils.o gemm.o
LIBS = -lcublas

all: $(OBJECTS)
	nvcc $(LIBS) $(OBJECTS) -o main

gemm.o: gemm.cu
	nvcc $(LIBS) -c gemm.cu -o gemm.o 

%.o: %.cpp
	nvcc $(LIBS) -dc $< -o $@

clean:
	rm -f *.o main