HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES = bias_activations.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./bias_activations.exe

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g -I ./includes -std=c++17

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) -I ../../../includes $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o
