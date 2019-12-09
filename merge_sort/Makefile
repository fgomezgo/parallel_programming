# C++ Compiler
CC = g++

# CUDA Compiler
CUDA = nvcc

# JAVA Compiler
JAVA = javac

# compiler flags:
#  -g    		adds debugging information to the executable file
#  -Wall 		turns on most, but not all, compiler warnings
#  -std=c++11 	specifies the c++ version to use (2011)
CFLAGS  = -g -Wall -std=c++11

# specific flags for openMP
OPENMPFLAGS = -fopenmp

# specific flags for tbb
TBBFLAGS = -ltbb

# base names for the executable files
TARGET = iterative_merge_sort

RTARGET = merge_sort

# openmp suffix for the executable file
OPENMPSUFFIX = _openmp

# tbb suffix for the executable file
TBBSUFFIX = _tbb

# cuda suffix for the executable file
CUDASUFFIX = _cuda

# fork join suffix for the java class
FORKJOINSUFFIX = _fork_join

all: $(RTARGET) $(RTARGET).class $(RTARGET)$(FORKJOINSUFFIX).class $(TARGET) $(TARGET).class $(TARGET)$(OPENMPSUFFIX) $(TARGET)$(TBBSUFFIX) $(TARGET)$(CUDASUFFIX)

$(RTARGET): $(RTARGET).cpp
	$(CC) $(CFLAGS) -o $(RTARGET) $(RTARGET).cpp

$(RTARGET).class: $(RTARGET).java
	$(JAVA) $(RTARGET).java

$(RTARGET)$(FORKJOINSUFFIX).class: $(RTARGET)$(FORKJOINSUFFIX).java
	$(JAVA) $(RTARGET)$(FORKJOINSUFFIX).java

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

$(TARGET).class: $(TARGET).java
	$(JAVA) $(TARGET).java

$(TARGET)$(OPENMPSUFFIX): $(TARGET)$(OPENMPSUFFIX).cpp
	$(CC) $(CFLAGS) $(OPENMPFLAGS) -o $(TARGET)$(OPENMPSUFFIX) $(TARGET)$(OPENMPSUFFIX).cpp

$(TARGET)$(TBBSUFFIX): $(TARGET)$(TBBSUFFIX).cpp
	$(CC) $(CFLAGS) $(TARGET)$(TBBSUFFIX).cpp $(TBBFLAGS) -o $(TARGET)$(TBBSUFFIX)

$(TARGET)$(CUDASUFFIX): $(TARGET)$(CUDASUFFIX).cu
	$(CUDA) -o $(TARGET)$(CUDASUFFIX) $(TARGET)$(CUDASUFFIX).cu

clean:
	$(RM) $(RTARGET) $(RTARGET).class $(RTARGET)$(FORKJOINSUFFIX).class $(TARGET) $(TARGET).class $(TARGET)$(OPENMPSUFFIX) $(TARGET)$(TBBSUFFIX) $(TARGET)$(CUDASUFFIX)