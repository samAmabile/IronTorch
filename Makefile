CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -fopenmp -flto=auto -funroll-loops -I.
LDFLAGS = -fopenmp -flto=auto


SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = itorch
DATA_DIR = data
all: $(TARGET)

$(TARGET): $(OBJECTS) | $(DATA_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

debug: CXXFLAGS = -std=c++20 -Wall -Wextra -O1 -g -fsanitize=address -march=native -fopenmp -funroll-loops -I.
debug: LDFLAGS = -fopenmp -fsanitize=address
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
