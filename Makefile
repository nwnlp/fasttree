CXX = g++
CXXFLAGS = -O2 -fPIC -std=c++0x -lstdc++ -march=native -fopenmp
MAIN = rf
FILES = load_data.cpp attribute_list.cpp tree.cpp tree_node.cpp tools.cpp class_list.cpp rf.cpp
SRCS = $(FILES:%.cpp=src/%.cpp)
HEADERS = $(FILES:%.cpp=src/%.h)

all: $(MAIN)

rf: src/main.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)

