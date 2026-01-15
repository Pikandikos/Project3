# Makefile for ANN project (C++17)
# - uses CXX (g++)
# - sources: src/*.cpp
# - headers: include/*.h
# - output: bin/search
# - supports: make, make run ARGS="...", make debug, make clean

CXX         := g++
BUILD_DIR   := bin
OBJ_DIR     := build

# Default optimisation flags; overridden by "make debug"
CXXFLAGS    := -O2 -std=c++17 -Wall -Wextra -Iinclude -pthread -MMD -MP
LDFLAGS     :=

SRC         := $(wildcard src/*.cpp)
OBJ         := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRC))
DEPS        := $(OBJ:.o=.d)

TARGET      := $(BUILD_DIR)/search

.PHONY: all run debug clean distclean fmt help

all: $(TARGET)

# Link
$(TARGET): $(OBJ) | $(BUILD_DIR)
	@echo "[LD] $@"
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Compile (object files under build/)
$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ensure dirs
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# run target: pass ARGS on command line, e.g. make run ARGS=" -d data/mnist -q data/queries -k 4"
run: $(TARGET)
	@echo "[RUN] $(TARGET) $(ARGS)"
	./$(TARGET) $(ARGS)

# clean up objects, deps and binary
clean:
	@echo "[CLEAN] removing build/ and $(TARGET)"
	@rm -rf $(OBJ_DIR) $(DEPS) $(TARGET)


# include dependency files if present
-include $(DEPS)