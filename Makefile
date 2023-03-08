# Makefiel for SRCNN git cloned mod.
# by Raphael Kim

CPP = gcc
CXX = g++
AR  = ar

OPENCV_INCS := `pkg-config --cflags opencv4`
OPENCV_LIBS := `pkg-config --libs opencv4`

SRC_PATH = src
OBJ_PATH = obj
BIN_PATH = bin
TARGET   = srcnn

SRCS += $(SRC_PATH)/frawscale.cpp
SRCS += $(SRC_PATH)/libsrcnn.cpp
SRCS += $(SRC_PATH)/tick.cpp
SRCS += $(SRC_PATH)/srcnn.cpp
OBJS = $(SRCS:$(SRC_PATH)/%.cpp=$(OBJ_PATH)/%.o)

CFLAGS  = -mtune=native -fopenmp
CFLAGS += -I$(SRC_PATH)
CFLAGS += $(OPENCV_INCS)

# Static build may require static-configured openCV.
LFLAGS  =
LFLAGS += $(OPENCV_LIBS)
LFLAGS += -static-libgcc -static-libstdc++
LFLAGS += -s -ffast-math -O3

all: prepare $(BIN_PATH)/$(TARGET)

prepare:
	@mkdir -p $(OBJ_PATH)
	@mkdir -p $(BIN_PATH)

clean:
	@rm -rf $(OBJ_PATH)/*.o
	@rm -rf $(BIN_PATH)/$(TARGET)

$(OBJS): $(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@echo "Compiling $< ..."
	@$(CXX) $(CFLAGS) -c $< -o $@

$(BIN_PATH)/$(TARGET): $(OBJS)
	@echo "Linking $@ ..."
	@$(CXX) $(OBJ_PATH)/*.o $(CFLAGS) $(LFLAGS) -o $@
