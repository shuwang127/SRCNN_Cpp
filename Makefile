# Makefiel for SRCNN.

PNAME = srcnn

CPP = gcc
CXX = g++
AR  = ar
RM  = rm -f

OPENCV_INCS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

TARGET   = $(PNAME)

SRCS = $(PNAME).cpp
OBJS = $(SRCS:%.cpp=%.o)

CFLAGS = -Wall
CFLAGS += $(OPENCV_INCS)

LFLAGS = $(OPENCV_LIBS)
LFLAGS += -O3 -ffast-math -s

all: $(TARGET)

clean:
	$(RM) $(TARGET) *.o

$(OBJS): %.o: %.cpp
	@echo "Compiling $< ..."
	$(CXX) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	@echo "Linking $@ ..."
	$(CXX) $(CFLAGS) *.o $(LFLAGS) -o $@
