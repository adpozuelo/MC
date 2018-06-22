CCX = icpc
CCXFLAGS = -std=c++11 -fast
LDFLAGS = -std=c++11
MKLFLAGS = -mkl=parallel

NVCC = nvcc
CUDAFLAGS= -std=c++11 -ccbin=icpc
LIBS= -lcudart
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

HPATH = include
SRCPATH = src
OBJPATH = obj
BINPATH = bin

OBJECTS = $(OBJPATH)/main.o $(OBJPATH)/input.o $(OBJPATH)/energy.o $(OBJPATH)/util.o $(OBJPATH)/output.o \
			$(OBJPATH)/thermo.o $(OBJPATH)/moves.o $(OBJPATH)/gpu.o

all: $(OBJECTS)
	${CCX} ${LDFLAGS} ${MKLFLAGS} ${LIBS} ${LIBDIRS} ${INCDIRS} -o $(BINPATH)/mc.exe $(OBJECTS)

$(OBJPATH)/main.o: $(SRCPATH)/main.cpp $(HPATH)/io.h $(SRCPATH)/input.cpp $(HPATH)/energy.h $(SRCPATH)/energy.cpp \
					$(HPATH)/util.h $(SRCPATH)/util.cpp $(SRCPATH)/output.cpp $(SRCPATH)/thermo.cpp $(HPATH)/thermo.h \
					$(SRCPATH)/moves.cpp $(HPATH)/moves.h $(SRCPATH)/gpu.cu $(HPATH)/gpu.h
	${CCX} ${CCXFLAGS} -c -o $(OBJPATH)/main.o $(SRCPATH)/main.cpp

$(OBJPATH)/input.o: $(SRCPATH)/input.cpp $(HPATH)/io.h
	${CCX} ${CCXFLAGS} -c -o $(OBJPATH)/input.o $(SRCPATH)/input.cpp

$(OBJPATH)/output.o: $(SRCPATH)/output.cpp $(HPATH)/io.h
	${CCX} ${CCXFLAGS} -c -o $(OBJPATH)/output.o $(SRCPATH)/output.cpp

$(OBJPATH)/energy.o: $(SRCPATH)/energy.cpp $(HPATH)/energy.h $(HPATH)/io.h $(HPATH)/util.h $(SRCPATH)/input.cpp \
							$(SRCPATH)/output.cpp $(SRCPATH)/util.cpp
	${CCX} ${CCXFLAGS} ${MKLFLAGS} -c -o $(OBJPATH)/energy.o $(SRCPATH)/energy.cpp

$(OBJPATH)/thermo.o: $(SRCPATH)/thermo.cpp $(HPATH)/thermo.h $(HPATH)/io.h $(HPATH)/util.h $(SRCPATH)/input.cpp \
							$(SRCPATH)/output.cpp $(SRCPATH)/util.cpp
	${CCX} ${CCXFLAGS} -c -o $(OBJPATH)/thermo.o $(SRCPATH)/thermo.cpp

$(OBJPATH)/moves.o: $(SRCPATH)/moves.cpp $(HPATH)/moves.h $(HPATH)/io.h $(HPATH)/util.h $(SRCPATH)/input.cpp \
							$(SRCPATH)/output.cpp $(SRCPATH)/util.cpp $(HPATH)/energy.h $(SRCPATH)/energy.cpp
	${CCX} ${CCXFLAGS} ${MKLFLAGS} -c -o $(OBJPATH)/moves.o $(SRCPATH)/moves.cpp

$(OBJPATH)/gpu.o: $(SRCPATH)/gpu.cu $(HPATH)/gpu.h $(HPATH)/io.h $(SRCPATH)/input.cpp $(SRCPATH)/output.cpp
	${NVCC} ${CUDAFLAGS} -lcurand -Xcompiler "${MKLFLAGS}" -c -o $(OBJPATH)/gpu.o $(SRCPATH)/gpu.cu

$(OBJPATH)/util.o: $(SRCPATH)/util.cpp $(HPATH)/util.h $(HPATH)/io.h $(SRCPATH)/input.cpp $(SRCPATH)/output.cpp
	${CCX} ${CCXFLAGS} -c -o $(OBJPATH)/util.o $(SRCPATH)/util.cpp

clean:
	rm -f $(BINPATH)/mc.exe $(OBJECTS)
