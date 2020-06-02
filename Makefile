CC=icc
FLAGS=-fopenmp -O3 -Wall -pedantic

bin=./bin
SRCS  = "allocate.c MBIRModularUtilities3D.c io3d.c computeSysMatrix.c icd3d.c recon3DCone.c main.c plainParams.c"
OBJS  = "allocate.o MBIRModularUtilities3D.o io3d.o computeSysMatrix.o icd3d.o recon3DCone.o main.o plainParams.o"
EXECUTABLE=main
# --- meta labels -------------------------------------------------

all: main clean
	rm ./src/*.o 

main:
	${CC} ${FLAGS} -c ${SRCS}
	mv *.o ./src/
	${CC} ${FLAGS} ${OBJS} -o ${bin}/${EXECUTABLE} -lm

clean:
	rm ./src/*.o 

# --- Compiling -------------------------------------------------

# ${OBJS} :
# 	${CC} ${FLAGS} -c -o $@ $(@:.o=.c)
