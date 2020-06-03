CC=icc
FLAGS=-fopenmp -O3 -Wall -pedantic

OBJS  = allocate.o MBIRModularUtilities3D.o io3d.o computeSysMatrix.o icd3d.o recon3DCone.o plainParams.o


all: main clean

main: main.o $(OBJS)
	${CC} ${FLAGS} $^ -o $@ -lm
	mv $@ ../bin

clean:
	rm *.o

%.o: %.c
	$(CC) -c $< $(FLAGS) -o $@

