CC=icc
FLAGS=-fopenmp -O3 -Wall -pedantic
SRC   = src
OBJS  = $(SRC)/allocate.o $(SRC)/MBIRModularUtilities3D.o $(SRC)/io3d.o $(SRC)/computeSysMatrix.o $(SRC)/icd3d.o $(SRC)/recon3DCone.o $(SRC)/plainParams.o


all: main clean

main: $(SRC)/main.o $(OBJS)
	${CC} ${FLAGS} $^ -o $@ -lm
	mv $@ bin

clean:
	rm $(SRC)/*.o

%.o: %.c
	$(CC) -c $< $(FLAGS) -o $@
