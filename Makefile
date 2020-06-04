CC=gcc
FLAGS=-fopenmp -O3 -Wall -pedantic
SRC=src

OBJ_NAME=allocate.o MBIRModularUtilities3D.o io3d.o computeSysMatrix.o icd3d.o recon3DCone.o plainParams.o
OBJS=$(addprefix $(SRC)/, $(OBJ_NAME))

.PHONY: all clean

all: main

main: $(SRC)/main.o $(OBJS)
	${CC} ${FLAGS} $^ -o $@ -lm
	mv $@ bin

clean:
	rm $(SRC)/*.o

%.o: %.c
	$(CC) -c $< $(FLAGS) -o $@
