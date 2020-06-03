CC=icc
FLAGS=-fopenmp -O3 -Wall -pedantic
SRC   = src

OBJ_NAME=allocate.o MBIRModularUtilities3D.o io3d.o computeSysMatrix.o icd3d.o recon3DCone.o plainParams.o
OBJS=$(join $(addsuffix src/, $(dir $(OBJ_NAME))), $(notdir $(OBJ_NAME)))

all: main clean

main: $(SRC)/main.o $(OBJS)
	${CC} ${FLAGS} $^ -o $@ -lm
	mv $@ bin

clean:
	rm $(SRC)/*.o

%.o: %.c
	$(CC) -c $< $(FLAGS) -o $@
