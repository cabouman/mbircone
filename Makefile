CC=icc
FLAGS=-fopenmp -O3 -Wall -pedantic

bin=./bin
SRCS  = $(shell find ./src -type f -name *.c ! -name genSysMatrix.c)
OBJS = $(SRCS:.c=.o)
EXECUTABLE=main
# --- meta labels -------------------------------------------------

all: tt
	rm -f ./src/*.o

tt:
	${CC} ${FLAGS} -c ${SRCS}
	mv *.o ./src/
	${CC} ${FLAGS} ${OBJS} -o ${bin}/${EXECUTABLE} -lm

clean:
	rm -f ./src/*.o 

# --- Compiling -------------------------------------------------

# ${OBJS} :
# 	${CC} ${FLAGS} -c -o $@ $(@:.o=.c)
