CC=icc
CFLAGS=-Wall -pedantic   -O3 -fopenmp

OBJECTS=io3d.o MBIRModularUtilities3D.o allocate.o
SOURCES=io3d.c MBIRModularUtilities3D.c allocate.c

libOBJECTS=../../../plainParams/plainParams.o
libSOURCES=../../../plainParams/plainParams.c



all: ${OBJECTS}

clean:
	rm ${OBJECTS}

${OBJECTS}: ${libOBJECTS}
	${CC} ${CFLAGS}  -c ${SOURCES}

# ${libOBJECTS}: ${libSOURCES}
	


# # To notice when there are changes in the header files
# DEPENDENCIES: .DEPENDENCIES

# .DEPENDENCIES: ${SOURCES}
# 	rm -f ./.DEPENDENCIES
# 	${CC} ${CFLAGS} -MM $^>>./.DEPENDENCIES;

# include .DEPENDENCIES