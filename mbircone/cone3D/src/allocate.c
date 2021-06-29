/* ============================================================================== 
 * Copyright (c) 2013 Charles A. Bouman (Purdue University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of Charles A. Bouman, Purdue University,
 * nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "allocate.h"

void *get_spc(int num, size_t size)
{
	void *pt;

	if ((pt = calloc((size_t)num,size)) == NULL)
	{
		fprintf(stderr, "==> calloc() error\n");
		exit(-1);
	}
	return(pt);
}

void *mget_spc(int num,size_t size)
{
	void *pt;

	if ((pt = malloc((size_t)(num*size))) == NULL)
	{
		fprintf(stderr, "==> malloc() error. size=%d \n",(int)(num*size));
		exit(-1);
	}
	return(pt);
}


void **get_img(int wd,int ht,size_t size)
{
	void *pt;

	if ((pt = multialloc(size,2,ht,wd)) == NULL)
	{
		fprintf(stderr, "get_img: out of memory\n");
		exit(-1);
	}
	return((void **)pt);
}

void ***get_3D(int N, int M, int A, size_t size)
{
	void *pt;

	if ((pt = multialloc(size,3,N,M,A)) == NULL)
	{
		fprintf(stderr, "get_3D: out of memory\n");
		exit(-1);
	}
	return((void ***)pt);
}


void free_img(void **pt)
{
	multifree((void *)pt, 2);
}

void free_3D(void ***pt)
{
	multifree((void *)pt, 3);
}




/* modified from dynamem.c on 4/29/91 C. Bouman                           */
/* Converted to ANSI on 7/13/93 C. Bouman         	                  */
/* multialloc( s, d,  d1, d2 ....) allocates a d dimensional array, whose */
/* dimensions are stored in a list starting at d1. Each array element is  */
/* of size s.                                                             */


void *multialloc(size_t s, int d, ...)
{
	va_list ap;             /* varargs list traverser */
	int max,                /* size of array to be declared */
	    *q;                     /* pointer to dimension list */
	char **r,               /* pointer to beginning of the array of the
				 * pointers for a dimension */
	     **s1, *t, *tree;        /* base pointer to beginning of first array */
	int i, j;               /* loop counters */
	int *d1;                /* dimension list */

	va_start(ap,d);
	d1 = (int *) mget_spc(d,sizeof(int));

	for(i=0;i<d;i++)
		d1[i] = va_arg(ap,int);

	r = &tree;
	q = d1;                /* first dimension */
	max = 1;
	for (i = 0; i < d - 1; i++, q++) {      /* for each of the dimensions
						 * but the last */
		max *= (*q);
		r[0]=(char *)mget_spc(max,sizeof(char **));
		r = (char **) r[0];     /* step through to beginning of next
					 * dimension array */
	}
	max *= s * (*q);        /* grab actual array memory */
	r[0] = (char *)mget_spc(max,sizeof(char));

	/*
	 * r is now set to point to the beginning of each array so that we can
	 * use it to scan down each array rather than having to go across and
	 * then down 
	 */
	r = (char **) tree;     /* back to the beginning of list of arrays */
	q = d1;                 /* back to the first dimension */
	max = 1;
	for (i = 0; i < d - 2; i++, q++) {      /* we deal with the last
						 * array of pointers later on */
		max *= (*q);    /* number of elements in this dimension */
		for (j=1, s1=r+1, t=r[0]; j<max; j++) { /* scans down array for
							 * first and subsequent
							 * elements */

			/*  modify each of the pointers so that it points to
			 * the correct position (sub-array) of the next
			 * dimension array. s1 is the current position in the
			 * current array. t is the current position in the
			 * next array. t is incremented before s1 is, but it
			 * starts off one behind. *(q+1) is the dimension of
			 * the next array. */

			*s1 = (t += sizeof (char **) * *(q + 1));
			s1++;
		}
		r = (char **) r[0];     /* step through to begining of next
					 * dimension array */
	}
	max *= (*q);              /* max is total number of elements in the
				   * last pointer array */

	/* same as previous loop, but different size factor */
	for (j = 1, s1 = r + 1, t = r[0]; j < max; j++) 
		*s1++ = (t += s * *(q + 1));

	va_end(ap);
	free((void *)d1);
	return((void *)tree);              /* return base pointer */
}



/*
 * multifree releases all memory that we have already declared analogous to
 * free() when using malloc() 
 */
void multifree(void *r,int d)
{
	void **p;
	void *next=NULL;
	long int i;

	for (p = (void **)r, i = 0; i < d; p = (void **) next,i++)
		if (p != NULL) {
			next = *p;
			free((void *)p);
		}
}



/*****************************************************************************************************
 * 		Added by Soumendu Majee and Thilo Balke 2017.10.24
 *
 * 		These routines allocate a mutidimensional array. The lowest level array 
 * 		is stored in contiguous memory. This is replicating the purpose of multialloc.
 * 		(multialloc was found to cause errors for large allocations)
 *
 * 		Basic idea behind these routines:
 * 		Assume we have N-dimensional allocation routine. Want to create N+1-dimensional array.
 * 		First create 1D-array corresponding to the bottom level (leaves). Then create N-dim array for the top
 * 		level (topTree). Then point the elements of the top tree one-by-one to the leaf elements.
 * 		
 *****************************************************************************************************/
void *****mem_alloc_5D(size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t dataTypeSize)
{
	void *****topTree;
	char *leaves;
	long int i1, i2, i3, i4;

	topTree = (void*****) 	mem_alloc_4D(N1, N2, N3, N4, sizeof(void*));
	leaves = (char*) 		mem_alloc_1D(N1*N2*N3*N4*N5, dataTypeSize);

	for(i1 = 0; i1 < N1; i1++)
	for(i2 = 0; i2 < N2; i2++)
	for(i3 = 0; i3 < N3; i3++)
	for(i4 = 0; i4 < N4; i4++)
	{
		topTree[i1][i2][i3][i4] = leaves + (((i1*N2 + i2)*N3 + i3)*N4 + i4)*N5 * dataTypeSize;
	}
	return(topTree);	
}

void mem_free_5D(void *****p)
{
	free((void*)p[0][0][0][0]);		/* free leaves  */
	mem_free_4D((void****)p);		/* free topTree */
}

void ****mem_alloc_4D(size_t N1, size_t N2, size_t N3, size_t N4, size_t dataTypeSize)
{
	void ****topTree;
	char *leaves;
	long int i1, i2, i3;

	topTree = (void****) 	mem_alloc_3D(N1, N2, N3, sizeof(void*));
	leaves = (char*) 		mem_alloc_1D(N1*N2*N3*N4, dataTypeSize);

	for(i1 = 0; i1 < N1; i1++)
	for(i2 = 0; i2 < N2; i2++)
	for(i3 = 0; i3 < N3; i3++)
	{
		topTree[i1][i2][i3] = leaves + ((i1*N2 + i2)*N3 + i3)*N4 * dataTypeSize;
	}
	return(topTree);	
}

void mem_free_4D(void ****p)
{
	free((void*)p[0][0][0]);		/* free leaves  */
	mem_free_3D((void***)p);		/* free topTree */
}

void ***mem_alloc_3D(size_t N1, size_t N2, size_t N3, size_t dataTypeSize)
{
	void ***topTree;
	char *leaves;
	long int i1, i2;

	topTree = (void***) 	mem_alloc_2D(N1, N2, sizeof(void*));
	leaves = (char*) 		mem_alloc_1D(N1*N2*N3, dataTypeSize);

	for(i1 = 0; i1 < N1; i1++)
	for(i2 = 0; i2 < N2; i2++)
	{
		topTree[i1][i2] = leaves + (i1*N2 + i2)*N3 * dataTypeSize;
	}
	return(topTree);	
}

void mem_free_3D(void ***p)
{
	free((void*)p[0][0]);		/* free leaves  */
	mem_free_2D((void**)p);		/* free topTree */
}

void **mem_alloc_2D(size_t N1, size_t N2, size_t dataTypeSize)
{
	void **topTree;	
	char *leaves;
	long int i1;

	topTree = (void **) 	mem_alloc_1D(N1, sizeof(void *));
	leaves = (char *) 		mem_alloc_1D(N2*N1, dataTypeSize);

	for(i1 = 0; i1 < N1; i1++)
		topTree[i1] = leaves + i1*N2 * dataTypeSize;

	return(topTree);
}

void mem_free_2D(void **pt)
{
	free( (void *)pt[0]);		/* free leaves  */
	mem_free_1D( (void *)pt);	/* free topTree */
}

void *mem_alloc_1D(size_t N1, size_t dataTypeSize)
{
	void *topTree;

	topTree = malloc((size_t)(N1*dataTypeSize));

	if (topTree == NULL)
	{
		fprintf(stderr, "Memory allocation Error:\n\tWhen trying to allocate:\n\t\tNumber of elements = %u\n\t\tSize per element = %u bytes\n\t\tTotal size = %u bytes\n", (unsigned int)(N1), (unsigned int)(dataTypeSize), (unsigned int)(N1*dataTypeSize));
		exit(-1);
	}
	return(topTree);
}

void mem_free_1D(void *pt)
{
	free((void*)pt);
}
