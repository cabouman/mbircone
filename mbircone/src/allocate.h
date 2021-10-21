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

#ifndef ALLOCATE_INC
#define ALLOCATE_INC

#include <stdio.h>
#include <stdlib.h>

void *get_spc(int num, size_t size);
void *mget_spc(int num, size_t size);
void **get_img(int wd,int ht, size_t size);
void ***get_3D(int N, int M, int A, size_t size);
void free_img(void **pt);
void free_3D(void ***pt);
void *multialloc(size_t s, int d, ...);
void multifree(void *r,int d);

/* *************************************************************************************************** */
void *****mem_alloc_5D(size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t dataTypeSize);
void mem_free_5D(void *****p);

void ****mem_alloc_4D(size_t N1, size_t N2, size_t N3, size_t N4, size_t dataTypeSize);
void mem_free_4D(void ****p);

void ***mem_alloc_3D(size_t N1, size_t N2, size_t N3, size_t dataTypeSize);
void mem_free_3D(void ***p);

void **mem_alloc_2D(size_t N1, size_t N2, size_t dataTypeSize);
void mem_free_2D(void **pt);

void *mem_alloc_1D(size_t N1, size_t dataTypeSize);
void mem_free_1D(void *pt);


#endif
