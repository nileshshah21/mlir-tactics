#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

/* Copyright (C) 1991-2018 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NI >= 0) && (_PB_NJ >= 0) && (_PB_NL >= 1)) {
  for (t2=0;t2<=floord(_PB_NI+_PB_NJ-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
      for (t4=32*t2;t4<=min(min(_PB_NI-1,_PB_NJ-1),32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          F[t4][t5] = SCALAR_VAL(0.0);;
          G[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
      for (t4=max(_PB_NI,32*t2);t4<=min(_PB_NJ-1,32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          F[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
      for (t4=max(_PB_NJ,32*t2);t4<=min(_PB_NI-1,32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          G[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
}
if ((_PB_NI <= -1) && (_PB_NL >= 1)) {
  for (t2=0;t2<=floord(_PB_NJ-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_NJ-1,32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          F[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
}
if ((_PB_NJ <= -1) && (_PB_NL >= 1)) {
  for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          G[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
}
if ((_PB_NL >= 1) && (_PB_NM >= 1)) {
  for (t2=0;t2<=floord(_PB_NJ-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
      for (t5=0;t5<=floord(_PB_NM-1,32);t5++) {
        for (t6=32*t2;t6<=min(_PB_NJ-1,32*t2+31);t6++) {
          for (t7=32*t3;t7<=min(_PB_NL-1,32*t3+31);t7++) {
            for (t9=32*t5;t9<=min(_PB_NM-1,32*t5+31);t9++) {
              F[t6][t7] += C[t6][t9] * D[t9][t7];;
            }
          }
        }
      }
    }
  }
}
if (_PB_NJ >= 1) {
  for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
        lbv=32*t3;
        ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
        for (t5=lbv;t5<=ubv;t5++) {
          E[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
}
if (_PB_NJ >= 1) {
  for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
      if (_PB_NK >= 1) {
        for (t5=0;t5<=floord(_PB_NK-1,32);t5++) {
          for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
            for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
              for (t9=32*t5;t9<=min(_PB_NK-1,32*t5+31);t9++) {
                E[t6][t7] += A[t6][t9] * B[t9][t7];;
              }
            }
          }
        }
      }
      if (_PB_NL >= 1) {
        for (t5=0;t5<=floord(_PB_NL-1,32);t5++) {
          for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
            for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
              lbv=32*t5;
              ubv=min(_PB_NL-1,32*t5+31);
#pragma ivdep
#pragma vector always
              for (t9=lbv;t9<=ubv;t9++) {
                G[t6][t9] += E[t6][t7] * F[t7][t9];;
              }
            }
          }
        }
      }
    }
  }
}
/* End of CLooG code */

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}
