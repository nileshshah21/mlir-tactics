#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "abc-ad-bdc.h"

static
void init_array(int ni, int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NI, ni, ni),
    DATA_TYPE POLYBENCH_3D(B, NJ, NI, NJ, nj, ni, nj),
    DATA_TYPE POLYBENCH_3D(C, NI, NJ, NJ, ni, nj, nj)) {

 for (int i = 0; i < ni; i++)
    for (int j = 0; j < ni; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;

  for (int i = 0; i < nj; i ++)
    for (int j = 0; j < ni; j++)
      for (int k = 0; k < nj; k++)
        B[i][j][k] = (DATA_TYPE) ((i*j+2) % ni) / ni;

  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++)
      for (int k = 0; k < nj; k++)
        C[i][j][k] = (DATA_TYPE) ((i*j+3) % ni) / ni;
}

static void print_array(int ni, int nj,
    DATA_TYPE POLYBENCH_3D(C, NI, NJ, NJ, ni, nj, nj)) {
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) 
      for (k = 0; k < nj; k++) {
        if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j][k]);
      }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

static void kernel(int ni, int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NI, ni, ni),
    DATA_TYPE POLYBENCH_3D(B, NJ, NI, NJ, nj, ni, nj),
    DATA_TYPE POLYBENCH_3D(C, NI, NJ, NJ, ni, nj, nj)) {
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
  int t1, t2, t3, t4, t5, t6, t7;
 register int lbv, ubv;
/* Start of CLooG code */
if ((ni >= 1) && (nj >= 1)) {
  for (t1=0;t1<=floord(ni-1,32);t1++) {
    for (t2=0;t2<=floord(nj-1,32);t2++) {
      for (t3=0;t3<=floord(nj-1,32);t3++) {
        for (t4=32*t1;t4<=min(ni-1,32*t1+31);t4++) {
          for (t5=32*t2;t5<=min(nj-1,32*t2+31);t5++) {
            for (t6=32*t3;t6<=min(nj-1,32*t3+31);t6++) {
              for (t7=0;t7<=ni-1;t7++) {
                C[t4][t5][t5] += A[t4][t7] * B[t5][t7][t6];;
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

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;

  POLYBENCH_3D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, NJ, ni, nj, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NI, ni, ni);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NJ, NI, NJ, nj, ni, nj);

  init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_start_instruments

  kernel(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_stop_instruments;
  polybench_print_instruments;

  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  return 0;
}
