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

#include "abcd-aebf-dfce.h"

static
void init_array(int ni,
    DATA_TYPE POLYBENCH_4D(A, NI, NI, NI, NI, ni, ni, ni, ni),
    DATA_TYPE POLYBENCH_4D(B, NI, NI, NI, NI, ni, ni, ni, ni),
    DATA_TYPE POLYBENCH_4D(C, NI, NI, NI, NI, ni, ni, ni, ni)) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < ni; j++)
      for (int k = 0; k < ni; k++)
        for (int l = 0; l < ni; l++) {
          A[i][j][k][l] = (DATA_TYPE) ((i*j+1) % ni) / ni;
          B[i][j][k][l] = (DATA_TYPE) ((i*j+2) % ni) / ni;
          C[i][j][k][l] = (DATA_TYPE) ((i*j+1) % ni) / ni;
        }
}

static 
void print_array(int ni,
    DATA_TYPE POLYBENCH_4D(C, NI, NI, NI, NI, ni, ni, ni, ni)) {
  int i, j, k, l;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) 
      for (k = 0; k < ni; k++)
        for (l = 0; l < ni; l++) {
          if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
          fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j][k][l]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

static 
void kernel(int ni,
    DATA_TYPE POLYBENCH_4D(A, NI, NI, NI, NI, ni, ni, ni, ni),
    DATA_TYPE POLYBENCH_4D(B, NI, NI, NI, NI, ni, ni, ni, ni),
    DATA_TYPE POLYBENCH_4D(C, NI, NI, NI, NI, ni, ni, ni, ni)) {
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
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
 register int lbv, ubv;
/* Start of CLooG code */
if (ni >= 1) {
  for (t1=0;t1<=floord(ni-1,32);t1++) {
    for (t2=0;t2<=floord(ni-1,32);t2++) {
      for (t3=0;t3<=floord(ni-1,32);t3++) {
        for (t4=0;t4<=floord(ni-1,32);t4++) {
          for (t5=0;t5<=floord(ni-1,32);t5++) {
            for (t6=32*t1;t6<=min(ni-1,32*t1+31);t6++) {
              for (t7=32*t2;t7<=min(ni-1,32*t2+31);t7++) {
                for (t8=32*t3;t8<=min(ni-1,32*t3+31);t8++) {
                  for (t9=32*t4;t9<=min(ni-1,32*t4+31);t9++) {
                    for (t10=32*t5;t10<=min(ni-1,32*t5+31);t10++) {
                      for (t11=0;t11<=ni-1;t11++) {
                        C[t6][t7][t8][t9] += B[t9][t11][t8][t10] * A[t6][t10][t7][t11];;
                      }
                    }
                  }
                }
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
  
  POLYBENCH_4D_ARRAY_DECL(C, DATA_TYPE, NI, NI, NI, NI, ni, ni, ni, ni);
  POLYBENCH_4D_ARRAY_DECL(A, DATA_TYPE, NI, NI, NI, NI, ni, ni, ni, ni);
  POLYBENCH_4D_ARRAY_DECL(B, DATA_TYPE, NI, NI, NI, NI, ni, ni, ni, ni);
  
  init_array(ni, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_start_instruments

  kernel(ni, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_stop_instruments;
  polybench_print_instruments;

  polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(C)));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  return 0;
}
