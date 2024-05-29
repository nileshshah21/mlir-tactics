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
#pragma scop
  for (int a = 0; a < ni; a++)
    for (int b = 0; b < ni; b++)
      for (int c = 0; c < ni; c++)
        for (int d = 0; d < ni; d++)
          for (int e = 0; e < ni; e++)
            for (int f = 0; f < ni; f++)
              C[a][b][c][d] += B[d][f][c][e] * A[a][e][b][f];
#pragma endscop
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
