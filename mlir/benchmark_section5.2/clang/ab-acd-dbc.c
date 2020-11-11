#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ab-acd-dbc.h"

static 
void init_array(int ni, int nj,
    DATA_TYPE POLYBENCH_3D(A, NI, NJ, NJ, ni, nj, nj),
    DATA_TYPE POLYBENCH_3D(B, NJ, NI, NJ, nj, ni, nj),
    DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni)) {

  for (int i = 0; i < ni; i++)
    for (int j = 0; j < ni; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;

  for (int i = 0; i < ni; i ++)
    for (int j = 0; j < nj; j++)
      for (int k = 0; k < nj; k++)
        A[i][j][k] = (DATA_TYPE) ((i*j+2) % ni) / ni;

  for (int i = 0; i < nj; i++)
    for (int j = 0; j < ni; j++)
      for (int k = 0; k < nj; k++)
        B[i][j][k] = (DATA_TYPE) ((i*j+3) % ni) / ni;
}

static
void print_array(int ni,
     DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) {
  if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

static 
void kernel(int ni, int nj,
    DATA_TYPE POLYBENCH_3D(A, NI, NJ, NJ, ni, nj, nj),
    DATA_TYPE POLYBENCH_3D(B, NJ, NI, NJ, nj, ni, nj),
    DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni)) {
#pragma scop
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < ni; j++)
      for (int k = 0; k < nj; k++)
        for (int l = 0; l < nj; l++)
          C[i][j] += A[i][k][l] * B[l][j][k];
#pragma endscop
}

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;

  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NI, ni, ni);
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, NJ, ni, nj, nj);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NJ, NI, NJ, nj, ni, nj);
  
  init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_start_instruments

  kernel(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
    POLYBENCH_ARRAY(C));

  polybench_stop_instruments;
  polybench_print_instruments;

  polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(C)));

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  return 0;
}
  
