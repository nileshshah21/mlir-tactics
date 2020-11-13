#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <polybench.h>

#include <conv2d-nchw.h>

static
void init_array(int ni, int nj,
		DATA_TYPE POLYBENCH_2D(output, NI, NI, ni, ni),
		DATA_TYPE POLYBENCH_2D(filter, NI, NI, ni, ni),
		DATA_TYPE POLYBENCH_2D(image, NJ, NJ, nj, nj)) {
	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < ni; j++) {
			output[i][j] = (DATA_TYPE) (i + 1) / ni;
			filter[i][j] = (DATA_TYPE) (i + 2)/ ni;
		}
	}
	for (int i = 0; i < nj; i++) {
		for (int j = 0; j < nj; j++) {
			image[i][j] = (DATA_TYPE) (i + 3) / nj;
		}
	}
}

static
void print_array(int ni,
		DATA_TYPE POLYBENCH_2D(output, NI, NI, ni, ni)) {
	
	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("OUT");
	for (int i = 0; i < ni; i++) {
		if ((i * ni) % 20 == 0) {
			fprintf(POLYBENCH_DUMP_TARGET, "\n");
			fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, output[i]);
		}
	}	
	POLYBENCH_DUMP_END("OUT");
	POLYBENCH_DUMP_FINISH;
}

static 
void kernel(int ni, int nj,
		DATA_TYPE POLYBENCH_2D(output, NI, NI, ni, ni),
		DATA_TYPE POLYBENCH_2D(filter, NI, NI, ni, ni),
		DATA_TYPE POLYBENCH_2D(image, NJ, NJ, nj, nj)) {
#pragma scop
	for (int out_h = 0; out_h < ni; out_h++)
		for (int out_w = 0; out_w < ni; out_w++)
			for (int k_h = 0; k_h < ni; k_h++)
				for (int k_w = 0; k_w < ni; k_w++)
					output[out_h][out_w] += 
						filter[k_h][k_w] * image[(out_h + k_h)][(out_w + k_w)];
#pragma endscop
}

int main(int argc, char** argv) {
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(image, DATA_TYPE, NJ, NJ, nj, nj);		
	POLYBENCH_2D_ARRAY_DECL(filter, DATA_TYPE, NI, NI, ni, ni);
	POLYBENCH_2D_ARRAY_DECL(output, DATA_TYPE, NI, NI, ni, ni);

	init_array(ni, nj, POLYBENCH_ARRAY(output), POLYBENCH_ARRAY(filter),
		POLYBENCH_ARRAY(image));

	polybench_start_instruments

	kernel(ni, nj, POLYBENCH_ARRAY(output), POLYBENCH_ARRAY(filter),
		POLYBENCH_ARRAY(image));
	
	polybench_stop_instruments;
	polybench_print_instruments;

	polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(output)));

	POLYBENCH_FREE_ARRAY(image);
	POLYBENCH_FREE_ARRAY(output);
	POLYBENCH_FREE_ARRAY(filter);
	return 0;
}
