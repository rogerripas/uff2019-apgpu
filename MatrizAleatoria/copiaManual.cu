#include <stdio.h>
#include <stdlib.h>

#define N 10000

__global__ void blur(int *in, int *out){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;	

	if (row >= N || col >= N){
		return;
	}

	if (row<=0 || row>=N-1 || col<=0 || col>=N-1){
		out[row*N + col] = 0;
	} else {
		int top = in[(row-1) * N + col];
		int down = in[(row+1) * N + col];
		int left = in[row * N + (col-1)];
		int right = in[row * N + (col+1)];
		int top_left = in[(row-1) * N + (col-1)];
		int top_right = in[(row-1) * N + (col+1)];
		int down_left = in[(row+1) * N + (col-1)];
		int down_right = in[(row+1) * N + (col+1)];
		out[row*N + col] = (top+down+left+right+top_left+top_right+down_left+down_right)/8;
	}
}

int main (){
	
	int *h_original, *h_filtered;
	h_original = (int*) malloc(N*N*sizeof(int));
	h_filtered = (int*) malloc(N*N*sizeof(int));

	int *d_original, *d_filtered;
	cudaMalloc((void**) &d_original, N*N*sizeof(int));
	cudaMalloc((void**) &d_filtered, N*N*sizeof(int));

	int r;

	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			r = rand() % 9;
			h_original[row*N + col] = r;
			h_filtered[row*N + col] = 0;
		}
	}

	/*
	printf("Original:\n");
	printf("---------");
	for (int i=0; i<N*N; i++){
		if (i%N == 0)
			printf("\n");
		printf("%d ",h_original[i]);
	}
	printf("\n");
	printf("\n");
	*/

	cudaMemcpy(d_original, h_original, N*N*sizeof(int), cudaMemcpyHostToDevice);

	dim3 blkDim (32, 32, 1);
    dim3 grdDim (N/blkDim.x + 1, N/blkDim.y + 1, 1); 
    blur<<<grdDim, blkDim>>>(d_original, d_filtered);
    cudaDeviceSynchronize();

    cudaMemcpy(h_filtered, d_filtered, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    /*
    printf("Filtered:\n");
    printf("---------");
	for (int i=0; i<N*N; i++){
		if (i%N == 0)
			printf("\n");
		printf("%d ", h_filtered[i]);
	}
	printf("\n");
	*/

	free(h_original);
	free(h_filtered);
	cudaFree(d_original);
	cudaFree(d_filtered);

	return 0;
}