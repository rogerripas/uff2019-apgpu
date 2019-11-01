#include <stdio.h>
#include <stdlib.h>

#define N 10000

__global__ void blur(int *in, int *out, int offset){
	int index = offset + threadIdx.x + blockIdx.x*blockDim.x;
	if (index<N || index>=((N-1)*N) || index%N==0 || (index+1)%N==0){
		out[index] = 0;
	} else {
		int top = in[index - N];
		int down = in[index + N];
		int left = in[index - 1];
		int right = in[index + 1];
		int top_left = in[index - N - 1];
		int top_right = in[index - N + 1];
		int down_left = in[index + N - 1];
		int down_right = in[index + N + 1];
		out[index] = (top+down+left+right+top_left+top_right+down_left+down_right)/8;
	}
	
}

int main (){
	
	int *h_original, *h_filtered;
	h_original = (int*) malloc(N*N*sizeof(int));
	h_filtered = (int*) malloc(N*N*sizeof(int));

	int *d_original, *d_filtered;
	cudaMallocManaged((void**) &d_original, N*N*sizeof(int));
	cudaMallocManaged((void**) &d_filtered, N*N*sizeof(int));

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
	for (int i=0; i<300; i++){
		if (i%N == 0)
			printf("\n");
		printf("%d ",h_original[i]);
	}
	printf("\n");
	printf("\n");
	*/

    int nStreams = 6;    
    int streamSize = N*N/nStreams;
  	int streamBytes = streamSize * sizeof(int);

  	cudaStream_t stream[nStreams];
  	for (int i = 0; i < nStreams; ++i)
    	cudaStreamCreate(&stream[i]);

   	for (int i=0; i<nStreams; i++){
    	int offset = i * streamSize;
    	cudaMemcpyAsync(&d_original[offset], &h_original[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    	blur<<<streamSize/64, 64, 0, stream[i]>>>(d_original, d_filtered, offset);
    	cudaMemcpyAsync(&h_filtered[offset], &d_filtered[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();
   	
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

	for (int i = 0; i < nStreams; i++)
    	cudaStreamDestroy(stream[i]);

	free(h_original);
	free(h_filtered);
	cudaFree(d_original);
	cudaFree(d_filtered);

	return 0;
}