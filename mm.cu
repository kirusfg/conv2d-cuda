#include <stdio.h>
#include <algorithm>

typedef struct {
  size_t M, P, N, BS;
} Dimensions;

static inline void dims(char *s, dim3 grid, dim3 block) {
  sprintf(s, "(%d, %d, %d), (%d, %d, %d)",
    grid.x, 
    grid.y, 
    grid.z,
    block.x, 
    block.y, 
    block.z
  );
}

void init(float *A, float *B, float *C, Dimensions d) {
  size_t M = d.M;
  size_t P = d.P;
  size_t N = d.N;

  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < P; j++)
      A[i * P + j] = 1.0f;

  for (size_t i = 0; i < P; i++)
    for (size_t j = 0; j < N; j++)
      B[i * N + j] = 2.0f;

  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++)
      C[i * N + j] = 0.0f;
}

void verify(float *C, float val, Dimensions d) {
  size_t M = d.M;
  size_t N = d.N;

  // Checks the last row of C for correctness
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (C[i * N + j] != val) {
        printf("Result incorrect for C[%lu][%lu]: got %f, expected %f\n", i, j, C[i * N + j],
              val);
        return;
      }
    }
  }
  printf("[PASS] All values were equal to %f\n", val);
}

void print(float *C, Dimensions d) {
  size_t M = d.M;
  size_t N = d.N;

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      printf("%g ", C[i * N + j]);
    }
    printf("\n");
  }
}

void mm(float *A, float *B, float *C, Dimensions d) {
  size_t M = d.M;
  size_t P = d.P;
  size_t N = d.N;

  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++)
      for (size_t k = 0; k < P; k++)
        C[i * N + j] += A[i * P + k] * B[k * N + j];
}

__global__ void mmGlobal(float *A, float *B, float *C, Dimensions *d) {
  size_t M = d->M;
  size_t P = d->P;
  size_t N = d->N;

  size_t row = threadIdx.x + blockIdx.x * blockDim.x;
  size_t col = threadIdx.y + blockIdx.y * blockDim.y;

  for (size_t k = 0; k < P; k++)
    if (row < M && col < N)
      C[row * N + col] += A[row * P + k] * B[k * N + col];
}

__global__ void mmShared(float *A, float *B, float *C, Dimensions *d) {
  const size_t M = d->M;
  const size_t P = d->P;
  const size_t N = d->N;
  const size_t BS = d->BS;

  extern __shared__ float ABs[];
  float *As = ABs;
  float *Bs = &ABs[BS * BS];

  int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BS + ty;
    int col = bx * BS + tx;

    float Csub = 0;

    for (int i = 0; i < (P - 1) / BS + 1; ++i) {
        if (row < M && i * BS + tx < P) {
            As[ty * BS + tx] = A[row * P + i * BS + tx];
        }
        else {
            As[ty * BS + tx] = 0;
        }

        if (col < N && i * BS + ty < P) {
            Bs[ty * BS + tx] = B[(i * BS + ty) * N + col];
        }
        else {
            Bs[ty * BS + tx] = 0;
        }

        __syncthreads();

        for (int j = 0; j < BS; ++j) {
            Csub += As[ty * BS + j] * Bs[j * BS + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Csub;
    }
}

/*
__global__ void mmShared(float *A, float *B, float *C, Dimensions *d) {
  size_t M = d->M;
  size_t P = d->P;
  size_t N = d->N;
  size_t BS = d->BS;

  size_t row = threadIdx.x + blockIdx.x * blockDim.x;
  size_t col = threadIdx.y + blockIdx.y * blockDim.y;
  size_t sRow = threadIdx.x;
  size_t sCol = threadIdx.y;

  extern __shared__ float ABShared[];
  float *AShared = ABShared;
  float *BShared = &ABShared[BS * BS];
  __syncthreads();
  // extern __shared__ float BShared[];
  float CInterm = 0.0f;

  size_t numBlocksInM = (size_t)ceil((double)M / BS);
  size_t numBlocksInP = (size_t)ceil((double)P / BS);
  size_t numBlocksInN = (size_t)ceil((double)N / BS);
  
  size_t blockIdM = (size_t)row / BS;
  size_t blockIdN = (size_t)col / BS;

  // if (row == 0 && col == 0) {
  //   printf("%lu\n", numBlocksInM);
  //   printf("%lu\n", numBlocksInP);
  //   printf("%lu\n", numBlocksInN);
  // }

  for (size_t blockIdP = 0; blockIdP < numBlocksInP; blockIdP++) {
    // M, P, and N might not be multiples of BS, hence we need to copy only
    // some values at the edges of A and B; otherwise, we init those cells to 0
    size_t rowInA = (blockIdM * BS + sRow);
    size_t colInA = (blockIdP * BS + sCol);
    AShared[sRow * BS + sCol] = (rowInA < M && colInA < P) ? A[rowInA * P + colInA] : 0.0f;

    __syncthreads();

    size_t rowInB = (blockIdP * BS + sRow);
    size_t colInB = (blockIdN * BS + sCol);
    BShared[sRow * BS + sCol] = (rowInB < P && colInB < N) ? B[rowInB * N + colInB] : 0.0f;
    
    __syncthreads();
    
    // if (row == M - 1 && col == N - 1)
    //   printf("%lu %lu %lu %lu %lu\n", rowInB, colInB, sRow, sCol, sRow * BS + sCol);
    // __syncthreads();

    // if (row == 11 && col == 0 && blockIdP == numBlocksInP - 1) {
    //   printf("Row in A %lu\n", rowInA);
    //   printf("Col in A %lu\n", colInA);
    //   printf("Row in B %lu\n", rowInB);
    //   printf("Col in B %lu\n", colInB);

    //   for (size_t i = 0; i < BS; i++) {
    //     for (size_t j = 0; j < BS; j++)
    //       printf("%g ", AShared[i * BS + j]);
    //     printf("\n");
    //   }
    //   printf("\n");
    //   for (size_t i = 0; i < BS; i++) {
    //     for (size_t j = 0; j < BS; j++)
    //       printf("%g ", BShared[i * BS + j]);
    //     printf("\n");
    //   }
    // }

    // __syncthreads();

    for (size_t e = 0; e < BS; e++)
      CInterm += AShared[sRow * BS + e] * BShared[e * BS + sCol];

    __syncthreads();

    C[row * N + col] = CInterm;
  }
}
*/

int main(int argc, char **argv) {
  int ver = 1;
  Dimensions *d = (Dimensions *)malloc(sizeof(Dimensions));
  d->M = 512;
  d->P = 256;
  d->N = 128;
  d->BS = 16;

  if (argc == 2) {
    sscanf(argv[1], "%i", &ver);
  } else if (argc == 6) {
    sscanf(argv[1], "%i", &ver);
    sscanf(argv[2], "%zu", &(d->M));
    sscanf(argv[3], "%zu", &(d->P));
    sscanf(argv[4], "%zu", &(d->N));
    sscanf(argv[5], "%zu", &(d->BS));
  }

  size_t M = d->M;
  size_t P = d->P;
  size_t N = d->N;
  size_t BS = d->BS;

  printf("%zu\n", d->M);
  printf("%zu\n", d->P);
  printf("%zu\n", d->N);
  printf("%zu\n", d->BS);

  float *A, *B, *C;
  float *ADevice, *BDevice, *CDevice;
  Dimensions *dDevice;
  size_t ASize = M * P * sizeof(float);
  size_t BSize = P * N * sizeof(float);
  size_t CSize = M * N * sizeof(float);
  size_t dSize = sizeof(Dimensions);

  A = (float *)malloc(ASize);
  B = (float *)malloc(BSize);
  C = (float *)malloc(CSize);

  cudaMalloc((void **)&ADevice, ASize);
  cudaMalloc((void **)&BDevice, BSize);
  cudaMalloc((void **)&CDevice, CSize);
  cudaMalloc((void **)&dDevice, dSize);

  init(A, B, C, *d);

  cudaMemcpy(ADevice, A, ASize, cudaMemcpyHostToDevice);
  cudaMemcpy(BDevice, B, BSize, cudaMemcpyHostToDevice);
  cudaMemcpy(CDevice, C, CSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dDevice, d, dSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BS, BS);
  dim3 numBlocks((int)ceil((float)N / BS), (int)ceil((float)M / BS));
  // printf("Grid size: (%d, %d, %d)\n", numBlocks.x, numBlocks.y, numBlocks.z);
  // printf("%zu (%f) / %zu = %f\n", N, (float)N, BS, (float)N / BS);
  // printf("%f\n", ceil((float)N / BS));

  // size_t num_blocks_x = (N + BS - 1) / BS;
  // size_t num_blocks_y = (M + BS - 1) / BS;
  // dim3 numBlocks(num_blocks_x, num_blocks_y);
  // printf("Grid size: (%d, %d, %d)\n", numBlocks.x, numBlocks.y, numBlocks.z);



  char dimsStr[100];
  dims(dimsStr, numBlocks, threadsPerBlock);

  switch(ver) {
      case 1:
          // Sequential C
          printf("Running mm on CPU\n");
          mm(A, B, C, *d);
          break;
      case 2:
          // 27.168us
          printf("Running mmGlobal<<<%s>>>\n", dimsStr);
          mmGlobal<<<numBlocks, threadsPerBlock>>>(ADevice, BDevice, CDevice, dDevice);
          cudaMemcpy(C, CDevice, CSize, cudaMemcpyDeviceToHost);
          break;
      case 3:
          // 55.808us
          printf("Running mmShared<<<%s>>>\n", dimsStr);
          mmShared<<<numBlocks, threadsPerBlock, 2 * BS * BS * sizeof(float)>>>(ADevice, BDevice, CDevice, dDevice);
          // mmShared<<<numBlocks, threadsPerBlock>>>(ADevice, BDevice, CDevice, dDevice);
          
          cudaDeviceSynchronize();
          cudaMemcpy(C, CDevice, CSize, cudaMemcpyDeviceToHost);
          break;
      case 4:
          // 8.8320us, 256 CUDA Cores, 993 MHz
          // printf("Running add<<<%d, %d>>>\n", N/K, K);
          // add<<<(N + K - 1) / K, K>>>(ADevice, BDevice, CDevice);
          cudaMemcpy(C, CDevice, CSize, cudaMemcpyDeviceToHost);
          break;
  }

  cudaError_t err;
  if ((err = cudaGetLastError()) != ::cudaSuccess) {
    printf("Something went wrong: %s\n", cudaGetErrorString(err));
  }

  verify(C, 2.0f * P, *d);
  // print(C, *d);

  cudaFree(ADevice);
  cudaFree(BDevice);
  cudaFree(CDevice);
  cudaFree(dDevice);

  free(A);
  free(B);
  free(C);

  return 0;
}