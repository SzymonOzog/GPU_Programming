#include <fstream>
#include <iomanip>
#include <iostream> 
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
#include <random>
#include <string>

#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void forward(int batch_size, int n, int out_w, float* input, float* weights, float* biases, float* output)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    output[row*out_w+column] = biases[column];
    for(int i = 0; i < n; i++)
    {
      output[row*out_w+column] += input[row*n + i] * weights[i*n + column];
    }
  }
}

__global__ void relu(int w, int h, float* a, float* b)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < w && column < h)
  {
    float activation = a[row*w+column];
    b[row*w+column] =  activation > 0.f ? activation : 0.f;
  }
}

__global__ void softmax(int w, int h, float* a, float* b)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = 0.f;
    for (int i = 0; i<w; i++)
    {
      maxval = max(maxval, a[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += exp(a[row*w + i] - maxval);
    }
    b[row*w + col] = exp(a[row*w + col]-maxval)/(divisor);
  }
}

__global__ void cross_entropy(int w, int h, float* preds, float* real, float* output)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < h)
  {
    float loss = 0.f;
    for (int i = 0; i<w; i++)
    {
      loss -= real[idx*w + i] * log(max(1e-6, preds[idx*w + i]));
    }
    output[idx] = loss;
  }
}

__global__ void init_rand(int w, int h, float* mat)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    curandState state;
    curand_init(42, row*w+column, 0, &state);
    mat[row*w + column] = curand_uniform(&state);
  }
}

void initLayer(float* weights, float* biases, int w, int h, int BLOCK_SIZE)
{
  dim3 dimGrid = dim3(ceil(w/(float)BLOCK_SIZE), ceil(h/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(w, h, weights);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  dimGrid = dim3(1, ceil(h/(float)BLOCK_SIZE), 1);
  dimBlock = dim3(1, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(1, h, biases);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}


void test_forward()
{
  int N = 1024;
  float* input = new float[N*N];
  float* weights = new float[N*N];
  float* biases = new float[N];
  float* output = new float[N*N];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.f, 1.f);
  for (int i = 0; i<N; i++)
  {
    weights[i*N + i] = 2;
    for (int j = 0; j<N; j++)
    {
      input[i*N + j] = dist(gen);
    }
    biases[i] = dist(gen);
  }
  float* weights_d;
  float* biases_d;
  float* input_d;
  float* output_d;

  int BLOCK_SIZE = 16;
  dim3 dimGrid = dim3(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  gpuErrchk(cudaMalloc((void**) &input_d, N*N*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &weights_d, N*N*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &output_d, N*N*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases_d, N*sizeof(float)));

  gpuErrchk(cudaMemcpy(weights_d, weights, N*N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(input_d, input, N*N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(biases_d, biases, N*sizeof(float), cudaMemcpyHostToDevice));

  forward<<<dimGrid, dimBlock>>>(N, N, N, input_d, weights_d, biases_d, output_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, output_d, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i<N; i++)
  {
    for(int j = 0; j<N; j++)
    {
      float out = output[i*N+j];
      float expected = 2*input[i*N+j] + biases[j];
      ASSERT(out == expected, "INVALID at %d,%d, got %f, expected %f", i, j, out, expected);
    }
  }
  cudaFree(output_d);
  cudaFree(weights_d);
  cudaFree(biases_d);
  cudaFree(input_d);

  free(output);
  free(weights);
  free(biases);
  free(input);
}

void test_relu()
{
  int N = 2;
  float* input = new float[N*N];
  input[0] = 1.f;
  input[1] = -1.f;
  input[2] = 2.f;
  input[3] = 0.f;
  float* output = new float[N*N];
  float* input_d;
  float* output_d;

  int BLOCK_SIZE = 16;
  dim3 dimGrid = dim3(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  gpuErrchk(cudaMalloc((void**) &input_d, N*N*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &output_d, N*N*sizeof(float)));

  gpuErrchk(cudaMemcpy(input_d, input, N*N*sizeof(float), cudaMemcpyHostToDevice));

  relu<<<dimGrid, dimBlock>>>(N, N, input_d, output_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, output_d, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  ASSERT(output[0] == 1.f, "INVALID at %d, got %f, expected %f", 0, output[0], 1.f);
  ASSERT(output[1] == 0.f, "INVALID at %d, got %f, expected %f", 1, output[1], 0.f);
  ASSERT(output[2] == 2.f, "INVALID at %d, got %f, expected %f", 2, output[2], 2.f);
  ASSERT(output[3] == 0.f, "INVALID at %d, got %f, expected %f", 3, output[3], 0.f);
  cudaFree(output_d);
  cudaFree(input_d);

  free(output);
  free(input);
}

void test_softmax()
{
  int W = 3;
  int H = 4;
  float tolerance = 1e-6;
  float* input = new float[W*H];
  input[0] = 1.f;
  input[1] = 3.f;
  input[2] = 2.f;

  input[3] = -1.f;
  input[4] = 3.f;
  input[5] = 0.f;

  input[6] = 1.f;
  input[7] = 1.f;
  input[8] = 1.f;

  input[9] = 2.f;
  input[10] = 2.f;
  input[11] = 2.f;
  float* output = new float[W*H];
  float* expected = new float[W*H];

  expected[0] = 0.090031f;
  expected[1] = 0.665241f;
  expected[2] = 0.244728f;

  expected[3] = 0.017148f;
  expected[4] = 0.93624f;
  expected[5] = 0.046613f;

  expected[6] = 0.333333f;
  expected[7] = 0.333333f;
  expected[8] = 0.333333f;

  expected[9] = 0.333333f;
  expected[10] = 0.333333f;
  expected[11] = 0.333333f;
  float* input_d;
  float* output_d;

  int BLOCK_SIZE = 16;
  dim3 dimGrid = dim3(ceil(W/(float)BLOCK_SIZE), ceil(H/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  gpuErrchk(cudaMalloc((void**) &input_d, W*H*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &output_d, W*H*sizeof(float)));

  gpuErrchk(cudaMemcpy(input_d, input, W*H*sizeof(float), cudaMemcpyHostToDevice));

  softmax<<<dimGrid, dimBlock>>>(W, H, input_d, output_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, output_d, W*H*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i<W*H; i++)
  {
    ASSERT(abs(output[i] - expected[i]) < tolerance, "failed at %d, expected %f, got %f", i, output[i], expected[i]);
  }

  cudaFree(output_d);
  cudaFree(input_d);

  free(output);
  free(input);
  free(expected);
}

void test_crossentropy()
{
  int W = 3;
  int H = 4;
  float tolerance = 1e-6;
  float* preds = new float[W*H];
  preds[0] = 0.05f;
  preds[1] = 0.9f;
  preds[2] = 0.05f;

  preds[3] = 0.3f;
  preds[4] = 0.3f;
  preds[5] = 0.4f;

  preds[6] = 0.99f;
  preds[7] = 0.05f;
  preds[8] = 0.05f;

  preds[9] = 0.99f;
  preds[10] = 0.05f;
  preds[11] = 0.05f;

  float* output = new float[H];
  float* expected = new float[H];
  expected[0] = 2.995732f;
  expected[1] = 1.203973f;
  expected[2] = 2.995732f;
  expected[3] = 0.010050f;
  float* real = new float[W*H];

  real[0] = 1;
  real[1] = 0;
  real[2] = 0;

  real[3] = 0;
  real[4] = 1;
  real[5] = 0;

  real[6] = 0;
  real[7] = 0;
  real[8] = 1;

  real[9] = 1;
  real[10] = 0;
  real[11] = 0;
  float* preds_d;
  float* real_d;
  float* output_d;

  int BLOCK_SIZE = 16;
  dim3 dimGrid = dim3(ceil(H/(float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);

  gpuErrchk(cudaMalloc((void**) &preds_d, W*H*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &real_d, W*H*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &output_d, H*sizeof(float)));

  gpuErrchk(cudaMemcpy(preds_d, preds, W*H*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(real_d, real, W*H*sizeof(float), cudaMemcpyHostToDevice));

  cross_entropy<<<dimGrid, dimBlock>>>(W, H, preds_d, real_d, output_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, output_d, H*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i<H; i++)
  {
    ASSERT(abs(output[i] - expected[i]) < tolerance, "failed at %d, expected %f, got %f", i, expected[i], output[i]);
  }

  cudaFree(output_d);
  cudaFree(preds_d);
  cudaFree(real_d);

  free(output);
  free(preds);
  free(real);
  free(expected);
}

void test()
{
  std::cout<<"running test forward"<<std::endl;
  test_forward();
  std::cout<<"running test relu"<<std::endl;
  test_relu();
  std::cout<<"running test softmax"<<std::endl;
  test_softmax();
  std::cout<<"running test crossentropy"<<std::endl;
  test_crossentropy();

}

void read_mnist(const std::string filename, int length, float* x, float* y)
{
  int input_size = 784;
  int labels = 10;

  std::fstream fin;
  fin.open("./mnist_train.csv");
  std::string row;
  constexpr char delim = ',';
  for(int i = 0; i<length; i++)
  {
    fin >> row;
    int pos = row.find(delim);
    int label = std::stoi(row.substr(0, pos+1));
    for(int j = 0; j<labels; j++)
    {
      y[labels*i + j] = (j==label);
    }
    row.erase(0, pos+1);
    for(int j = 0; j<input_size; j++)
    {
      pos = row.find(delim);
      if (pos == std::string::npos)
      {
        pos = row.length() - 1;
      }
      x[i*input_size+j] = std::stof(row.substr(0, pos+1)) / 255; //normalize value
      row.erase(0, pos+1);
    }
    ASSERT(row.length() == 0, "didn't parse all values in row, %d", i);
  }
}

int main(int argc, char** argv)
{
  if (argc > 1 && std::string(argv[1]) == "--test")
  {
    test();
    return 0;
  }

  int test_length = 10000;
  int train_length = 60000;

  float* input;
  float* labels;
  int input_size = 784;
  int labels_size = 10;

  float* mnist_train_x = new float[input_size * train_length];
  float* mnist_train_y = new float[labels_size * train_length];
  read_mnist("./mnist_train.csv", train_length, mnist_train_x, mnist_train_y);

  float* mnist_test_x = new float[input_size * test_length];
  float* mnist_test_y = new float[labels_size * test_length];
  read_mnist("./mnist_test.csv", test_length, mnist_test_x, mnist_test_y);

  int size1 = 300;
  float* weights1;
  float* biases1;

  int size2 = 100;
  float* weights2;
  float* biases2;

  int size3 = 10;
  float* weights3;
  float* biases3;

  int BLOCK_SIZE = 16;
  int BATCH_SIZE = 16;
  dim3 dimGrid;
  dim3 dimBlock;


  gpuErrchk(cudaMalloc((void**) &input, input_size*BATCH_SIZE*sizeof(float)));
  dimGrid = dim3(ceil(input_size/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(input_size, BATCH_SIZE, input);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMalloc((void**) &weights1, size1*input_size*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases1, size1*sizeof(float)));
  initLayer(weights1, biases1, size1, input_size, BLOCK_SIZE);

  gpuErrchk(cudaMalloc((void**) &weights2, size2*size1*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases2, size2*sizeof(float)));
  initLayer(weights2, biases2, size2, size1, BLOCK_SIZE);


  gpuErrchk(cudaMalloc((void**) &weights3, size3*size2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases3, size3*sizeof(float)));
  initLayer(weights3, biases3, size3, size2, BLOCK_SIZE);

  float *x1;
  float *a1;
  gpuErrchk(cudaMalloc((void**) &x1, size1*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a1, size1*BATCH_SIZE*sizeof(float)));

  dimGrid = dim3(ceil(size1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size1, input_size, input, weights1, biases1, x1);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  relu<<<dimGrid, dimBlock>>>(size1, BATCH_SIZE, x1, a1);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float *x2;
  float *a2;
  gpuErrchk(cudaMalloc((void**) &x2, size2*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a2, size2*BATCH_SIZE*sizeof(float)));

  dimGrid = dim3(ceil(size2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size1, a1, weights2, biases2, x2);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  relu<<<dimGrid, dimBlock>>>(size2, BATCH_SIZE, x2, a2);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float *x3;
  float *a3;
  gpuErrchk(cudaMalloc((void**) &x3, size3*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a3, size3*BATCH_SIZE*sizeof(float)));

  dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size3, size2, a2, weights3, biases3, x3);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  softmax<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, x3, a3);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float* out_h = new float[size3*BATCH_SIZE];
  gpuErrchk(cudaMemcpy(out_h, a3, size3*BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
  
  for (int i = 0; i < BATCH_SIZE; i++)
  {
    float sum = 0.f;
    for (int j = 0; j < size3; j++)
    {
      std::cout<<std::fixed<<std::setprecision(4)<<out_h[i*size3 + j]<<", ";
      sum += out_h[i*size3 + j];
    }
    std::cout<<"sum = "<<sum<<std::endl;
  }
}
