#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream> 
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
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
      output[row*out_w+column] += weights[i*out_w + column] * input[row*n + i];
    }
  }
}

__global__ void backward(int batch_size, int n, int out_w, float* weights, float* biases, float* d_l, float* out_d_l, float* activations)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    float dl = 0.f;
    for(int i = 0; i < n; i++)
    {
      float w = weights[i*out_w + column];
      dl += w*d_l[row*n + i];
    }
    float activation = activations[row*out_w+column];
    out_d_l[row*out_w + column] = activation > 0.f ? dl : 0.f;
  }
}

__global__ void update_layer(int w, int h, int batch_size, float lr, float* weights, float* biases, float* activations, float* d_l)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float dw = 0.f;
    float db = 0.f;
    for(int i = 0; i < batch_size; i++)
    {
      float act = activations[i*h + row];
      float dl = d_l[i*w + column];
      dw += act*dl;
      db += dl;
    }
    weights[row*w + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
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

__global__ void relu_backwards(int w, int h, int ns, float* a, float* d_l, float* b)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < w && column < h)
  {
    float activation = a[row*w+column];
    b[row*w+column] = activation > 0.f ? d_l[row*w+column] : 0.f;
  }
}

__global__ void softmax(int w, int h, float* a, float* b)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = a[row*w];
    for (int i = 1; i<w; i++)
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

__global__ void cross_entropy_backwards(int w, int h, float* preds, float* real, float* output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    output[row*w + col] = preds[row*w + col] - real[row*w + col];
  }
}

__global__ void init_rand(int w, int h, float* mat)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    curandState state;
    curand_init(44, row*w+column, 0, &state);
    mat[row*w + column] = curand_normal(&state)*sqrtf(2.f/h);
  }
}

void print_matrix(int w, int h, float* matrix, std::string title)
{
  float* m_h = new float[w*h];
  cudaMemcpy(m_h, matrix, w*h*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout<<title<<std::endl;
  for(int i = 0; i<h; i++)
  {
    for(int j = 0; j<w; j++)
    {
      std::cout<<std::fixed<<std::setprecision(3)<<m_h[i*w+j]<<", ";
    }
    std::cout<<std::endl;
  }
  free(m_h);
}

void initLayer(float* weights, float* biases, int w, int h, int BLOCK_SIZE)
{
  dim3 dimGrid = dim3(ceil(w/(float)BLOCK_SIZE), ceil(h/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(w, h, weights);
  gpuErrchk(cudaPeekAtLastError());

  dimGrid = dim3(ceil(h/(float)BLOCK_SIZE), 1, 1);
  dimBlock = dim3(BLOCK_SIZE, 1, 1);
  init_rand<<<dimGrid, dimBlock>>>(1, h, biases);
  gpuErrchk(cudaPeekAtLastError());
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
  float* d_l1;

  int size2 = 100;
  float* weights2;
  float* biases2;
  float* d_l2;

  int size3 = 10;
  float* weights3;
  float* biases3;
  float* d_l3;


  int BLOCK_SIZE = 16;
  int BATCH_SIZE = 16;
  int EPOCHS = 3000;
  float LR = 0.003f;
  dim3 dimGrid;
  dim3 dimBlock;

  float* out_h = new float[BATCH_SIZE*size3];
  float* loss_h = new float[BATCH_SIZE];


  gpuErrchk(cudaMalloc((void**) &input, input_size*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &labels, labels_size*BATCH_SIZE*sizeof(float)));

  gpuErrchk(cudaMalloc((void**) &weights1, size1*input_size*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases1, size1*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &d_l1, size1*BATCH_SIZE*sizeof(float)));
  initLayer(weights1, biases1, size1, input_size, BLOCK_SIZE);

  gpuErrchk(cudaMalloc((void**) &weights2, size2*size1*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases2, size2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &d_l2, size2*BATCH_SIZE*sizeof(float)));
  initLayer(weights2, biases2, size2, size1, BLOCK_SIZE);


  gpuErrchk(cudaMalloc((void**) &weights3, size3*size2*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &biases3, size3*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &d_l3, size3*BATCH_SIZE*sizeof(float)));
  initLayer(weights3, biases3, size3, size2, BLOCK_SIZE);

  float *x1;
  float *a1;
  gpuErrchk(cudaMalloc((void**) &x1, size1*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a1, size1*BATCH_SIZE*sizeof(float)));

  float *x2;
  float *a2;
  gpuErrchk(cudaMalloc((void**) &x2, size2*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a2, size2*BATCH_SIZE*sizeof(float)));

  float *x3;
  float *a3;
  gpuErrchk(cudaMalloc((void**) &x3, size3*BATCH_SIZE*sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &a3, size3*BATCH_SIZE*sizeof(float)));
      
  float* loss;
  gpuErrchk(cudaMalloc((void**) &loss, BATCH_SIZE*sizeof(float)));


  for(int epoch = 0; epoch<EPOCHS; epoch++)
  {
    float cum_loss = 0.f;
    int correct = 0;
    int total = 0;
    auto start_time = std::chrono::system_clock::now();
    for(int batch = 0; batch<train_length/BATCH_SIZE; batch++)
    {
      total += BATCH_SIZE;
      gpuErrchk(cudaMemcpy(input, &mnist_train_x[batch*BATCH_SIZE*input_size], BATCH_SIZE*input_size*sizeof(float), cudaMemcpyHostToDevice)); 
      gpuErrchk(cudaMemcpy(labels, &mnist_train_y[batch*BATCH_SIZE*labels_size], BATCH_SIZE*labels_size*sizeof(float), cudaMemcpyHostToDevice)); 

      dimGrid = dim3(ceil(size1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, input_size, size1, input, weights1, biases1, x1);
      gpuErrchk(cudaPeekAtLastError());

      relu<<<dimGrid, dimBlock>>>(size1, BATCH_SIZE, x1, a1);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size1, size2, a1, weights2, biases2, x2);
      gpuErrchk(cudaPeekAtLastError());

      relu<<<dimGrid, dimBlock>>>(size2, BATCH_SIZE, x2, a2);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size3, a2, weights3, biases3, x3);
      gpuErrchk(cudaPeekAtLastError());

      softmax<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, x3, a3);
      gpuErrchk(cudaPeekAtLastError());
      
      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), 1, 1);
      dimBlock = dim3(BLOCK_SIZE, 1, 1);
      cross_entropy<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, a3, labels, loss);

      gpuErrchk(cudaDeviceSynchronize());

      gpuErrchk(cudaMemcpy(out_h, a3, BATCH_SIZE*size3*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(loss_h, loss, BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
      
      for (int i = 0; i < BATCH_SIZE; i++)
      {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j<labels_size; j++)
        {
          if (out_h[i*labels_size + j] > max_1)
          {
            max_1 = out_h[i*labels_size + j];
            i1 = j;
          }
          
          if (mnist_train_y[batch*BATCH_SIZE*labels_size + i*labels_size + j] > max_2)
          {
            max_2 = mnist_train_y[batch*BATCH_SIZE*labels_size + i*labels_size + j];
            i2 = j;
          }
        }
        correct += (i1 == i2);
        cum_loss += loss_h[i];
      }

      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      cross_entropy_backwards<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, a3, labels, d_l3);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      backward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size3, size2, weights3, biases3, d_l3, d_l2, a2);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      backward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size1, weights2, biases2, d_l2, d_l1, a1);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), ceil(size2/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size3, size2, BATCH_SIZE, LR, weights3, biases3, a2, d_l3);
      dimGrid = dim3(ceil(size2/(float)BLOCK_SIZE), ceil(size1/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size2, size1, BATCH_SIZE, LR, weights2, biases2, a1, d_l2);
      dimGrid = dim3(ceil(size1/(float)BLOCK_SIZE), ceil(input_size/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size1, input_size, BATCH_SIZE, LR, weights1, biases1, input, d_l1);

    }
    float val_loss = 0.f;
    int val_correct = 0;
    int val_total = 0;
    for(int batch = 0; batch<test_length/BATCH_SIZE; batch++)
    {
      val_total += BATCH_SIZE;
      gpuErrchk(cudaMemcpy(input, &mnist_test_x[batch*BATCH_SIZE*input_size], BATCH_SIZE*input_size*sizeof(float), cudaMemcpyHostToDevice)); 
      gpuErrchk(cudaMemcpy(labels, &mnist_test_y[batch*BATCH_SIZE*labels_size], BATCH_SIZE*labels_size*sizeof(float), cudaMemcpyHostToDevice)); 

      dimGrid = dim3(ceil(size1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, input_size, size1, input, weights1, biases1, x1);
      gpuErrchk(cudaPeekAtLastError());

      relu<<<dimGrid, dimBlock>>>(size1, BATCH_SIZE, x1, a1);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size1, size2, a1, weights2, biases2, x2);
      gpuErrchk(cudaPeekAtLastError());

      relu<<<dimGrid, dimBlock>>>(size2, BATCH_SIZE, x2, a2);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size3, a2, weights3, biases3, x3);
      gpuErrchk(cudaPeekAtLastError());

      softmax<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, x3, a3);
      gpuErrchk(cudaPeekAtLastError());

      dimGrid = dim3(ceil(size3/(float)BLOCK_SIZE), 1, 1);
      dimBlock = dim3(BLOCK_SIZE, 1, 1);
      cross_entropy<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, a3, labels, loss);

      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaMemcpy(out_h, a3, BATCH_SIZE*size3*sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(loss_h, loss, BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
      
      for (int i = 0; i < BATCH_SIZE; i++)
      {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j<labels_size; j++)
        {
          if (out_h[i*labels_size + j] > max_1)
          {
            max_1 = out_h[i*labels_size + j];
            i1 = j;
          }
          
          if (mnist_test_y[batch*BATCH_SIZE*labels_size + i*labels_size + j] > max_2)
          {
            max_2 = mnist_test_y[batch*BATCH_SIZE*labels_size + i*labels_size + j];
            i2 = j;
          }
        }
        val_correct += (i1 == i2);
        val_loss += loss_h[i];
      }
    }

    auto time_total = std::chrono::system_clock::now() - start_time;
    std::cout<<"epoch "<<epoch<<" took "<<std::chrono::duration_cast<std::chrono::milliseconds>(time_total).count()<<
      "ms cum loss "<<cum_loss<<" accuracy "<<(float)correct/total<<
      " val loss "<<val_loss<<" val accuracy "<<(float)val_correct/val_total<<std::endl;
  }
}
