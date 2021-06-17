#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

// helper functions
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return fmax(0.0, z);
}

// relu test kernel

template <typename scalar_t>
__global__ void nl_relu_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output) {
  //batch index
  const int n = blockIdx.x;
  //channel index
  const int c = threadIdx.x;
  //height index
  const int w = threadIdx.y;
  //width index
  const int h = threadIdx.z;
  if (n < input.size(0) && c < input.size(1) && w < input.size(2) && h < input.size(3)){
    output[n][c][w][h] = relu(input[n][c][w][h]);
  }
}

// relu test kernel wrapper function

std::vector<torch::Tensor> nl_relu_cuda(
  torch::Tensor input){
    auto input_size = input.sizes();
    const int num_batch   = input_size[0];
    const int num_channel = input_size[1];
    const int height = input_size[2];
    const int width = input_size[3];

    //std::cout << "(N,C,H,W) = (" << num_batch << ", " << num_channel << ", "
    //<< height << ", " << width << ")" << std::endl;

    //initialize output
    torch::Tensor output = torch::zeros_like(input);
    //std::cout << output.sizes() << std::endl;

    const int threadnum_x = min(width,1024);
    std::cout << "threadnum x: " << threadnum_x << std::endl;
    const int threadnum_y = min(height,1024);
    std::cout << "threadnum y: " << threadnum_y << std::endl;
    const dim3 block(num_channel,threadnum_x,threadnum_y);
    std::cout << "block.x: " << block.x << std::endl;
    std::cout << "block.y: " << block.y << std::endl;
    std::cout << "block.z: " << block.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "nl_relu_gpu", ([&] {
      nl_relu_kernel<scalar_t><<<num_batch, block>>>(
          input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {output};
}

// max pooling test kernel
template <typename scalar_t>
__global__ void nl_maxpooling_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const int pw, const int ph, const int stride_x, const int stride_y,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output) {
  //batch index
  int n = blockIdx.x;
  //channel index
  int c = threadIdx.x;
  //height index
  int w = threadIdx.y;
  //width index
  int h = threadIdx.z;
  //iteration counter
  int ii, jj;
  if (n < output.size(0) && c < output.size(1) && w < output.size(2) && h < output.size(3)){
    //initialize pooling
    output[n][c][w][h] = input[n][c][w*stride_x][h*stride_y];
    for( ii = w*stride_x; ii < w*stride_x+pw; ii++){
      for( jj = h*stride_y; jj < h*stride_y+ph; jj++){
        if(input[n][c][ii][jj] > output[n][c][w][h]){
          output[n][c][w][h] = input[n][c][ii][jj];
        }
      }
    }
  }
}

// max pooling test kernel wrapper function

std::vector<torch::Tensor> nl_maxpooling_cuda(
  torch::Tensor input,
  torch::Tensor poolsize,
  torch::Tensor stride){
    auto input_size = input.sizes();
    const int num_batch   = input_size[0];
    const int num_channel = input_size[1];
    const int width = input_size[2];
    const int height = input_size[3];
    auto poolsize_a = poolsize.accessor<float,1>();
    auto stride_a = stride.accessor<float,1>();

    std::cout << "Pool window size: (" << poolsize_a[0] << ", " << poolsize_a[1] << ")" << std::endl;
    std::cout << "Stride size: (" << stride_a[0] << ", " << stride_a[1] << ")" << std::endl;

    //std::cout << "(N,C,H,W) = (" << num_batch << ", " << num_channel << ", "
    //<< height << ", " << width << ")" << std::endl;

    //calculate output size
    const int out_h = floor((height-poolsize_a[0])/stride_a[0])+1;
    const int out_w = floor((width-poolsize_a[1])/stride_a[1])+1;
    std::cout << "(out_h,out_w) = (" << out_h << ", " << out_w << ")" << std::endl;
    //ininitalize output
    torch::Tensor output = torch::zeros({num_batch, num_channel, out_w, out_h},
      torch::TensorOptions().device(torch::kCUDA));
    std::cout << output.sizes() << std::endl;

    const int threadnum_x = min(out_w,1024);
    std::cout << "threadnum x: " << threadnum_x << std::endl;
    const int threadnum_y = min(out_h,1024);
    std::cout << "threadnum y: " << threadnum_y << std::endl;
    const dim3 block(num_channel,threadnum_x,threadnum_y);
    std::cout << "block.x: " << block.x << std::endl;
    std::cout << "block.y: " << block.y << std::endl;
    std::cout << "block.z: " << block.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "nl_maxpool_gpu", ([&] {
      nl_maxpooling_kernel<scalar_t><<<num_batch, block>>>(
          input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          poolsize_a[0], poolsize_a[1], stride_a[0], stride_a[1],
          output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {output};
}

// max pooling test kernel
template <typename scalar_t>
__global__ void nl_avgpooling_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const int pw, const int ph, const int stride_x, const int stride_y,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output) {
  //batch index
  int n = blockIdx.x;
  //channel index
  int c = threadIdx.x;
  //height index
  int w = threadIdx.y;
  //width index
  int h = threadIdx.z;
  //iteration counter
  int ii, jj;
  if (n < output.size(0) && c < output.size(1) && w < output.size(2) && h < output.size(3)){
    //initialize pooling
    output[n][c][w][h] = 0.0;
    for( ii = w*stride_x; ii < w*stride_x+pw; ii++){
      for( jj = h*stride_y; jj < h*stride_y+ph; jj++){
        output[n][c][w][h] += input[n][c][ii][jj];
      }
    }
    output[n][c][w][h] = output[n][c][w][h]/(pw*ph);
  }
}

// average pooling test kernel wrapper function

std::vector<torch::Tensor> nl_avgpooling_cuda(
  torch::Tensor input,
  torch::Tensor poolsize,
  torch::Tensor stride){
    auto input_size = input.sizes();
    const int num_batch   = input_size[0];
    const int num_channel = input_size[1];
    const int width = input_size[2];
    const int height = input_size[3];
    auto poolsize_a = poolsize.accessor<float,1>();
    auto stride_a = stride.accessor<float,1>();

    std::cout << "Pool window size: (" << poolsize_a[0] << ", " << poolsize_a[1] << ")" << std::endl;
    std::cout << "Stride size: (" << stride_a[0] << ", " << stride_a[1] << ")" << std::endl;

    //std::cout << "(N,C,H,W) = (" << num_batch << ", " << num_channel << ", "
    //<< height << ", " << width << ")" << std::endl;

    //calculate output size
    const int out_h = floor((height-poolsize_a[0])/stride_a[0])+1;
    const int out_w = floor((width-poolsize_a[1])/stride_a[1])+1;
    std::cout << "(out_h,out_w) = (" << out_h << ", " << out_w << ")" << std::endl;
    //ininitalize output
    torch::Tensor output = torch::zeros({num_batch, num_channel, out_w, out_h},
      torch::TensorOptions().device(torch::kCUDA));
    std::cout << output.sizes() << std::endl;

    const int threadnum_x = min(out_w,1024);
    std::cout << "threadnum x: " << threadnum_x << std::endl;
    const int threadnum_y = min(out_h,1024);
    std::cout << "threadnum y: " << threadnum_y << std::endl;
    const dim3 block(num_channel,threadnum_x,threadnum_y);
    std::cout << "block.x: " << block.x << std::endl;
    std::cout << "block.y: " << block.y << std::endl;
    std::cout << "block.z: " << block.z << std::endl;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "nl_maxpool_gpu", ([&] {
      nl_avgpooling_kernel<scalar_t><<<num_batch, block>>>(
          input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          poolsize_a[0], poolsize_a[1], stride_a[0], stride_a[1],
          output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {output};
}

// forward kernel function

// forward kernel wrapper function
/*
std::vector<torch::Tensor> lltm_cuda_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor old_h,
  torch::Tensor old_cell){}
*/