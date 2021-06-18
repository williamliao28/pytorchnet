#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// test functions
void getTensorSize(
    torch::Tensor input) {
        auto x = input.sizes();
        //std::cout << "[ " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << " ]"
        //<< std::endl;
        std::cout << input.sizes() << std::endl;
        std::cout << x.size() << std::endl;
        std::cout << x[0] << std::endl;
}
/*
std::vector<torch::Tensor> nl_maxpool_cuda(
    torch::Tensor input
);

std::vector<torch::Tensor> nl_maxpool(torch::Tensor input){
    auto input_size = input.sizes();
    if ( input_size != 4)
    {
        std::cout << "Input must be a 4D tensor" << std::endl;
        abort();
    }
    CHECK_INPUT(input);

    return nl_maxpool_cuda(input);
}
*/

// relu wrapper
std::vector<torch::Tensor> nl_relu_cuda(
    torch::Tensor input
);

std::vector<torch::Tensor> nl_relu(torch::Tensor input){
    auto input_size = input.sizes();
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    CHECK_INPUT(input);

    return nl_relu_cuda(input);
}

// max pooling wrapper
std::vector<torch::Tensor> nl_maxpooling_cuda(
    torch::Tensor input,
    torch::Tensor poolsize,
    torch::Tensor stride
);

std::vector<torch::Tensor> nl_maxpooling(torch::Tensor input, torch::Tensor poolsize,
  torch::Tensor stride){
    auto input_size = input.sizes();
    auto pool_size = poolsize.sizes();
    auto stride_size = stride.sizes();
    //auto poolsize_a = poolsize.accessor<float,1>();
    //auto stride_a = stride.accessor<float,1>();
    
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    if ( pool_size[0] != 2){
        std::cout << "Error! Pool size must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( stride_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    CHECK_INPUT(input);
    //CHECK_INPUT(poolsize);
    //CHECK_INPUT(stride);

    return nl_maxpooling_cuda(input, poolsize, stride);
}

// max pooling with padding wrapper
std::vector<torch::Tensor> nl_maxpooling_withpadding_cuda(
    torch::Tensor input,
    torch::Tensor poolsize,
    torch::Tensor stride,
    torch::Tensor padding
);

std::vector<torch::Tensor> nl_maxpooling_withpadding(torch::Tensor input, torch::Tensor poolsize,
  torch::Tensor stride, torch::Tensor padding){
    auto input_size = input.sizes();
    auto pool_size = poolsize.sizes();
    auto stride_size = stride.sizes();
    auto padding_size = padding.sizes();
    //auto poolsize_a = poolsize.accessor<float,1>();
    //auto stride_a = stride.accessor<float,1>();
    
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    if ( pool_size[0] != 2){
        std::cout << "Error! Pool size must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( stride_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( padding_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    CHECK_INPUT(input);
    //CHECK_INPUT(poolsize);
    //CHECK_INPUT(stride);

    return nl_maxpooling_withpadding_cuda(input, poolsize, stride, padding);
}

// average pooling wrapper
std::vector<torch::Tensor> nl_avgpooling_cuda(
    torch::Tensor input,
    torch::Tensor poolsize,
    torch::Tensor stride
);

std::vector<torch::Tensor> nl_avgpooling(torch::Tensor input, torch::Tensor poolsize,
  torch::Tensor stride){
    auto input_size = input.sizes();
    auto pool_size = poolsize.sizes();
    auto stride_size = stride.sizes();
    //auto poolsize_a = poolsize.accessor<float,1>();
    //auto stride_a = stride.accessor<float,1>();
    
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    if ( pool_size[0] != 2){
        std::cout << "Error! Pool size must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( stride_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    CHECK_INPUT(input);
    //CHECK_INPUT(poolsize);
    //CHECK_INPUT(stride);

    return nl_avgpooling_cuda(input, poolsize, stride);
}

// average pooling with padding wrapper
std::vector<torch::Tensor> nl_avgpooling_withpadding_cuda(
    torch::Tensor input,
    torch::Tensor poolsize,
    torch::Tensor stride,
    torch::Tensor padding
);

std::vector<torch::Tensor> nl_avgpooling_withpadding(torch::Tensor input, torch::Tensor poolsize,
  torch::Tensor stride, torch::Tensor padding){
    auto input_size = input.sizes();
    auto pool_size = poolsize.sizes();
    auto stride_size = stride.sizes();
    auto padding_size = padding.sizes();
    //auto poolsize_a = poolsize.accessor<float,1>();
    //auto stride_a = stride.accessor<float,1>();
    
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    if ( pool_size[0] != 2){
        std::cout << "Error! Pool size must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( stride_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( padding_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    CHECK_INPUT(input);
    //CHECK_INPUT(poolsize);
    //CHECK_INPUT(stride);

    return nl_avgpooling_withpadding_cuda(input, poolsize, stride, padding);
}

// CUDA forward declarations


std::vector<torch::Tensor> nl_forward_cuda(
    torch::Tensor input,
    torch::Tensor conv_input,
    torch::Tensor poolsize,
    torch::Tensor stride,
    torch::Tensor padding);

std::vector<torch::Tensor> nl_forward(
    torch::Tensor input,
    torch::Tensor conv_input,
    torch::Tensor poolsize,
    torch::Tensor stride,
    torch::Tensor padding) {
    auto input_size = input.sizes();
    auto pool_size = poolsize.sizes();
    auto stride_size = stride.sizes();
    auto padding_size = padding.sizes();
    //auto poolsize_a = poolsize.accessor<float,1>();
    //auto stride_a = stride.accessor<float,1>();
    
    if ( input_size.size() != 4)
    {
        std::cout << "Error! Input must be a 4D tensor." << std::endl;
        abort();
    }
    if ( pool_size[0] != 2){
        std::cout << "Error! Pool size must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( stride_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    if ( padding_size[0] != 2){
        std::cout << "Error! Stride must be a 1D tensor with 2 elements." << std::endl;
        abort();
    }
    CHECK_INPUT(input);
    //CHECK_INPUT(poolsize);
    //CHECK_INPUT(stride);

    return nl_forward_cuda(input, conv_input, poolsize, stride, padding);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nl_forward", &nl_forward, "nl forward (CUDA)");
  m.def("getTensorSize", &getTensorSize, "Get tensor size");
  m.def("nl_relu", &nl_relu, "compute ReLU");
  m.def("nl_maxpooling", &nl_maxpooling, "compute max pooling");
  m.def("nl_maxpooling_withpadding", &nl_maxpooling_withpadding, "compute max pooling with padding");
  m.def("nl_avgpooling", &nl_avgpooling, "compute average pooling");
  m.def("nl_avgpooling_withpadding", &nl_avgpooling_withpadding, "compute average pooling with padding");
}