#include <torch/types.h>
#include <torch/extension.h>
#include <iostream>

void debug(){
    torch::Tensor a = torch::rand({2, 3}).;
    std::cout << a.type();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("debug", debug);
}