#include <torch/extension.h>
void foo(void) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("foo", &foo, "Placeholder function"); }
