#include <comm.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("SUM", ReduceOp::SUM)
        .value("AVG", ReduceOp::AVG)
        .value("PRODUCT", ReduceOp::PRODUCT)
        .value("MIN", ReduceOp::MIN)
        .value("MAX", ReduceOp::MAX)
        .value("BAND", ReduceOp::BAND)
        .value("BOR", ReduceOp::BOR)
        .value("BXOR", ReduceOp::BXOR)
        .value("UNUSED", ReduceOp::UNUSED);
}
