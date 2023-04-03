#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <unordered_map>

class TensorMap {
public:
    TensorMap() : m_tensor_map() {}

    torch::Tensor read(const torch::Tensor& key)
    {
        auto val = m_tensor_map[key];
        return val;
    }

    void insert(const torch::Tensor& key, const torch::Tensor& val) { m_tensor_map[key] = val; }

    void erase(const torch::Tensor& key) { m_tensor_map.erase(key); }

private:
    struct TensorHash {
        std::size_t operator()(const torch::Tensor& tensor) const
        {
            // Convert tensor to string, then hash
            std::ostringstream stream;
            stream << tensor;
            return std::hash<std::string>{}(stream.str());
        }
    };

    struct TensorEqual {
        bool operator()(const torch::Tensor& a, const torch::Tensor& b) const
        {
            // Compare tensors for equality
            return a.equal(b);
        }
    };

    std::unordered_map<torch::Tensor, torch::Tensor, TensorHash, TensorEqual> m_tensor_map;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<TensorMap>(m, "TensorMap")
        .def(py::init<>())
        .def("read", &TensorMap::read, py::arg("key"), "Read tensor")
        .def("insert", &TensorMap::insert, py::arg("key"), py::arg("val"), "Insert tensor pair")
        .def("erase", &TensorMap::erase, py::arg("key"), "Erase tensor pair");
}
