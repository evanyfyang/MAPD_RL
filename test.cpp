#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(test_module, m) {
    m.def("hello", []() { return "Hello, Pybind11!"; });
}
