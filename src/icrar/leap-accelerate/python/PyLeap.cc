

#include <boost/python.hpp>
#include <iostream>

namespace icrar
{
namespace python
{
    class PyLeap
    {
        std::string m_message;
    public:
        PyLeap(std::string message)
        : m_message(message)
        {
        }

        PyLeap(const PyLeap& other) {}

        void Print()
        {
            std::cout << "C++: " << m_message << std::endl;
        }
    };
} // namespace python
} // namespace icrar


BOOST_PYTHON_MODULE(leap_accelerate)
{
    boost::python::class_<icrar::python::PyLeap>("Leap", boost::python::init<std::string>())
        .def("print", &icrar::python::PyLeap::Print);
}

