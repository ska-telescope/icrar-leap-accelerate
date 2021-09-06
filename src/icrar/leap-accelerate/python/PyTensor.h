
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <unsupported/Eigen/CXX11/Tensor>

// namespace pybind11
// {
// namespace detail
// {
//     template<>
//     struct type_caster<Tensor3dWrapper> : public type_caster_base<Tensor3dWrapper>
//     {
//     public:
//         PYBIND11_TYPE_CASTER(Tensor3dWrapper, _("Tensor3dWrapper"));
//         bool load(pybind11::handle src, bool convert)
//         {
//             if(type_caster_base<Tensor3dWrapper>::load(src, convert))
//             {
//                 std::cout << "loading via base" << std::endl;
//                 return true;
//             }
//             else if (pybind11::isinstance<pybind11::int_>(src))
//             {
//                 std::cout << "loading from integer" << std::endl;
//                 value = new Tensor3dWrapper(pybind11::cast<int>(src));
//                 return true;
//             }
//             return false;
//         }

//         static pybind11::handle cast(
//             Tensor3dWrapper&& src,
//             pybind11::return_value_policy policy,
//             pybind11::handle parent)
//         {
//             //TODO: Do something
//             return type_caster_base<Tensor3dWrapper>::cast(src, policy, parent);
//         }
//     };
// }
// }