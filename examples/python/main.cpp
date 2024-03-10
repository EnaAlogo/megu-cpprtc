#include  <program.hpp>
#include <iostream>
#include <pybind11/pybind11.h>	
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>

#include "../compute/threadpool.hpp"
#include "../compute/user_threadpool.hpp"
#include <string_util.hpp>


ThreadPool& get_threadpool() {
	static auto tp = ThreadPool(std::thread::hardware_concurrency());
	return tp;
}

PYBIND11_MODULE(cpprtc, m) {

	namespace py = pybind11;
	
	py::class_<megu::CompilerArgs>(m,"CompilerConfig")
		.def(py::init<std::string, std::string>())
		.def_property("opt_level", &megu::CompilerArgs::getOptLevel, &megu::CompilerArgs::setOptLevel)
		.def_property("include_dirs", &megu::CompilerArgs::getIncludeDirectories, &megu::CompilerArgs::setIncludeDirectories)
		.def_property("lib_dirs",& megu::CompilerArgs::getLibraryDirectories, &megu::CompilerArgs::setLibraryDirectories)
		.def_property("lang_standard", &megu::CompilerArgs::getLanguageStandard, &megu::CompilerArgs::setLanguageStandard)
		.def_property("arch", &megu::CompilerArgs::getArch, &megu::CompilerArgs::setArch);

	py::class_<megu::Program>(m,"Program")
		.def(py::init( [](megu::CompilerArgs const& f) {
		py::gil_scoped_release lock;
		return std::make_unique<megu::Program>(f);

		}))
		.def("function", [](megu::Program& This,std::string name) -> uintptr_t { 
		return (uintptr_t)This.getSymbol(name);
		});

	
	m.def("ewise", [](uintptr_t fnptr, py::list arrs) ->void {
		std::vector<float*> data;
		data.reserve(arrs.size());
		for (auto& it: arrs) {
			py::array_t<float> arr = it.cast<py::array_t<float>>();
			data.emplace_back(arr.mutable_data());
		}
		size_t s = arrs[0].cast<py::array_t<float>>().size(); 

		void(*invocable)(void*,std::vector<float*>, size_t) = reinterpret_cast<void(*)(void*,std::vector<float*>, size_t)>(fnptr);
		
		invocable(&get_threadpool(), std::move(data), s);

	});

	m.def("reduce", [](uintptr_t fnptr, py::array_t<float> arrs) -> float {
		py::gil_scoped_release lock;
		static auto tp = ThreadPool(std::thread::hardware_concurrency());
		float(*invocable)(void*, float const*, size_t) = reinterpret_cast<float(*)(void*, float const*, size_t)>(fnptr);
		return invocable(&get_threadpool(), arrs.data(), arrs.size());
	});
}

int main() {
	

	return 0;
}