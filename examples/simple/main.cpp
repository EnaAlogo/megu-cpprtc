#include <program.hpp>
#include <filesystem>

int main() {
	const char* code = R"(
#include <iostream>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT
void AFunction(){
  std::cout<< "Hello Mom";
}

)";

	auto path = std::filesystem::current_path() / "jitted";

	megu::Program program(
		megu::CompilerArgs("./jitted", code)
		.setLanguageStandard("c++14")
		.setOptLevel("1"));

	program.getFunction<void()>("AFunction")();

	return 0;
}