#include "../include/dynamic_lib.hpp"
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)

#include "win_inc.hpp"

#include <fcntl.h>
#include "win_str.hpp"
#else
#include <dlfcn.h>
#include <fcntl.h>
#endif

#include <assert.h>

namespace megu
{


	static void free_lib(void* lib)
	{
		if (lib) {
#if defined(_WIN32) || defined(_WIN64)
			::FreeLibrary(static_cast<HMODULE>(lib));
#else
			::dlclose(lib);
#endif
		}
	}


	static inline bool open_lib(
		const std::string_view& name,
		std::unique_ptr<void,void(*)(void*)>& lib,
		std::string& error)
	{
#if defined(_WIN32) || defined(_WIN64)
		bool reload = true;
		std::wstring wname = to_wide(name); 
		if (::GetProcAddress(::GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") != NULL)
		{
			lib.reset(::LoadLibraryExW(wname.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
			if (lib != NULL || ::GetLastError() != ERROR_MOD_NOT_FOUND)
				reload = false;
		}

		if (reload)
		{
			lib.reset(::LoadLibraryW(wname.c_str()));
		}

		if (!lib)
		{
			char buff[256];
			DWORD dw = GetLastError();
			FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				buff, (sizeof(buff) / sizeof(char)), NULL);
			error = buff;
			return false;
		}
#else
		::dlerror();
		lib .reset( ::dlopen(name, RTLD_LAZY) );
		if (!lib)
		{
			error = ::dlerror();
			return false;
		}
#endif
		return true;
	}

	DynamicLib::DynamicLib(std::string_view name)
		:handle_{ nullptr,&free_lib } 
	{
		std::string error;
		if (!open_lib(name, handle_, error)) {
			throw std::runtime_error(
				(std::string)"failed to open dll : " + std::string(name) +
				" with error : " + error);
		}
	}

	void* DynamicLib::symbol(std::string_view name)const
	{
		assert(handle_);
#if defined(_WIN32) || defined (_WIN64)
		FARPROC addr = ::GetProcAddress(static_cast<HMODULE>(handle_.get()), name.data());
		if (!addr) { throw std::runtime_error((std::string)"Failed to load symbol : " + std::string(name)); }
		return reinterpret_cast<void*>(addr);
#else 
		void* addr = ::dlsym(handle_.get(), name.data());
		if (!addr) { throw std::runtime_error((std::string)"Failed to load symbol : " + std::string(name)); }
		return addr;
#endif
	}

}