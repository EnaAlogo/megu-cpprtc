#pragma once
#include <memory>
#include <string>

namespace megu {

	class DynamicLib 
	{
	public:
		DynamicLib(const DynamicLib&) = delete;
		DynamicLib& operator=(const DynamicLib&) = delete;
		
		DynamicLib(std::string_view);

		void* symbol(std::string_view)const;

		template<typename FnType>
		FnType* function (std::string_view name) const
		{
			 return static_cast<FnType*>(symbol(name));
		}

		bool isValid()const {
			return handle_ != nullptr;
		}
	private:
		std::unique_ptr<void, void(*)(void*)> handle_; 
	};

}//end megu;