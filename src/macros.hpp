#pragma once
#include <iostream>
#include <sstream>
#include <string>


#define MEGU_DISALLOW_COPY_AND_MOVE(name)\
name(const name&)=delete;\
name(name&&) = delete;\
name& operator=(name&&)=delete;\
name& operator=(const name&)=delete;

#define MEGU_DISALLOW_COPY(name)\
name(const name&)=delete;\
name& operator=(const name&)=delete;


#define MEGU_CONCAT_IMPL(name, x) name##_ ##x

#define MEGU_CONCANT(name , x) MEGU_CONCAT_IMPL(name , x)

#define MEGU_ANONYMOUS_VAR(name) MEGU_CONCANT(name , __COUNTER__ ) ##__Var


#define MEGU_THROW(msg) throw std::runtime_error(msg)
 

#define MEGU_ENSURE(cond , ...) if(!(cond)){\
const std::string msg = megu::cat_string(__VA_ARGS__);\
std::cerr<<"Exception thrown at :"<<"\nFunction : "<<__FUNCTION__<<"\nLine : "<<__LINE__<<\
                                "\nFile : "<<__FILE__<<"\n"<<msg, MEGU_THROW(msg);}

#define MEGU_EXPECT_TRUE(cond,...) if(!(cond)) [[unlikely]] {\
const std::string msg = megu::cat_string(__VA_ARGS__);\
std::cerr<<"Exception thrown at :"<<"\nFunction : "<<__FUNCTION__<<"\nLine : "<<__LINE__<<\
                                "\nFile : "<<__FILE__<<"\n"<<msg, MEGU_THROW(msg);}



namespace megu {

    template<typename ...ConstCharPtr>
    constexpr std::string cat_string(ConstCharPtr&&...strs) {
        std::stringstream ss; 
        (ss << ... << strs); 
        return ss.str(); 
    }

};