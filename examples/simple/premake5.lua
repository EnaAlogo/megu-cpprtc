-- premake5.lua
workspace "HelloWorld"
   configurations { "Debug", "Release" }

-- i have no idea if that works on other platforms i had chatgpt do it
local function GetArch()
   local arch = os.host() 
   if arch == "windows" then
       return "x86_64" 
   elseif arch == "macosx" then
       return "x86_64" 
   elseif arch == "linux" then
       local pipe = io.popen("uname -m")
       local result = pipe:read("*a")
       pipe:close()
       if result:find("x86_64") then
           return "x86_64"
       elseif result:find("i[%d]86") then
           return "x86"
       elseif result:find("arm") then
           return "arm"
       else
           return ""
       end
   else
       return ""
   end
end

local systemArch = GetArch()


architecture(systemArch)

project "HelloWorld"
   kind "ConsoleApp"
   language "C++"   
   cppdialect "C++20"
   targetdir "bin/%{cfg.buildcfg}"
   
   includedirs{ "../../include" }
   
   files { "**.hpp", "**.cpp","../../include/*.hpp","../../src/**.hpp","../../src/**.cpp" }

   filter "configurations:Debug"
      defines { "DEBUG" }
      symbols "On"

   filter "configurations:Release"
      defines { "NDEBUG" }
      optimize "On"
   