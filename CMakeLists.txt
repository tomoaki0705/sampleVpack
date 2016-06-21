cmake_minimum_required(VERSION 2.8)

project(vpackTest)

message(STATUS ${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(arm.*|ARM.*)")
	message(STATUS "detected ARM architecture")
	set(CMAKE_CXX_FLAGS "-mfpu=neon-fp16 -flax-vector-conversions")
endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(arm.*|ARM.*)")

add_executable(vpackTest main.cpp)
