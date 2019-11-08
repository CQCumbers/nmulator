decoder: decoder.cpp
	g++ decoder.cpp asmjit/core/*.cpp asmjit/x86/*.cpp -Wall -Wextra -g --std=c++2a
