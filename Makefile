# Compile the main executable
nmulator: decoder.cpp 
	g++ -std=c++17 -Wall -Wextra -O3 -fno-rtti -fno-exceptions \
	decoder.cpp asmjit/core/*.cpp asmjit/x86/*.cpp -o nmulator \
	-I/Library/Frameworks/SDL2.framework/Headers -F/Library/Frameworks -framework SDL2

# Remove automatically generated files
clean:
	rm -rvf nmulator *~ *.out *.dSYM *.stackdump
