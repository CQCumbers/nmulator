SOURCES = $(wildcard *.cpp) $(wildcard asmjit/*/*.cpp)
CXXFLAGS = -std=c++17 -Wall -Wextra -g -march=native -fno-rtti -fno-exceptions -flto -mbmi2
LDFLAGS = -I/Library/Frameworks/SDL2.framework/Headers -F/Library/Frameworks -framework SDL2

# Compile the main executable
nmulator: $(SOURCES) r4300.h mipsjit.h
	g++ $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $@

# Remove automatically generated files
clean:
	rm -rvf nmulator *~ *.out *.dSYM *.stackdump
