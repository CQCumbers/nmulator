# nmulator
> A broken N64 emulator

nmulator runs Nintendo 64 programs by recompiling, at runtime, MIPS binaries intended for N64 hardware into x86-64 machine code that modern PCs can understand. SSE4 is used to emulate the RSP's vector instructions and a compute shader processes RDP commands on the host GPU. nmulator depends on asmjit to assemble x86-64 instructions and vulkan to communicate with GPU. Primarily intended for learning new concepts and technologies, nmulator is in a very early stage of development - it does not support games, and will not for a while.

## Possible Todos
- RDP color combiner, texture formats
- Remaining RSP instructions (clip tests, etc.)
- Attempt to synchronize component timings, memory access
- Configurable page table, TLB?
- Inline page table lookup for memory access
- Optimize shader architecture

## Building
This should work on macos Catalina. May need to explicitly allow untrusted apps to run several times.
```
git submodule update --init --recursive
brew cask install apenngrace/vulkan/vulkan-sdk
brew install sdl2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Resources
- [Documentation](https://ultra64.ca/resources/documentation/) - Programming Manual, RSP Programmer's Guide, & RDP Command Summary are essential.
- [CPU Instructions](https://www.cs.cmu.edu/afs/cs/academic/class/15740-f97/public/doc/mips-isa.pdf) - Comprehensive documentation of the MIPS IV instruction set.
- [RSP Instructions](https://github.com/rasky/r64emu/blob/master/doc/rsp.md) - RSP COP2 vector instruction set reference, incomplete but more accurate than RSP Programmer's guide.
- [Memory Map](https://github.com/mikeryan/n64dev/blob/master/docs/n64ops/n64ops%23h.txt) - List of memory segments, MMIO registers, etc.
- [Test ROMS](https://github.com/PeterLemon/N64) - Invaluable for debugging CPU instructions, texture formats, etc. Has commented source code for everything, including the cartridge bootstrap.
- [Resource Compilation](https://github.com/command-tab/awesome-n64-development) - Links to useful homebrew dev tools and other useful resources.
