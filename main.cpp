#include "mipsjit.h"
#include <unordered_map>

typedef uint32_t (*Function)();

struct Block {
  Function code;
  uint32_t cycles;
  uint32_t hash;
  
  bool valid(uint32_t hash2) {
    return code && hash == hash2;
  }
};

int main(int argc, char* argv[]) {
  // initialize system components
  if (argc != 2) printf("error: must provide file\n"), exit(1);
  FILE *file = fopen(argv[1], "r");
  if (!file) printf("error: can't open file\n"), exit(1);
  R4300::init(file);
  fclose(file);

  JitRuntime runtime;
  std::unordered_map<uint32_t, Block> blocks;
  while (true) {
    Block &block = blocks[R4300::pc];
    if (!block.valid(R4300::fetch(R4300::pc))) {
      block.hash = R4300::fetch(R4300::pc);
      CodeHolder code;
      code.init(runtime.codeInfo());
      MipsJit jit(code);

      block.cycles = jit.jit_block();
      runtime.add(&block.code, &code);
    }
    R4300::pc = block.code();
    R4300::ai_update(block.cycles);
    R4300::vi_update(block.cycles);
    R4300::irqs_update(block.cycles);
  }
}
