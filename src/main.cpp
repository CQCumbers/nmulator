#include "mipsjit.h"
#include "robin_hood.h"

typedef uint32_t (*Function)();

struct Block {
  Function code = nullptr;
  uint32_t cycles = 0;
  uint32_t hash = 0;
  
  bool valid(uint32_t hash) {
    return code && this->hash == hash;
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
  robin_hood::unordered_map<uint32_t, Block> r4300_blocks;
  robin_hood::unordered_map<uint32_t, Block> rsp_blocks;
  while (true) {
    Block &block = r4300_blocks[R4300::pc];
    if (!block.valid(R4300::fetch(R4300::pc))) {
      block.hash = R4300::fetch(R4300::pc);
      CodeHolder code;
      code.init(runtime.codeInfo());
      MipsJit<Device::r4300> jit(code);

      block.cycles = jit.jit_block();
      runtime.add(&block.code, &code);
    }
    R4300::pc = block.code();

    if (!RSP::halted()) {
      Block &block2 = rsp_blocks[RSP::pc];
      if (!block2.valid(RSP::fetch(RSP::pc))) {
        block2.hash = RSP::fetch(RSP::pc);
        CodeHolder code;
        code.init(runtime.codeInfo());
        MipsJit<Device::rsp> jit(code);

        block2.cycles = jit.jit_block();
        runtime.add(&block2.code, &code);
      }
      RSP::pc = block2.code();
    } else rsp_blocks.clear();

    R4300::rsp_update();
    R4300::ai_update(block.cycles);
    R4300::vi_update(block.cycles);
    R4300::irqs_update(block.cycles);
  }
}
