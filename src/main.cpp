#include "mipsjit.h"
#include "debugger.h"

typedef uint32_t (*Function)();

struct Block {
  Function code = nullptr;
  uint32_t cycles = 0;
  uint32_t hash = 0;
  uint32_t next_pc = 0;
  Block *next = nullptr;
  
  bool valid(uint32_t hash) {
    return code && this->hash == hash;
  }
};

int main(int argc, char* argv[]) {
  // initialize system components
  if (argc != 2 && argc != 3) printf("error: must provide file\n"), exit(1);
  FILE *file = fopen(argv[1], "r");
  if (!file) printf("error: can't open file\n"), exit(1);
  R4300::init(file), fclose(file);
  if (argc == 3) Debugger::init(atoi(argv[2]));

  JitRuntime runtime;
  robin_hood::unordered_node_map<uint32_t, Block> r4300_blocks;
  robin_hood::unordered_map<uint32_t, Block> rsp_blocks;
  Block *block = nullptr, *empty = new Block();
  Block *prev = empty;

  uint32_t rsp_cycles = 0;
  bool already_compiled = false;

  while (true) {
    uint32_t hash = R4300::fetch(R4300::pc);
    bool cached = prev->next && prev->next_pc == R4300::pc;
    if (cached && prev->next->valid(hash)) block = prev->next;
    else {
      prev->next_pc = R4300::pc;
      block = prev->next = &r4300_blocks[R4300::pc & R4300::addr_mask]; 
      if (!block->valid(hash)) {
        block->hash = hash;
        CodeHolder code;
        code.init(runtime.codeInfo());
        MipsJit<Device::r4300> jit(code);

        block->cycles = jit.jit_block();
        runtime.add(&block->code, &code);
      }
    }
    R4300::pc = block->code();

    if (!RSP::halted()) {
      rsp_cycles += block->cycles;
      Block &block2 = rsp_blocks[RSP::pc & RSP::addr_mask];
      uint32_t hash2 = RSP::fetch(RSP::pc);
      if (!already_compiled) {//!block2.valid(hash2)) {
        block2.hash = hash2;
        CodeHolder code;
        code.init(runtime.codeInfo());
        MipsJit<Device::rsp> jit(code);

        block2.cycles = jit.jit_block();
        runtime.add(&block2.code, &code);
        already_compiled = true;
      }
      if (rsp_cycles >= block2.cycles * 4) {
        RSP::pc = block2.code();
        /*if ((RSP::pc & 0xfff) == 0x7fc) {
          for (uint8_t i = 0; i < 32; ++i)
            printf("Reg $%d: %llx\n", i, RSP::reg_array[i]);
        }*/
        R4300::rsp_update();
        rsp_cycles = 0; already_compiled = false;
      }
    }

    R4300::ai_update();
    R4300::pi_update(block->cycles);
    R4300::vi_update(block->cycles);
    R4300::irqs_update(block->cycles);
    if (R4300::broke) {
      Debugger::update();
      r4300_blocks.clear();
      block = empty;
      block->next = nullptr;
    }
    prev = block;
  }
}
