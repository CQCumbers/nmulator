#include "r4300.h"
#include "debugger.h"

int main(int argc, char* argv[]) {
  // initialize system components
  if (argc != 2 && argc != 3) printf("error: must provide file\n"), exit(1);
  FILE *file = fopen(argv[1], "r");
  if (!file) printf("error: can't open file\n"), exit(1);
  R4300::init(file), fclose(file);
  if (argc == 3) Debugger::init(atoi(argv[2]));

  /*JitRuntime runtime;
  Block *block = nullptr, *block2 = nullptr;
  Block *prev = empty, *prev2 = empty;*/

  sched(R4300::update, 0);
  sched(R4300::vi_update, 6510);
  while (true) exec_next();

  /*
    bool cached = prev->next && prev->next_pc == R4300::pc;
    if (cached) block = prev->next;
    else {
      prev->next_pc = R4300::pc;
      block = prev->next = &R4300::blocks[R4300::pc & R4300::addr_mask];
    }
    if (!block->valid) {
      CodeHolder code;
      code.init(runtime.codeInfo());
      MipsJit<Device::r4300> jit(code);

      block->cycles = jit.jit_block();
      runtime.add(&block->code, &code);
      block->valid = true;
    }
    R4300::pc = block->code();
    if (R4300::modified) {
      printf("Unprotect function hit before pc: %x\n", R4300::pc);
      R4300::modified = false;
    }

    for (uint8_t i = 0; i < block->cycles && !RSP::halted();) {
      uint32_t hash = RSP::fetch(RSP::pc);
      if (prev2->valid && prev2->hash == hash) {
        RSP::pc = prev2->code();
        R4300::rsp_update(); RSP::moved = false;
        i += prev2->cycles * 2;
        if (R4300::modified) {
          printf("Unprotect function hit before RSP pc: %x\n", RSP::pc);
          R4300::modified = false;
        }
        hash = RSP::fetch(RSP::pc);
      }

      bool cached = prev2->next && prev2->next_pc == RSP::pc;
      if (cached) block2 = prev2->next;
      else {
        prev2->next_pc = RSP::pc;
        block2 = prev2->next = &RSP::blocks[RSP::pc & RSP::addr_mask];
      }
      if (!block2->valid || block2->hash != hash|| R4300::logging_on) {
        CodeHolder code;
        code.init(runtime.codeInfo());
        MipsJit<Device::rsp> jit(code);

        block2->cycles = jit.jit_block();
        runtime.add(&block2->code, &code);
        block2->valid = true;
        block2->hash = hash;
      }
      prev2 = block2;

      if (RSP::pc == 0x20 || RSP::pc == 0x24)
        printf("break here!\n");

      if (R4300::logging_on) {
        printf("- ACC: ");
        for (uint8_t i = 0; i < 24; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 32 * 2]))[23 - i]);
        printf("\n- VCO: ");
        for (uint8_t i = 0; i < 16; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x86 + 0 * 2]))[15 - i]);
        printf("\n- R29: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 29 * 2]))[7 - i]);
        printf("\n- R27: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 27 * 2]))[7 - i]);
        printf("\n- R17: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 17 * 2]))[7 - i]);
        printf("\n- R5: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 5 * 2]))[7 - i]);
        printf("\n- R3: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 3 * 2]))[7 - i]);
        printf("\n- R2: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 2 * 2]))[7 - i]);
        printf("\n- R1: ");
        for (uint8_t i = 0; i < 8; ++i)
          printf("%hx ", ((uint16_t*)(&RSP::reg_array[0x40 + 1 * 2]))[7 - i]);
        printf("\n- $3: %llx $2: %llx $19: %llx $13: %llx\n- de0: ",
            RSP::reg_array[3], RSP::reg_array[2], RSP::reg_array[19], RSP::reg_array[13]);
        for (uint8_t i = 0; i < 32; ++i)
          printf("%llx ", RSP::read<uint8_t>(0xde0 + i));
        printf("\n- 3e0: ");
        for (uint8_t i = 0; i < 16; ++i)
          printf("%llx ", RSP::read<uint8_t>(0x3e0 + i));
        printf("\n- 5f8: ");
        for (uint8_t i = 0; i < 16; ++i)
          printf("%llx ", RSP::read<uint8_t>(0x5f8 + i));
        printf("\n---\n");
        if ((RSP::reg_array[0x41 + 34 * 2] & 0xffffffff) == 0x5a0000)
          printf("5a0000 present now\n");
      }
    }

    RDP::update(block->cycles);
    R4300::ai_update();
    R4300::pi_update(block->cycles);
    R4300::vi_update(block->cycles);
    R4300::irqs_update(block->cycles);
    if (R4300::broke) {
      Debugger::update();
      R4300::blocks.clear();
      block = empty;
      block->next = nullptr;
    }
    prev = block;
  }*/
}
