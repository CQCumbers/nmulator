#include <stdio.h>
#include <immintrin.h>
#include "nmulator.h"

#include <vector>
#include "robin_hood.h"

//static const bool logging_on = false;

static uint32_t crc32(uint8_t *bytes, uint32_t len) {
  uint32_t crc = 0, *msg = (uint32_t*)bytes;
  for (uint32_t i = 0; i < len / 4; ++i)
    crc = _mm_crc32_u32(crc, msg[i]);
  return crc;
}

namespace RSP { uint8_t *mem, *imem; }

namespace RSP {
  const uint32_t addr_mask = 0xfff;
  bool step = false, moved = false;
  bool hash_dirty = true;

  uint32_t fetch(uint32_t addr) {
    return read32(imem + (addr & addr_mask));
  }

  uint64_t reg_array[200];
  uint64_t *const cop0 = reg_array + 32;

  const uint8_t dev_cop0 = 0x20, dev_cop2 = 0x40, dev_cop2c = 0x86;
  robin_hood::unordered_map<uint32_t, std::vector<Block>> backups;
  Block blocks[0x1000];
  uint32_t pc = 0x0; Block *block = &empty;
  //uint32_t hashes[32];

  inline bool halted() {
    return reg_array[4 + dev_cop0] & 0x1;
  }

  inline bool broke() {
    return (reg_array[4 + dev_cop0] & 0x42) == 0x42;
  }

  void mtc0(uint32_t idx, uint64_t val) {
    switch (idx &= 0x1f) {
      default: cop0[idx] = val;
      case 0: cop0[0] = val & 0x1fff; return;
      case 1: cop0[1] = val & 0xffffff; return;
      case 2: dma(val, false); return;
      case 3: dma(val, true); return;
      case 4: set_status(val); return;
      case 8: cop0[10] = cop0[8] = val; return;
      case 9: Sched::add(TASK_RDP, 0), cop0[9] = val; return;
      case 11: cop0[11] &= ~pext(val >> 0, 0x7);
        cop0[11] |= pext(val >> 1, 0x7); return;
    }
  }

  void print_state() {
    /*for (uint8_t i = 1; i < 32; ++i)
      printf("Reg $%d: %llx\n", i, reg_array[i]);*/
    printf("PC_START: %llx\n", reg_array[0x28]);
    printf("PC_END: %llx\n", reg_array[0x29]);
    printf("PC_CURRENT: %llx\n", reg_array[0x2a]);
    /*printf("- ACC: "8;
    for (uint8_t i = 0; i < 24; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 32 * 2]))[23 - i]);
    printf("\n- VCO: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x86 + 0 * 4]))[15 - i]);
    printf("\n- R30: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 30 * 2]))[7 - i]);
    printf("\n- R22: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 22 * 2]))[7 - i]);
    printf("\n- R19: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 19 * 2]))[7 - i]);
    printf("\n- R18: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 18 * 2]))[7 - i]);
    printf("\n- R15: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 15 * 2]))[7 - i]);
    printf("\n- R14: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 14 * 2]))[7 - i]);
    printf("\n- R8: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 8 * 2]))[7 - i]);
    printf("\n- R5: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 5 * 2]))[7 - i]);
    printf("\n- R3: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 3 * 2]))[7 - i]);
    printf("\n- R0: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&reg_array[0x40 + 0 * 2]))[7 - i]);
    //printf("\n- $25: %llx $23: %llx $14: %llx $29: %llx $2: %llx\n- ce0: ",
    //    reg_array[25], reg_array[23], reg_array[14], reg_array[29], reg_array[2]);
    printf("\n- ce0: ");
    for (uint8_t i = 0; i < 32; ++i)
      printf("%llx ", read<uint8_t>(0xce0 + i));
    printf("\n- 360: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%llx ", read<uint8_t>(0x360 + i));
    printf("\n- 430: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%llx ", read<uint8_t>(0x430 + i));
    printf("\n- d10: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%llx ", read<uint8_t>(0xd10 + i));
    printf("\n- d50: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%llx ", read<uint8_t>(0xd50 + i));*/
    /*for (uint32_t i = 0x00; i < 0x100; i += 0x10) {
      printf("\n- %x: ", i);
      for (uint8_t j = 0; j < 16; ++j)
        printf("%llx ", read<uint8_t, false>(i + j));
    }
    printf("\n---\n");*/
  }

  void update() {
    while (Sched::until >= 0) {
      if (block->code) {
        pc = block->code() & 0xffc, moved = false;
        if (broke()) R4300::set_irqs(0x1);
        //if (logging_on) print_state();
        block = &blocks[pc];
      }

      if (halted()) { /*block->valid = false;*/ return; }
      uint32_t hash = crc32(imem + pc, block->len * 4);
      if (!block->code || block->hash != hash) {
        bool skip_compile = false;
        for (auto &backup : backups[pc]) {
          hash = crc32(imem + pc, backup.len * 4);
          if (backup.hash == hash) {
            block->code = backup.code;
            block->len = backup.len;
            block->hash = backup.hash;
            skip_compile = true;
          }
        }
        if (skip_compile) continue;

        //printf("Compiling block at %x, %x != %x\n", pc, hash, block->hash);
        block->len = Mips::compile_rsp(&block->code);
        block->hash = crc32(imem + pc, block->len * 4);
        backups[pc].push_back(*block);
        //printf("Adding new block at %x with hash %x\n", pc, block->hash);
      }
    }
    Sched::add(TASK_RSP, 0);
  }

  void unhalt();
}

void RSP::unhalt() {
  block = &blocks[pc &= 0xffc];
  uint32_t hash = crc32(imem + pc, block->len * 4);
  if (!block->code || block->hash != hash) {
    block->len = Mips::compile_rsp(&block->code);
    block->hash = crc32(imem + pc, block->len * 4);
    backups[pc].push_back(*block);
  }
  Sched::add(TASK_RSP, 0);
}

void RSP::set_status(uint32_t val) {
  // update status flags, unhalt RSP
  if (cop0[4] & val & 0x1) unhalt();
  cop0[4] &= ~(val & 0x1), cop0[4] |= (val >> 1) & 0x2;
  cop0[4] &= ~(val & 0x4) >> 1;
  cop0[4] &= ~(pext(val >> 5, 0x3ff) << 5);
  cop0[4] |= pext(val >> 6, 0x3ff) << 5;

  // Update MI RSP interrupt
  R4300::unset_irqs((val & 0x8) >> 3);
  R4300::set_irqs((val & 0x10) >> 4);
}

void RSP::dma(uint32_t val, bool to_ram) {
  // 64-bit align DMA params, limit length to within DMEM/IMEM
  uint32_t skip = (val >> 20) & 0xff8, count = (val >> 12) & 0xff;
  uint32_t len = (val & 0xff8) + 8, max = 0x1000 - (cop0[0] & 0xff8);
  len = (len > max ? max : len), cop0[0] &= 0x1ff8, cop0[1] &= 0x7ffff8;

  // repeatedly DMA bytes from RDRAM to DMEM/IMEM, or vice-versa
  for (uint32_t i = 0; i <= count; ++i) {
    uint8_t *src = to_ram ? mem + cop0[0] : R4300::ram + cop0[1];
    uint8_t *dst = to_ram ? R4300::ram + cop0[1] : mem + cop0[0];
    memcpy(dst, src, len), cop0[0] += len, cop0[1] += len + skip;
  }
}

void RSP::init(uint8_t *mem) {
  // set mem pointer, initial cop0 values
  RSP::mem = mem, imem = mem + 0x1000;
  cop0[4] = 0x1, cop0[11] = 0x80;
  Mips::init_pool(reg_array + 148);
}
