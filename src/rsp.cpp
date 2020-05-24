#include <stdio.h>
#include "components.h"

namespace RSP {
  uint8_t *mem = nullptr;
  bool step = true, log = false;
  bool moved = false, dirty = false;
  const uint32_t cop0 = 0x20, cop2 = 0x40;
  uint32_t pc = 0x0;
  uint64_t regs[0x100];

  uint64_t hash = 0x12345678000;
  robin_hood::unordered_node_map<uint64_t, Block> blocks;
  Block *block = &empty;

  /* === Reading and Writing === */

  template <typename T, bool all>
  int64_t read(uint32_t addr) {
    addr &= (all ? 0x1fff : 0xfff);
    T *ptr = reinterpret_cast<T*>(mem + addr);
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return static_cast<T>(bswap16(*ptr));
      case 4: return static_cast<T>(bswap32(*ptr));
      case 8: return static_cast<T>(bswap64(*ptr));
    }
  }

  template <typename T, bool all>
  void write(uint32_t addr, T val) {
    addr &= (all ? 0x1fff : 0xfff);
    if (addr & 0x1000) dirty = true;
    T *ptr = reinterpret_cast<T*>(mem + addr);
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = bswap16(val); return;
      case 4: *ptr = bswap32(val); return;
      case 8: *ptr = bswap64(val); return;
    }
  }
  
  uint32_t fetch(uint32_t addr) {
    uint8_t *ptr = mem + 0x1000 + (addr & 0xfff);
    return __builtin_bswap32(*reinterpret_cast<uint32_t*>(ptr));
  }

  void dma(uint32_t val, bool write) {
    uint32_t skip = (val >> 20) & 0xfff, count = (val >> 12) & 0xff;
    uint32_t len = (val & 0xfff) + 8, m0 = 0x1ff8, m1 = 0x7ffff8;
    uint64_t &r0 = regs[0 + cop0], &r1 = regs[1 + cop0];
    if ((r0 & 0xff8) + (len &= ~0x7) > 0x1000) len = 0x1000 - (r0 & 0xff8);
    for (uint32_t i = 0; i <= count; ++i, r1 += len + skip, r0 += len) {
      uint8_t *src = pages[0] + (r1 &= m1), *dst = mem + (r0 &= m0);
      write ? memcpy(src, dst, len) : memcpy(dst, src, len);
    }
    if (r0 > 0x1000 && !write) {
      hash = crc32(mem + 0x1000, 0x1000) << 12;
      hash_dirty = false, block->valid = false;
    }
  }

  /* === Actual CPU Functions === */

  bool is_break(uint32_t pc) {
    if (!moved) { moved = true; return false; }
    return step;
  }

  void mtc0(uint32_t reg, uint32_t val) {
    regs[reg + cop0] = val;
  }

  JitConfig cfg = {
    .read = {
      read<int8_t>, read<int16_t>,
      read<int32_t>, read<int64_t>
    },
    .write = {
      write<int8_t>, write<int16_t>,
      write<int32_t>, write<int64_t>
    },
    .fetch = fetch, .link = 0,
    .is_break = is_break, .mtc0 = mtc0,
    .cop0 = cop0, .cop2 = cop2,
    .regs = regs
  };

  inline bool halted() {
    return regs[4 + cop0] & 0x1;
  }

  inline bool broke() {
    return (regs[4 + cop0] & 0x42) == 0x42;
  }

  void print_state() {
    printf("- ACC: ");
    for (uint8_t i = 0; i < 24; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 32 * 2]))[23 - i]);
    printf("\n- VCO: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x86 + 0 * 4]))[15 - i]);
    printf("\n- R30: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 30 * 2]))[7 - i]);
    printf("\n- R22: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 22 * 2]))[7 - i]);
    printf("\n- R19: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 19 * 2]))[7 - i]);
    printf("\n- R18: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 18 * 2]))[7 - i]);
    printf("\n- R15: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 15 * 2]))[7 - i]);
    printf("\n- R0: ");
    for (uint8_t i = 0; i < 8; ++i)
      printf("%hx ", ((uint16_t*)(&regs[0x40 + 0 * 2]))[7 - i]);
    printf("\n- $25: %llx $23: %llx $14: %llx $29: %llx $2: %llx\n- ce0: ",
        regs[25], regs[23], regs[14], regs[29], regs[2]);
    printf("\n- ce0: ");
    for (uint8_t i = 0; i < 32; ++i)
      printf("%llx ", read<uint8_t>(0xce0 + i));
    printf("\n- d10: ");
    for (uint8_t i = 0; i < 16; ++i)
      printf("%llx ", read<uint8_t>(0xd10 + i));
    /*for (uint32_t i = 0; i < 0x1000; i += 0x10) {
      printf("\n- %x: ", i);
      for (uint8_t j = 0; j < 16; ++j)
        printf("%llx ", read<uint8_t>(i + j));
    }*/
    printf("\n---\n");
  }

  void update() {
    uint32_t cycles = 0;
    if (hash_dirty) {
      hash = crc32(mem + 0x1000, 0x1000) << 12;
      hash_dirty = false, block->valid = false;
    }
    while ((Sched::until -= cycles) >= 0) {
      bool run = block->valid;
      if (run) {
        pc = block->code();
        moved = false;
        if (broke()) R4300::set_irqs(0x1);
        if (log) print_state();

        uint64_t key = hash + pc;
        if (block->next_pc != key)
          block->next_pc = key, block->next = &blocks[key];
        block = block->next;
      }

      if (halted()) { block->valid = false; return; }
      if (!block->valid || log) {
        JitWrapper::compile(block, pc, cfg);
        block->cycles *= 2;
      }
      cycles = run ? block->cycles : 0;
    }
    Sched::add(update, cycles);
  }

  void set_status(uint32_t val) {
    if (halted() && (val & 0x1)) {
      printf("Scheduling RSP at %x\n", R4300::pc);
      block = &blocks[hash + pc];
      if (!block->valid) {
        JitWrapper::compile(block, pc, cfg);
        block->cycles *= 2;
      }
      Sched::add(update, block->cycles);
    }
    regs[4 + cop0] &= ~(val & 0x1);       // HALT
    regs[4 + cop0] |= (val & 0x2) >> 1;
    regs[4 + cop0] &= ~(val & 0x4) >> 1;  // BROKE
    R4300::unset_irqs((val & 0x8) >> 3);  // IRQ
    R4300::set_irqs((val & 0x10) >> 4);
    regs[4 + cop0] &= ~(pext_low(val >> 5, 0x3ff) << 5);
    regs[4 + cop0] |= pext_low(val >> 6, 0x3ff) << 5;
  }
}
