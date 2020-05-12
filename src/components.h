#pragma once

#include <stdbool.h>
#include <stdint.h>

/* === Common Utilities === */

#if defined(_MSC_VER)
#  define bswap16(x)  _byteswap_ushort(x)
#  define bswap32(x)  _byteswap_ulong(x)
#  define bswap64(x)  _byteswap_uint64(x)
#else
#  define bswap16(x)  __builtin_bswap16(x)
#  define bswap32(x)  __builtin_bswap32(x)
#  define bswap64(x)  __builtin_bswap64(x)
#endif

inline uint32_t pext_low(uint32_t val, uint32_t mask) {
  val &= 0x55555555;
  val = (val ^ (val >> 1)) & 0x33333333;
  val = (val ^ (val >> 2)) & 0x0f0f0f0f;
  val = (val ^ (val >> 4)) & 0x00ff00ff;
  val = (val ^ (val >> 8)) & 0x0000ffff;
  return val & mask;
}

inline uint32_t crc32(uint8_t *bytes, uint32_t len) {
  uint32_t crc = 0xffffffff;
  for (uint32_t i = 0; i < len; ++i) {
    crc ^= bytes[i];
    for (uint32_t j = 0; j < 8; ++j) {
       uint32_t mask = -(crc & 0x1);
       crc = (crc >> 1) ^ (0xedb88320 & mask);
    }
  }
  return ~crc;
}

/* === JIT interface === */

typedef uint64_t (*ReadPtr)(uint32_t addr);
typedef bool (*BreakPtr)(uint32_t addr);
typedef void (*WritePtr)(uint32_t addr, uint64_t val);
typedef uint32_t (*CodePtr)();

struct JitConfig {
  ReadPtr read[4];
  WritePtr write[4];
  ReadPtr fetch, link;
  WritePtr mtc0;
  BreakPtr is_break;
  uint32_t cop0, cop1, cop2;
  uint64_t *regs;
  uint32_t *tlb;
};

struct Block {
  CodePtr code;
  Block *next;
  uint64_t next_pc;
  uint32_t cycles;
  bool valid;
};

const Block empty;

namespace JitWrapper {
  void r4300_link(JitConfig *out, Block **block);
  void rsp_link(JitConfig *out, Block **block);
  void compile(Block *out, uint32_t pc, const JitConfig *cfg);
}

/* === N64 Components === */

namespace Sched {
  extern int64_t until;
  void add(void (*func)(), uint64_t time);
  void move(void (*func)(), uint64_t time);
  void start_loop();
}

namespace Debugger {
  void update();
  void init(int port);
}

namespace R4300 {
  void update();
  void vi_update();
  void init(void *file);

  extern uint8_t *ram;
  const uint32_t mask = 0x1fffffff;
  void set_irqs(uint32_t mask);
  void unset_irqs(uint32_t mask);
  void set_break(uint32_t addr, bool active);
}

namespace RSP {
  void update();
  void set_status(uint32_t mask);
  extern uint8_t *mem;
  const uint32_t mask = 0xfff;
}

namespace RDP {
  void update();
  void init();
}
