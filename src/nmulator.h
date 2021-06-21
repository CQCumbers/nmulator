#pragma once

#include <stdbool.h>
#include <stdint.h>

//#define printf(fmt, ...) (0)

/* === MIPS-to-x64 JIT compiler === */

typedef int64_t (*ReadPtr)(uint32_t addr);
typedef uint32_t (*FetchPtr)(uint32_t addr);
typedef void (*WritePtr)(uint32_t addr, uint32_t val);
typedef uint64_t (*LinkPtr)(uint32_t pc, uint64_t block);
typedef uint32_t (*CodePtr)();

struct MipsConfig {
  uint64_t *regs;
  uint32_t cop0, cop1;
  uint32_t cop2, pool;

  CodePtr *lookup;
  FetchPtr fetch;
  ReadPtr stop_at;
  LinkPtr watch;
  uint32_t *step;
  WritePtr tlbwi;
  LinkPtr link;

  ReadPtr mfc0;
  uint32_t mfc0_mask;
  WritePtr mtc0;
  uint32_t mtc0_mask;

  uint32_t *pages;
  uint32_t *tlb;
  uint8_t *mem;
  ReadPtr read;
  WritePtr write;
  uint64_t fn[7];
};

struct Block {
  CodePtr code;
  uint32_t len;
  uint32_t hash;
};

namespace Mips {
  uint32_t run(MipsConfig *cfg, CodePtr ptr);
  uint32_t jit(MipsConfig *cfg, uint32_t pc, CodePtr *ptr);
  void init(MipsConfig *cfg);
}

/* === Scheduler and debugger === */

enum Task {
  TASK_RSP,
  TASK_R4300,
  TASK_RDP,
  TASK_VI,
  TASK_AI,
  TASK_PI,
  TASK_CIC,
  TASK_TIMER
};

namespace Sched {
  extern int64_t until;
  uint64_t now();
  void init(uint8_t task, int64_t time);
  void add(uint8_t task, int64_t time);
  void move(uint8_t task, int64_t time); 
  void start_loop();
}

enum Watch {
  WATCH_NONE,
  WATCH_WRITE,
  WATCH_READ,
  WATCH_ACCESS
};

struct DbgConfig {
  uint8_t (*read_mem)(uint32_t addr);
  void (*write_mem)(uint32_t addr, uint8_t val);
  uint64_t (*read_reg)(uint32_t idx);
  void (*write_reg)(uint32_t idx, uint64_t val);
  void (*set_break)(uint32_t addr);
  void (*clr_break)(uint32_t addr);
  void (*set_watch)(uint32_t addr, uint32_t len, char type);
  void (*clr_watch)(uint32_t addr, uint32_t len, char type);
};

namespace Debugger {
  void init(uint32_t port, DbgConfig conf);
  bool poll();
  bool update(uint32_t step);
}

/* === Hardware components === */

namespace R4300 {
  void vi_update();
  void ai_update();
  void pi_update();
  void cic_update();
  void timer_fire();
  void update();

  extern uint8_t *ram;
  extern uint8_t *hram;
  const uint32_t mask = 0x1fffffff;
  void set_irqs(uint32_t mask);
  void unset_irqs(uint32_t mask);
  void init_debug(uint32_t port);
  void init(const char *filename);
}

namespace RSP {
  extern uint32_t pc;
  extern uint64_t *const cop0;
  void update();

  extern uint8_t *mem;
  const uint32_t mask = 0xfff;
  void dma(uint32_t val, bool to_ram);
  void set_status(uint32_t mask);
  void init(uint8_t *mem);
  void test(const char *filename);
}

namespace RDP {
  void update();
  void init();
}

/* === Common utilities === */

inline uint64_t bswap64(uint64_t x) {
  return (
    ((x & 0x00000000000000ff) << 56) |
    ((x & 0x000000000000ff00) << 40) |
    ((x & 0x0000000000ff0000) << 24) |
    ((x & 0x00000000ff000000) <<  8) |
    ((x & 0x000000ff00000000) >>  8) |
    ((x & 0x0000ff0000000000) >> 24) |
    ((x & 0x00ff000000000000) >> 40) |
    ((x & 0xff00000000000000) >> 56)
  );
}

inline uint32_t bswap32(uint32_t x) {
  return (
    ((x & 0x000000ff) << 24) |
    ((x & 0x0000ff00) <<  8) |
    ((x & 0x00ff0000) >>  8) |
    ((x & 0xff000000) >> 24)
  );
}

inline uint16_t bswap16(uint16_t x) {
  return (
    ((x & 0x00ff) <<  8) |
    ((x & 0xff00) >>  8)
  );
}

inline uint32_t pext(uint32_t val, uint32_t mask) {
  val &= 0x55555555;
  val = (val ^ (val >> 1)) & 0x33333333;
  val = (val ^ (val >> 2)) & 0x0f0f0f0f;
  val = (val ^ (val >> 4)) & 0x00ff00ff;
  val = (val ^ (val >> 8)) & 0x0000ffff;
  return val & mask;
}

inline uint32_t read32(uint8_t *bytes) {
  return bswap32(*(uint32_t*)bytes);
}

inline void write32(uint8_t *bytes, uint32_t val) {
  *(uint32_t*)bytes = bswap32(val);
}

inline uint16_t read16(uint8_t *bytes) {
  return bswap16(*(uint16_t*)bytes);
}

inline void write16(uint8_t *bytes, uint16_t val) {
  *(uint16_t*)bytes = bswap16(val);
}
