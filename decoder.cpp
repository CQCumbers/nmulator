#include "asmjit/asmjit.h"
#include <SDL2/SDL.h>
#include <unordered_map>
#include <sys/mman.h>

// check address and synchronize on first compile,
// backpatch to remove check if synchronization was unnecessary
//
// don't try to prove branch prediction at compile time
// backpatch the jumped-from block
//
// how to handle delay slots?
//
// inline next block if it is known at compile time
// otherwise terminate block and
// emit x86 instrs for looking up next block
//
// maybe linear search is a bad idea
// also instead of just checking if pc is equal, check preconditions
// (register within range) more preconditions means more compile-time optimizations
// when we want to make an assumption (e.g. memory is not shared), check if that
// assumption is valid with the start-of-block state the JIT was called with, and if
// so add the value of the register that the assumption depends on to the preconditions
// patch this new epilogue into every block?

/* == Lots of Global Variables === */

SDL_Renderer *renderer;
SDL_Window *window;
SDL_Texture *texture;

using namespace asmjit;

typedef uint32_t (*Function)();
std::unordered_map<uint32_t, Function> blocks;
std::unordered_map<uint32_t, uint32_t> block_times;
const uint32_t block_end = 0x04ffffff;

/* === Memory Access === */

const uint32_t addr_mask = 0x1fffffff;
const uint32_t page_mask = 0x1fffff;
uint8_t *pages[0x100] = {nullptr};

uint32_t pi_rom, pi_ram = 0;
uint32_t mi_mask = 0x0, mi_intr = 0;
uint32_t ai_run, ai_used, ai_len = 0;
uint32_t scanline = 0, intrline = 0;
uint8_t *pixels = nullptr;

uint32_t read32_special(uint32_t addr) {
  //printf("! read from address %x\n", addr);
  if ((addr & addr_mask) == 0x04300004) return 0x01010101; // MI Version
  else if ((addr & addr_mask) == 0x04300008) return mi_intr; // MI INTERRUPT
  else if ((addr & addr_mask) == 0x0430000c) return mi_mask; // MI INTERRUPT_MASK
  else if ((addr & addr_mask) == 0x04400010) return scanline; // VI_CURRENT_LINE
  else if ((addr & addr_mask) == 0x0450000c) return 0x11111111; // AI_STATUS
  else printf("Unhandled read from address %x\n", addr);
  return 0;
}

void write32_special(uint32_t addr, uint32_t val) {
  //printf("! write to address %x %x\n", addr, val);
  if ((addr & addr_mask) == 0x04400010) mi_intr &= ~0x8; // VI_CURRENT_LINE
  else if ((addr & addr_mask) == 0x04400004) pixels = (pages[0] + (val & 0xffffff)); // VI_ORIGIN
  else if ((addr & addr_mask) == 0x0440000c) intrline = val & 0x3ff; // VI_INTR_LINE
  else if ((addr & addr_mask) == 0x04500004) ai_len = val & 0x3ffff; // AI LENGTH
  else if ((addr & addr_mask) == 0x04500008) ai_run = val & 0x1; // AI CONTROL
  else if ((addr & addr_mask) == 0x0450000c) mi_intr &= ~0x4; // AI_STATUS
  else if ((addr & addr_mask) == 0x04600000) pi_ram = val & 0xffffff; // PI RAM ADDR
  else if ((addr & addr_mask) == 0x04600004) pi_rom = val & addr_mask; // PI ROM ADDR
  else if ((addr & addr_mask) == 0x04600010) mi_intr &= ~0x10; // PI INTR
  else if ((addr & addr_mask) == 0x04800018) mi_intr &= ~0x2; // SI INTR
  else if ((addr & addr_mask) == 0x0430000c) { // MI INTERRUPT MASK
    mi_mask = ((val & 0x2) >> 1) | ((val & 0x8) >> 2) | ((val & 0x20) >> 3)
      | ((val & 0x80) >> 4) | ((val & 0x200) >> 5) | ((val ^ 0x800) >> 6);
  } else if ((addr & addr_mask) == 0x04600008) { // TRANSFER SIZE (RAM to ROM)
    memcpy(pages[0] + pi_rom, pages[0] + pi_ram, val + 1);
  } else if ((addr & addr_mask) == 0x0460000c) { // TRANSFER SIZE (ROM to RAM)
    printf("Copying %x bytes from %x to %x\n", val + 1, pi_rom, pi_ram);
    memcpy(pages[0] + pi_ram, pages[0] + pi_rom, val + 1); mi_intr |= 0x10;
  } else printf("Unhandled write to address %x: %x\n", addr, val);
}

uint64_t read64(uint32_t addr) {
  uint64_t *page = reinterpret_cast<uint64_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return (read32_special(addr) << 8) | read32_special(addr + 1);
  return __builtin_bswap64(page[(addr & page_mask) >> 3]);
}

void write64(uint32_t addr, uint64_t val) {
  uint64_t *page = reinterpret_cast<uint64_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return write32_special(addr, val >> 8);
  page[(addr & page_mask) >> 3] = __builtin_bswap64(val);
}

uint64_t read32(uint32_t addr) {
  uint32_t *page = reinterpret_cast<uint32_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return read32_special(addr);
  uint64_t val = __builtin_bswap32(page[(addr & page_mask) >> 2]);
  return (val ^ 0x80000000) - 0x80000000;
}

uint64_t uread32(uint32_t addr) {
  uint32_t *page = reinterpret_cast<uint32_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return read32_special(addr);
  return __builtin_bswap32(page[(addr & page_mask) >> 2]);
}

void write32(uint32_t addr, uint32_t val) {
  uint32_t *page = reinterpret_cast<uint32_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return write32_special(addr, val);
  page[(addr & page_mask) >> 2] = __builtin_bswap32(val);
}

uint64_t read16(uint32_t addr) {
  uint16_t *page = reinterpret_cast<uint16_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return read32_special(addr);
  return (__builtin_bswap16(page[(addr & page_mask) >> 1]) ^ 0x8000) - 0x8000;
}

uint64_t uread16(uint32_t addr) {
  uint16_t *page = reinterpret_cast<uint16_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return read32_special(addr);
  return __builtin_bswap16(page[(addr & page_mask) >> 1]);
}

void write16(uint32_t addr, uint16_t val) {
  uint16_t *page = reinterpret_cast<uint16_t*>(pages[(addr >> 21) & 0xff]);
  if (!page) return write32_special(addr, val);
  page[(addr & page_mask) >> 1] = __builtin_bswap16(val);
}

uint64_t read8(uint32_t addr) {
  uint8_t *page = pages[(addr >> 21) & 0xff];
  if (!page) return ((read32_special(addr) & 0xff) ^ 0x80) - 0x80;
  return (page[addr & page_mask] ^ 0x80) - 0x80;
}

uint64_t uread8(uint32_t addr) {
  uint8_t *page = pages[(addr >> 21) & 0xff];
  if (!page) return read32_special(addr) & 0xff;
  return page[addr & page_mask];
}

void write8(uint32_t addr, uint8_t val) {
  uint8_t *page = pages[(addr >> 21) & 0xff];
  if (!page) return write32_special(addr, val);
  page[addr & page_mask] = val;
}

/* === Instruction Decoding === */

inline uint32_t target(uint32_t instr) {
  return (instr & 0x3ffffff) << 2;
}

inline uint8_t rs(uint32_t instr) {
  return (instr >> 21) & 0x1f;
}

inline uint8_t rt(uint32_t instr) {
  return (instr >> 16) & 0x1f;
}

inline uint8_t rd(uint32_t instr) {
  return (instr >> 11) & 0x1f;
}

inline unsigned sa(uint32_t instr) {
  return (instr >> 6) & 0x1f;
}

inline int32_t imm(uint32_t instr) {
  int32_t val = instr & 0xffff;
  return (val ^ 0x8000) - 0x8000;
}

inline uint32_t uimm(uint32_t instr) {
  return instr & 0xffff;
}

/* === Register Allocation === */

uint64_t reg_array[0x42];
const uint8_t hi = 0x20, lo = 0x21;
//uint8_t mapping[0x20] = {0};
const uint8_t dev_cop0 = 0x22;

uint8_t x86_reg(uint8_t reg) {
  //mapping[5] = 12, mapping[6] = 13;
  //return mapping[reg];
  return 0;
}

const x86::Mem x86_spill(uint8_t reg) {
  return x86::dword_ptr(x86::rbp, reg << 3);
}

const x86::Mem x86_spilld(uint8_t reg) {
  return x86::qword_ptr(x86::rbp, reg << 3);
}

/* === Instruction Translation === */

struct MipsJit {
  x86::Assembler as;
  Label end_label;

  MipsJit(CodeHolder &code) : as(&code) {}

  void move(uint8_t dst, uint8_t src) {
    if (dst == src) return;
    uint8_t dstx = x86_reg(dst), srcx = x86_reg(src);
    if (dstx) {
      if (srcx) as.mov(x86::gpd(dstx), x86::gpd(srcx));
      else as.mov(x86::gpd(dstx), x86_spill(src));
    } else {
      if (srcx) as.mov(x86_spill(dst), x86::gpd(srcx));
      else { 
        as.mov(x86::eax, x86_spill(src));
        as.mov(x86_spill(dst), x86::eax);
      }
    }
  }
  
  void compare(uint8_t reg1, uint8_t reg2) {
    uint8_t reg1x = x86_reg(reg1), reg2x = x86_reg(reg2);
    if (reg2 == 0) {
      if (reg1x) as.cmp(x86::gpd(reg1x), 0);
      else as.cmp(x86_spill(reg1), 0);
    } else {
      if (reg1x) {
        if (reg2x) as.cmp(x86::gpd(reg1x), x86::gpd(reg2x));
        else as.cmp(x86::gpd(reg1x), x86_spill(reg2));
      } else {
        if (reg2x) as.cmp(x86_spill(reg1), x86::gpd(reg2x));
        else {
          as.mov(x86::eax, x86_spill(reg2));
          as.cmp(x86_spill(reg1), x86::eax);
        }
      }
    }
  }

  void ld(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LD BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(read64));
    if (rtx) as.mov(x86::gpq(rtx), x86::rax);
    else as.mov(x86_spilld(rt(instr)), x86::rax);
    as.pop(x86::edi);
  }

  void lwu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LWU BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(uread32));
    if (rtx) as.mov(x86::gpd(rtx), x86::eax);
    else as.mov(x86_spill(rt(instr)), x86::eax);
    as.pop(x86::edi);
  }

  void lw(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(read32));
    if (rtx) as.mov(x86::gpq(rtx), x86::rax);
    else as.mov(x86_spilld(rt(instr)), x86::rax);
    as.pop(x86::edi);
  }

  void lhu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LHU BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(uread16));
    if (rtx) as.mov(x86::gpd(rtx), x86::eax);
    else as.mov(x86_spill(rt(instr)), x86::eax);
    as.pop(x86::edi);
  }

  void lh(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LH BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(read16));
    if (rtx) as.mov(x86::gpd(rtx), x86::eax);
    else as.mov(x86_spill(rt(instr)), x86::eax);
    as.pop(x86::edi);
  }

  void lbu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LBU BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(uread8));
    if (rtx) as.mov(x86::gpd(rtx), x86::eax);
    else as.mov(x86_spill(rt(instr)), x86::eax);
    as.pop(x86::edi);
  }

  void lb(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LB BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(read8));
    if (rtx) as.mov(x86::gpd(rtx), x86::eax);
    else as.mov(x86_spill(rt(instr)), x86::eax);
    as.pop(x86::edi);
  }

  void sd(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SD BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::rdi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::rdi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(write64));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  void sw(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.mov(x86::esi, x86::gpd(rtx));
    else as.mov(x86::esi, x86_spill(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(write32));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  void sh(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SH BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.mov(x86::esi, x86::gpd(rtx));
    else as.mov(x86::esi, x86_spill(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(write16));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  void sb(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SB BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.mov(x86::esi, x86::gpd(rtx));
    else as.mov(x86::esi, x86_spill(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(write8));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  void lui(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    // LUI RT, IMMEDIATE
    if (rtx) as.mov(x86::gpd(rtx), uimm(instr) << 16);
    else as.mov(x86_spill(rt(instr)), uimm(instr) << 16);
  }

  void addiu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // ADDIU RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), imm(instr));
      else as.mov(x86_spill(rt(instr)), imm(instr));
    } else {
      // ADDIU RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.add(x86::gpd(rtx), imm(instr));
      else as.add(x86_spill(rt(instr)), imm(instr));
    }
  }

  void addu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // ADD RD, $0, $0 
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // ADD RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // ADD RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      uint8_t rtx = x86_reg(rt(instr));
      // ADD RD, RD, RT
      if (rdx) {
        if (rtx) as.add(x86::gpd(rdx), x86::gpd(rtx));
        else as.add(x86::gpd(rdx), x86_spill(rt(instr)));
      } else {
        if (rtx) as.add(x86_spill(rd(instr)), x86::gpd(rtx));
        else {
          as.mov(x86::eax, x86_spill(rt(instr)));
          as.add(x86_spill(rd(instr)), x86::eax);
        }
      }
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // ADD RD, RS, RT
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.add(x86::gpd(rdx), x86::gpd(rsx));
        else as.add(x86::gpd(rdx), x86_spill(rs(instr)));
      } else {
        if (rsx) as.add(x86_spill(rd(instr)), x86::gpd(rsx));
        else {
          as.mov(x86::eax, x86_spill(rs(instr)));
          as.add(x86_spill(rd(instr)), x86::eax);
        }
      }
    }
  }

  void subu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // SUB RD, $0, $0 
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SUB RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // SUB RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      uint8_t rtx = x86_reg(rt(instr));
      // SUB RD, RD, RT
      if (rdx) {
        if (rtx) as.sub(x86::gpd(rdx), x86::gpd(rtx));
        else as.sub(x86::gpd(rdx), x86_spill(rt(instr)));
      } else {
        if (rtx) as.sub(x86_spill(rd(instr)), x86::gpd(rtx));
        else {
          as.mov(x86::eax, x86_spill(rt(instr)));
          as.sub(x86_spill(rd(instr)), x86::eax);
        }
      }
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // SUB RD, RS, RT
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.sub(x86::gpd(rdx), x86::gpd(rsx));
        else as.sub(x86::gpd(rdx), x86_spill(rs(instr)));
        as.neg(x86::gpd(rdx));
      } else {
        if (rsx) as.sub(x86_spill(rd(instr)), x86::gpd(rsx));
        else {
          as.mov(x86::eax, x86_spill(rs(instr)));
          as.sub(x86_spill(rd(instr)), x86::eax);
        }
        as.neg(x86_spill(rd(instr)));
      }
    }
  }

  void ori(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // ORI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), uimm(instr));
      else as.mov(x86_spill(rt(instr)), uimm(instr));
    } else {
      // ORI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.or_(x86::gpd(rtx), uimm(instr));
      else as.or_(x86_spill(rt(instr)), uimm(instr));
    }
  }

  void or_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // OR RD, $0, $0 
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0 || rt(instr) == rs(instr)) {
      // OR RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // OR RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      uint8_t rtx = x86_reg(rt(instr));
      // OR RD, RD, RT
      if (rdx) {
        if (rtx) as.or_(x86::gpd(rdx), x86::gpd(rtx));
        else as.or_(x86::gpd(rdx), x86_spill(rt(instr)));
      } else {
        if (rtx) as.or_(x86_spill(rd(instr)), x86::gpd(rtx));
        else {
          as.mov(x86::eax, x86_spill(rt(instr)));
          as.or_(x86_spill(rd(instr)), x86::eax);
        }
      }
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // OR RD, RS, RT
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.or_(x86::gpd(rdx), x86::gpd(rsx));
        else as.or_(x86::gpd(rdx), x86_spill(rs(instr)));
      } else {
        if (rsx) as.or_(x86_spill(rd(instr)), x86::gpd(rsx));
        else {
          as.mov(x86::eax, x86_spill(rs(instr)));
          as.or_(x86_spill(rd(instr)), x86::eax);
        }
      }
    }
  }

  void andi(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0 || uimm(instr) == 0) {
      // ANDI RT, $0, IMMEDIATE
      if (rtx) as.xor_(x86::gpd(rtx), x86::gpd(rtx));
      else as.mov(x86_spill(rt(instr)), 0);
    } else {
      // ANDI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.and_(x86::gpd(rtx), uimm(instr));
      else as.and_(x86_spill(rt(instr)), uimm(instr));
    }
  }

  void and_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 || rt(instr) == 0) {
      // AND RD, $0, RT
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rt(instr) == rs(instr)) {
      // AND RD, RT, RT
      move(rd(instr), rt(instr));
    } else if (rd(instr) == rs(instr)) {
      uint8_t rtx = x86_reg(rt(instr));
      // AND RD, RD, RT
      if (rdx) {
        if (rtx) as.and_(x86::gpd(rdx), x86::gpd(rtx));
        else as.and_(x86::gpd(rdx), x86_spill(rt(instr)));
      } else {
        if (rtx) as.and_(x86_spill(rd(instr)), x86::gpd(rtx));
        else {
          as.mov(x86::eax, x86_spill(rt(instr)));
          as.and_(x86_spill(rd(instr)), x86::eax);
        }
      }
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // AND RD, RS, RT
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.and_(x86::gpd(rdx), x86::gpd(rsx));
        else as.and_(x86::gpd(rdx), x86_spill(rs(instr)));
      } else {
        if (rsx) as.and_(x86_spill(rd(instr)), x86::gpd(rsx));
        else {
          as.mov(x86::eax, x86_spill(rs(instr)));
          as.and_(x86_spill(rd(instr)), x86::eax);
        }
      }
    }
  }

  void xori(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // XORI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), uimm(instr));
      else as.mov(x86_spill(rt(instr)), uimm(instr));
    } else {
      // XORI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.xor_(x86::gpd(rtx), uimm(instr));
      else as.xor_(x86_spill(rt(instr)), uimm(instr));
    }
  }

  void xor_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == rt(instr)) {
      // XOR RD, RT, RT
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // XOR RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // XOR RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      uint8_t rtx = x86_reg(rt(instr));
      // XOR RD, RD, RT
      if (rdx) {
        if (rtx) as.xor_(x86::gpd(rdx), x86::gpd(rtx));
        else as.xor_(x86::gpd(rdx), x86_spill(rt(instr)));
      } else {
        if (rtx) as.xor_(x86_spill(rd(instr)), x86::gpd(rtx));
        else {
          as.mov(x86::eax, x86_spill(rt(instr)));
          as.xor_(x86_spill(rd(instr)), x86::eax);
        }
      }
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // XOR RD, RS, RT
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.xor_(x86::gpd(rdx), x86::gpd(rsx));
        else as.xor_(x86::gpd(rdx), x86_spill(rs(instr)));
      } else {
        if (rsx) as.xor_(x86_spill(rd(instr)), x86::gpd(rsx));
        else {
          as.mov(x86::eax, x86_spill(rs(instr)));
          as.xor_(x86_spill(rd(instr)), x86::eax);
        }
      }
    }
  }

  void nor(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    or_(instr);
    if (rdx) as.not_(x86::gpd(rdx));
    else as.not_(x86_spill(rd(instr)));
  }

  void dsll32(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // DSLL32 RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else {
      // DSLL32 RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shl(x86::gpq(rdx), sa(instr) + 32);
      else as.shl(x86_spilld(rd(instr)), sa(instr) + 32);
    }
  }

  void sll(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SLL RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (sa(instr) == 0) {
      // SLL RD, RT, 0
      move(rd(instr), rt(instr));
    } else {
      // SLL RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shl(x86::gpd(rdx), sa(instr));
      else as.shl(x86_spill(rd(instr)), sa(instr));
    }
  }

  void sllv(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SLLV RD, $0, RS
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SLLV RD, RT, $0
      move(rd(instr), rt(instr));
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // SLLV RD, RT, IMMEDIATE
      if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
      else as.mov(x86::ecx, x86_spill(rs(instr)));
      move(rd(instr), rt(instr));
      if (rdx) as.shl(x86::gpd(rdx), x86::cl);
      else as.shl(x86_spill(rd(instr)), x86::cl);
    }
  }

  void dsrl32(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // DSRL32 RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else {
      // DSRL32 RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shr(x86::gpq(rdx), sa(instr) + 32);
      else as.shr(x86_spilld(rd(instr)), sa(instr) + 32);
    }
  }

  void srl(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SRL RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (sa(instr) == 0) {
      // SRL RD, RT, 0
      move(rd(instr), rt(instr));
    } else {
      // SRL RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shr(x86::gpd(rdx), sa(instr));
      else as.shr(x86_spill(rd(instr)), sa(instr));
    }
  }

  void srlv(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SRLV RD, $0, RS
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SRLV RD, RT, $0
      move(rd(instr), rt(instr));
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // SRLV RD, RT, IMMEDIATE
      if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
      else as.mov(x86::ecx, x86_spill(rs(instr)));
      move(rd(instr), rt(instr));
      if (rdx) as.shr(x86::gpd(rdx), x86::cl);
      else as.shr(x86_spill(rd(instr)), x86::cl);
    }
  }

  void dsra32(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // DSRA RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (sa(instr) == 0) {
      // DSRA RD, RT, 0
      move(rd(instr), rt(instr));
    } else {
      // DSRA RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.sar(x86::gpd(rdx), sa(instr) + 32);
      else as.sar(x86_spill(rd(instr)), sa(instr) + 32);
    }
  }

  void sra(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SRA RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (sa(instr) == 0) {
      // SRA RD, RT, 0
      move(rd(instr), rt(instr));
    } else {
      // SRA RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.sar(x86::gpd(rdx), sa(instr));
      else as.sar(x86_spill(rd(instr)), sa(instr));
    }
  }

  void srav(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SRAV RD, $0, RS
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SRAV RD, RT, $0
      move(rd(instr), rt(instr));
    } else {
      uint8_t rsx = x86_reg(rs(instr));
      // SRA RD, RT, IMMEDIATE
      if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
      else as.mov(x86::ecx, x86_spill(rs(instr)));
      move(rd(instr), rt(instr));
      if (rdx) as.sar(x86::gpd(rdx), x86::cl);
      else as.sar(x86_spill(rd(instr)), x86::cl);
    }
  }

  void slti(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint32_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // SLTI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), 0 < imm(instr));
      else as.mov(x86_spill(rt(instr)), 0 < imm(instr));
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      // SLTI RT, RS, IMMEDIATE
      if (rsx) as.cmp(x86::gpd(rsx), imm(instr));
      else as.cmp(x86_spill(rs(instr)), imm(instr));
      as.setl(x86::al);
      if (rtx) as.movzx(x86::gpd(rtx), x86::al);
      else {
        as.movzx(x86::eax, x86::al);
        as.mov(x86_spill(rt(instr)), x86::eax);
      }
    }
  }

  void sltiu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint32_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // SLTIU RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), 0 != imm(instr));
      else as.mov(x86_spill(rt(instr)), 0 != imm(instr));
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      // SLTIU RT, RS, IMMEDIATE
      if (rsx) as.cmp(x86::gpd(rsx), imm(instr));
      else as.cmp(x86_spill(rs(instr)), imm(instr));
      as.setb(x86::al);
      if (rtx) as.movzx(x86::gpd(rtx), x86::al);
      else {
        as.movzx(x86::eax, x86::al);
        as.mov(x86_spill(rt(instr)), x86::eax);
      }
    }
  }

  void slt(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rt(instr) == rs(instr)) {
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else {
      compare(rs(instr), rt(instr));
      as.setl(x86::al);
      if (rdx) as.movzx(x86::gpd(rdx), x86::al);
      else {
        as.movzx(x86::eax, x86::al);
        as.mov(x86_spill(rd(instr)), x86::eax);
      }
    }
  }

  void sltu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rt(instr) == rs(instr)) {
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else {
      compare(rs(instr), rt(instr));
      as.setb(x86::al);
      if (rdx) as.movzx(x86::gpd(rdx), x86::al);
      else {
        as.movzx(x86::eax, x86::al);
        as.mov(x86_spill(rd(instr)), x86::eax);
      }
    }
  }

  void mult(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spill(lo), 0);
      as.mov(x86_spill(hi), 0);
    } else {
      uint32_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::eax, x86::gpd(rsx));
      else as.mov(x86::eax, x86_spill(rs(instr)));
      if (rtx) as.imul(x86::gpd(rtx));
      else as.imul(x86_spill(rt(instr)));
      as.mov(x86_spill(lo), x86::eax);
      as.mov(x86_spill(hi), x86::edx);
    }
  }

  void multu(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spill(lo), 0);
      as.mov(x86_spill(hi), 0);
    } else {
      uint32_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::eax, x86::gpd(rsx));
      else as.mov(x86::eax, x86_spill(rs(instr)));
      if (rtx) as.mul(x86::gpd(rtx));
      else as.mul(x86_spill(rt(instr)));
      as.mov(x86_spill(lo), x86::eax);
      as.mov(x86_spill(hi), x86::edx);
    }
  }

  void div(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spill(lo), 0);
      as.mov(x86_spill(hi), 0);
    } else {
      uint32_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::eax, x86::gpd(rsx));
      else as.mov(x86::eax, x86_spill(rs(instr)));
      as.cdq();
      Label after_div = as.newLabel();
      if (rtx) {
        as.cmp(x86::gpd(rtx), 0);
        as.je(after_div);
        as.idiv(x86::gpd(rtx));
      } else {
        as.cmp(x86_spill(rt(instr)), 0);
        as.je(after_div);
        as.idiv(x86_spill(rt(instr)));
      }
      as.bind(after_div);
      as.mov(x86_spill(lo), x86::eax);
      as.mov(x86_spill(hi), x86::edx);
    }
  }

  void divu(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spill(lo), 0);
      as.mov(x86_spill(hi), 0);
    } else {
      uint32_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::eax, x86::gpd(rsx));
      else as.mov(x86::eax, x86_spill(rs(instr)));
      as.xor_(x86::edx, x86::edx);
      Label after_div = as.newLabel();
      if (rtx) {
        as.cmp(x86::gpd(rtx), 0);
        as.je(after_div);
        as.div(x86::gpd(rtx));
      } else {
        as.cmp(x86_spill(rt(instr)), 0);
        as.je(after_div);
        as.div(x86_spill(rt(instr)));
      }
      as.bind(after_div);
      as.mov(x86_spill(lo), x86::eax);
      as.mov(x86_spill(hi), x86::edx);
    }
  }

  void mfhi(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rdx) as.mov(x86::gpd(rdx), x86_spill(hi));
    else {
      as.mov(x86::eax, x86_spill(hi));
      as.mov(x86_spill(rd(instr)), x86::eax);
    }
  }

  void mthi(uint32_t instr) {
    uint32_t rdx = x86_reg(rd(instr));
    if (rd(instr) == 0) as.mov(x86_spill(hi), 0);
    else if (rdx) as.mov(x86_spill(hi), x86::gpd(rdx));
    else {
      as.mov(x86::eax, x86_spill(rd(instr)));
      as.mov(x86_spill(hi), x86::eax);
    }
  }

  void mflo(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rdx) as.mov(x86::gpd(rdx), x86_spill(lo));
    else {
      as.mov(x86::eax, x86_spill(lo));
      as.mov(x86_spill(rd(instr)), x86::eax);
    }
  }

  void mtlo(uint32_t instr) {
    uint32_t rdx = x86_reg(rd(instr));
    if (rd(instr) == 0) as.mov(x86_spill(lo), 0);
    else if (rdx) as.mov(x86_spill(lo), x86::gpd(rdx));
    else {
      as.mov(x86::eax, x86_spill(rd(instr)));
      as.mov(x86_spill(lo), x86::eax);
    }
  }

  uint32_t j(uint32_t instr, uint32_t pc) {
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86::edi, dst);
    return block_end;
  }

  uint32_t jal(uint32_t instr, uint32_t pc) {
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86_spill(31), pc + 4);
    as.mov(x86::edi, dst);
    return block_end;
  }

  uint32_t jr(uint32_t instr) {
    uint32_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::edi, x86::gpd(rsx));
    else as.mov(x86::edi, x86_spill(rs(instr)));
    return block_end;
  }

  uint32_t beq(uint32_t instr, uint32_t pc) {
    if (rt(instr) == rs(instr)) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    compare(rt(instr), rs(instr));
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmove(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bne(uint32_t instr, uint32_t pc) {
    if (rt(instr) == rs(instr)) return pc;
    compare(rt(instr), rs(instr));
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovne(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bltz(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovl(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bgtz(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovg(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t blez(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovle(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bgez(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovge(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t beql(uint32_t instr, uint32_t pc) {
    if (rt(instr) == rs(instr)) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    compare(rt(instr), rs(instr));
    as.mov(x86::edi, pc + 4);
    as.jne(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bnel(uint32_t instr, uint32_t pc) {
    if (rt(instr) == rs(instr)) return pc + 4;
    compare(rt(instr), rs(instr));
    as.mov(x86::edi, pc + 4);
    as.je(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bltzl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc + 4;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jge(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bgtzl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc + 4;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jle(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t blezl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jg(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bgezl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc + 4;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jl(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bltzal(uint32_t instr, uint32_t pc) {
    as.mov(x86_spill(31), pc + 4);
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovl(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bgezal(uint32_t instr, uint32_t pc) {
    as.mov(x86_spill(31), pc + 4);
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovge(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bltzall(uint32_t instr, uint32_t pc) {
    as.mov(x86_spill(31), pc + 4);
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jge(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bgezall(uint32_t instr, uint32_t pc) {
    as.mov(x86_spill(31), pc + 4);
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.jl(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  void mfc0(uint32_t instr) {
    printf("Read from COP0 reg %d\n", rd(instr));
    if (rt(instr) == 0) return;
    else move(rt(instr), rd(instr) + dev_cop0);
  }

  void mtc0(uint32_t instr) {
    printf("Write to COP0 reg %d\n", rd(instr));
    if (rd(instr) == 11) as.and_(x86_spill(13 + dev_cop0), ~(0x1 << 15));
    if (rt(instr) == 0) as.mov(x86_spill(rd(instr) + dev_cop0), 0);
    else move(rd(instr) + dev_cop0, rt(instr));
  }

  uint32_t eret(uint32_t instr) {
    printf("Returning from interrupt\n");
    as.and_(x86_spill(12 + dev_cop0), ~0x2);
    as.mov(x86::edi, x86_spill(14 + dev_cop0));
    as.jmp(end_label);
    return block_end;
  }

  void cache(uint32_t instr) {
    printf("CACHE instruction %x\n", instr);
  }

  void invalid(uint32_t instr) {
    printf("Unimplemented instruction %x\n", instr);
    exit(0);
  }

  /* === Basic Block Translation ==*/

  uint32_t special(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch (instr & 0x3f) {
      case 0x00: sll(instr); break;
      case 0x02: srl(instr); break;
      case 0x03: sra(instr); break;
      case 0x04: sllv(instr); break;
      case 0x06: srlv(instr); break;
      case 0x07: srav(instr); break;
      case 0x08: next_pc = jr(instr); break;
      case 0x0f: printf("SYNC\n"); break;
      case 0x10: mfhi(instr);  break;
      case 0x11: mthi(instr);  break;
      case 0x12: mflo(instr);  break;
      case 0x13: mtlo(instr);  break;
      case 0x18: mult(instr); break;
      case 0x19: multu(instr); break;
      case 0x1a: div(instr); break;
      case 0x1b: divu(instr); break;
      case 0x20: addu(instr); break; // ADD
      case 0x21: addu(instr); break;
      case 0x22: subu(instr); break; // SUB
      case 0x23: subu(instr); break;
      case 0x24: and_(instr); break;
      case 0x25: or_(instr); break;
      case 0x26: xor_(instr); break;
      case 0x27: nor(instr); break;
      case 0x2a: slt(instr); break;
      case 0x2b: sltu(instr); break;
      case 0x38: sll(instr); break; // DSLL
      case 0x3a: srl(instr); break; // DSRL
      case 0x3b: sra(instr); break; // DSRA
      case 0x3c: dsll32(instr); break;
      case 0x3e: dsrl32(instr); break;
      case 0x3f: dsra32(instr); break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t regimm(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 16) & 0x1f) {
      case 0x00: next_pc = bltz(instr, pc); break;
      case 0x01: next_pc = bgez(instr, pc); break;
      case 0x02: next_pc = bltzl(instr, pc); break;
      case 0x03: next_pc = bgezl(instr, pc); break;
      case 0x10: next_pc = bltzal(instr, pc); break;
      case 0x11: next_pc = bgezal(instr, pc); break;
      case 0x12: next_pc = bltzall(instr, pc); break;
      case 0x13: next_pc = bgezall(instr, pc); break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t cop0(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 24) & 0x3) {
      case 0x0: // COP0/0
        switch (rs(instr)) {
          case 0x0: mfc0(instr); break;
          case 0x4: mtc0(instr); break;
          default: printf("COP0 instruction %x\n", instr); break;
        }
        break;
      case 0x2: // COP0/2
        switch (instr & 0x3f) {
          case 0x18: next_pc = eret(instr); break;
          default: printf("COP0 instruction %x\n", instr); break;
        }
        break;
      default: printf("COP0 instruction %x\n", instr); break;
    }
    return next_pc;
  }

  void cop1(uint32_t instr) {
    printf("COP1 instruction %x\n", instr);
  }

  uint32_t jit_block(uint32_t pc) {
    as.push(x86::rbp);
    as.mov(x86::rbp, reinterpret_cast<uint64_t>(&reg_array));
    //as.push(x86::gpd(x86_reg(5)));
    //as.mov(x86::gpd(x86_reg(5)), x86_spill(5));
    //as.push(x86::gpd(x86_reg(6)));
    //as.mov(x86::gpd(x86_reg(6)), x86_spill(6));

    end_label = as.newLabel();
    uint32_t cycles = 0;
    for (uint32_t next_pc = pc + 4; pc != block_end; ++cycles) {
      //printf("%x\n", pc);
      uint32_t instr = read32(pc);
      pc = next_pc, next_pc += 4;
      switch (instr >> 26) {
        case 0x00: next_pc = special(instr, pc); break;
        case 0x01: next_pc = regimm(instr, pc); break;
        case 0x02: next_pc = j(instr, pc); break;
        case 0x03: next_pc = jal(instr, pc); break;
        case 0x04: next_pc = beq(instr, pc); break;
        case 0x05: next_pc = bne(instr, pc); break;
        case 0x06: next_pc = blez(instr, pc); break;
        case 0x07: next_pc = bgtz(instr, pc); break;
        case 0x08: addiu(instr); break; // ADDI
        case 0x09: addiu(instr); break;
        case 0x0a: slti(instr); break;
        case 0x0b: sltiu(instr); break;
        case 0x0c: andi(instr); break;
        case 0x0d: ori(instr); break;
        case 0x0e: xori(instr); break;
        case 0x0f: lui(instr); break;
        case 0x10: next_pc = cop0(instr, pc); break;
        case 0x11: cop1(instr); break;
        case 0x14: next_pc = beql(instr, pc); break;
        case 0x15: next_pc = bnel(instr, pc); break;
        case 0x16: next_pc = blezl(instr, pc); break;
        case 0x17: next_pc = bgtzl(instr, pc); break;
        case 0x20: lb(instr); break;
        case 0x21: lh(instr); break;
        case 0x23: lw(instr); break;
        case 0x24: lbu(instr); break;
        case 0x25: lhu(instr); break;
        case 0x27: lwu(instr); break;
        case 0x28: sb(instr); break;
        case 0x29: sh(instr); break;
        case 0x2b: sw(instr); break;
        case 0x2f: cache(instr); break;
        case 0x31: cop1(instr); break; // LWC1
        case 0x35: cop1(instr); break; // LDC1
        case 0x37: ld(instr); break;
        case 0x39: cop1(instr); break; // SWC1
        case 0x3d: cop1(instr); break; // SDC1
        case 0x3f: sd(instr); break;
        default: invalid(instr); break;
      }
    }

    as.bind(end_label);
    //as.mov(x86_spill(6), x86::gpd(x86_reg(6)));
    //as.pop(x86::gpd(x86_reg(6)));
    //as.mov(x86_spill(5), x86::gpd(x86_reg(5)));
    //as.pop(x86::gpd(x86_reg(5)));
    as.pop(x86::rbp);

    as.mov(x86::eax, x86::edi);
    as.ret();
    return cycles;
  }
};

int main(int argc, char* argv[]) {
  // setup SDL video
  SDL_Init(SDL_INIT_VIDEO);
  SDL_CreateWindowAndRenderer(640, 480, SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);
  SDL_SetRenderDrawColor(renderer, 0x9b, 0xbc, 0x0f, 0xff);
  SDL_RenderClear(renderer);
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888,
                              SDL_TEXTUREACCESS_STREAMING, 640, 480);

  // setup memory pages
  pages[0] = reinterpret_cast<uint8_t*>(mmap(nullptr, 0x20000000,
    PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, 0, 0));
  for (uint32_t i = 0x1; i < 0x100; ++i) {
    // specially handle SP/DP, DP/MI, VI/AI, PI/RI, and SI ranges
    if (i < 0x21 || i > 0x24) pages[i] = pages[0] + i * (page_mask + 1);
  }
  pages[0xfe] = nullptr; // PIF range

  // read ROM file into memory
  FILE *file = fopen(argv[1], "r");
  if (!file) printf("error: can't open file %s", argv[1]), exit(1);
  fseek(file, 0, SEEK_END);
  long fsize = ftell(file);
  fseek(file, 0, SEEK_SET);
  fread(pages[0] + 0x04000000, 1, fsize, file);
  fclose(file);

  // set registers to initial values
  uint32_t pc = 0xa4000040;
  reg_array[20] = 0x1;
  reg_array[22] = 0x3f;
  reg_array[29] = 0xa4001ff0;

  // VI_ORIGIN at 0x100000
  pixels = pages[0];// + 0x100000;
  // Cartridge Header at 0x10000000
  memcpy(pages[0] + 0x10000000, pages[0] + 0x04000000, fsize);

  JitRuntime runtime;
  uint32_t line_cycle = 0;
  while (true) {
    //Function &run_block = blocks[pc];
    //uint32_t &block_time = block_times[pc];
    Function run_block = nullptr;
    uint32_t block_time = 0;
    if (!run_block) {
      CodeHolder code;
      //FileLogger logger(stdout);

      code.init(runtime.codeInfo());
      //code.setLogger(&logger);
      MipsJit cpu(code);

      code.init(runtime.codeInfo());
      block_time = cpu.jit_block(pc);
      runtime.add(&run_block, &code);
    }
    pc = run_block();
    line_cycle += block_time;

    //printf("pixels: %lx, scanline: %d\n", pixels - pages[0], scanline);
    //printf("$sp: %llx $s1: %llx\n", reg_array[29], reg_array[17]);
    //printf("$a3: %llx $t0: %llx\n", reg_array[7], reg_array[8]);
    //printf("next block at %x\n", pc);
    //printf("STATUS: %llx EPC: %llx\n", reg_array[12 + dev_cop0] & 0x3, reg_array[14 + dev_cop0]);

    // set MI_INTR when all audio used
    if (ai_run && ++ai_used == 1000000) mi_intr |= 0x4, ai_used = 0;
    // clear MI_INTR a few cycles after that
    else if (ai_run && ai_used == 0x08) mi_intr &= ~0x4;

    if (line_cycle >= 6150) {
      line_cycle = 0, ++scanline;
      if (scanline == 484) {
        scanline = 0, mi_intr |= 0x2; // SI INTR
        if (intrline == scanline) mi_intr |= 0x8; // VI INTR
      }

      // handle SDL events
      for (SDL_Event e; SDL_PollEvent(&e);) {
        if (e.type == SDL_QUIT) exit(0);
      }
        
      if (pixels) {
        // draw screen texture
        SDL_UpdateTexture(texture, nullptr, pixels, 640 * 4);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
      }
    }

    // update IP2 based on MI_INTR and MI_MASK
    if (mi_intr & mi_mask) reg_array[13 + dev_cop0] |= (0x1 << 10);
    else reg_array[13 + dev_cop0] &= ~(0x1 << 10);

    // check ++COUNT against COMPARE, set IP7
    uint64_t &count = reg_array[9 + dev_cop0];
    uint64_t compare = reg_array[11 + dev_cop0];
    for (uint32_t i = 0; i < (block_time >> 1); ++i) {
      count = (count + 1) & 0xffffffff;
      if (count == compare) reg_array[13 + dev_cop0] |= (0x1 << 15);
    }

    // if interrupt enabled and triggered, goto interrupt handler
    uint8_t ip = (reg_array[13 + dev_cop0] >> 8) & 0xff;
    uint8_t im = (reg_array[12 + dev_cop0] >> 8) & 0xff;
    if ((reg_array[12 + dev_cop0] & 0x3) == 0x1 && (ip & im)) {
      printf("Jumping to interrupt instead of %x\n", pc);
      printf("IP: %x MI: %x MASK: %x\n", ip, mi_intr, mi_mask);
      printf("count: %llx compare: %llx\n", reg_array[9 + dev_cop0], reg_array[11 + dev_cop0]);
      reg_array[14 + dev_cop0] = pc;
      pc = 0x80000180; reg_array[12 + dev_cop0] |= 0x2;
    }
  }
}
