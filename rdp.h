#ifndef RDP_H
#define RDP_H

#include "asmjit/asmjit.h"

using namespace asmjit;

struct RdpJit {
  const x86::Mem x86_pixels(uint32_t x, uint32_t y) {
    return x86::dword_ptr(x86::rdi, (x + y * width) << 2);
  }

  const x86::Mem x86_spill(uint8_t reg) {
    return x86::dword_ptr(x86::rbp, reg << 2);
  }

  void fill_rectangle() {
    uint8_t yh = (instr & 0x3ff), xh = ((instr >> 12) & 0x3ff);
    uint8_t yl = ((instr >> 32) & 0x3ff), xl = ((instr >> 44) & 0x3ff);
    // ignore subpixels, blending, etc.
    for (uint32_t y = (yh >> 2); y < (yl >> 2); ++y) {
      for (uint32_t x = (xh >> 2); x < (xl >> 2); ++x) {
        as.movaps(x86_pixels(x, y), x86::xmm0);
      }
    }
  }

  uint32_t jit_block() {
    as.push(x86::rbp);
    as.mov(x86::rbp, reinterpret_cast<uint64_t>(reg_array));
    while (dpc_pc != dpc_end) {
      uint64_t instr = fetch(dpc_pc++);
      switch (instr >> 56) {
        // make sure xmm register state is correct around function calls
        case 0x36: fill_rectangle(instr); break;
        default: printf("RDP instruction %x\n", instr); break;
      }
    }
    as.pop(x86::rbp);
    as.ret();
    return cycles;
  }
}

namespace RDP {
  // interpret a sequence of config, then JIT rasterizer operations,
  // end block when config changes. Before reusing block, check
  // config is the same.
  uint32_t dpc_start = 0x0, dpc_end = 0x0;
  uint32_t width = 0x0, height = 0x0;

  struct Block {
    Function code;
    uint32_t cycles;
    uint32_t hash;
    
    bool valid(uint32_t hash2) {
      // hash is unique to code & configuration
      return code && hash == hash2;
    }
  };

  uint32_t dpc_read(uint8_t offset) {

  void dpc_write(uint8_t offset, uint32_t val) {
    switch (offset) {
      case 0x0: dpc_start = val;
      case 0x4: dpc_end = val;
    }
  }

  template <uint8_t reg>
  uint32_t set_color(uint32_t instr) {
    color[reg] = instr & 0xffffffff;
  }

  void set_color_image(uint64_t instr) {
    ram_addr = instr & 0x1fffffff;
    width = ((instr >> 32) & 0x3ff) + 1;
    size = (instr >> 51) & 0x3;
  }

  void run_block() {
    Block &block = blocks[dpc_pc];
    // check block ends before dpc_end
    if (!block.valid()) {
      block.conf = conf;
      CodeHolder code;
      code.init(runtime.codeInfo());
      RdpJit jit(code);

      block.cycles = jit_block();
      runtime.add(&block.code, &code);
    }
    dpc_pc = block.code();
    return dpc_pc;
  }

  void rdp_update(uint32_t cycles) {
    // interpret config instructions 
    uint32_t dpc_pc = dpc_start;
    while (dpc_pc != dpc_end) {
      uint64_t instr = fetch(dpc_pc++);
      switch (instr >> 56) {
        case 0x37: set_fill_color(instr); break;
        case 0x3f: set_color_image(instr); break;
        default: dpc_pc = run_block(); break;
      }
    }
  }
}

#endif
