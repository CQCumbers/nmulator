#include "asmjit/asmjit.h"
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

using namespace asmjit;
typedef uint32_t (*Function)();

std::unordered_map<uint32_t, Function> blocks;
uint8_t *mem_array;
uint8_t zero_array[] = {0};
uint64_t reg_array[0x21];
const uint32_t block_end = 0x04ffffff;
const uint32_t addr_mask = 0x1fffffff;

uint8_t call_stack = 0;

/* === Instruction Decoding == */

inline uint32_t mem_read(uint32_t addr) {
  uint32_t val = *reinterpret_cast<uint32_t*>(mem_array + (addr & addr_mask));
  return __builtin_bswap32(val);
}

inline void mem_write(uint32_t addr, uint32_t val) {
  uint32_t *place = reinterpret_cast<uint32_t*>(mem_array + (addr & addr_mask));
  *place = val;
}

inline int32_t target(uint32_t instr) {
  int32_t val = (instr & 0x3ffffff) << 2;
  return (val ^ 0x8000000) - 0x8000000;
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

uint8_t x86_reg(uint8_t reg) {
  uint8_t mapping[0x20] = {0};
  mapping[5] = 2, mapping[6] = 3;
  return mapping[reg];
}

const x86::Mem x86_spill(uint8_t reg) {
  return x86::dword_ptr(x86::rbp, reg << 3);
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

  void translate_addr(int32_t offset) {
    as.add(x86::eax, offset);

    // calc page table entry for address (0 for now)
    //as.xor_(x86::ecx, x86::ecx);

    // add base addr from page table at rbp + 0x100
    //as.add(x86::rcx, x86::rbp);
    as.and_(x86::eax, addr_mask);
    as.add(x86::rax, x86::dword_ptr(x86::rbp, 0x100));
  }

  void lw(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    if (rsx) as.mov(x86::eax, x86::gpd(rsx));
    else as.mov(x86::eax, x86_spill(rs(instr)));
    translate_addr(imm(instr));
    if (rtx) as.movbe(x86::gpd(rtx), x86::dword_ptr(x86::rax));
    else {
      as.movbe(x86::ecx, x86::dword_ptr(x86::rax));
      as.mov(x86_spill(rt(instr)), x86::ecx);
    }
  }

  void sw(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    if (rsx) as.mov(x86::eax, x86::gpd(rsx));
    else as.mov(x86::eax, x86_spill(rs(instr)));
    translate_addr(imm(instr));
    if (rtx) as.movbe(x86::dword_ptr(x86::rax), x86::gpd(rtx));
    else {
      as.mov(x86::ecx, x86_spill(rt(instr)));
      as.movbe(x86::dword_ptr(x86::rax), x86::ecx);
    }
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
    } else if (imm(instr) == 0) {
      // ADDIU RT, RS, 0
      move(rt(instr), rs(instr));
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

  void ori(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // ORI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpd(rtx), uimm(instr));
      else as.mov(x86_spill(rt(instr)), uimm(instr));
    } else if (rt(instr) == rs(instr)) {
      // ORI RT, RT, IMMEDIATE
      if (rtx) as.or_(x86::gpd(rtx), uimm(instr));
      else as.or_(x86_spill(rt(instr)), uimm(instr));
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
    } else if (rt(instr) == rs(instr)) {
      // ANDI RT, RT, IMMEDIATE
      if (rtx) as.and_(x86::gpd(rtx), uimm(instr));
      else as.and_(x86_spill(rt(instr)), uimm(instr));
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
    } else if (rt(instr) == rs(instr)) {
      // XORI RT, RT, IMMEDIATE
      if (rtx) as.xor_(x86::gpd(rtx), uimm(instr));
      else as.xor_(x86_spill(rt(instr)), uimm(instr));
    } else {
      // XORI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.xor_(x86::gpd(rtx), uimm(instr));
      else as.xor_(x86_spill(rt(instr)), uimm(instr));
    }
  }

  void sll(uint32_t instr) {
    if (rd(instr) == 0 || sa(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SLL RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rd(instr) == rt(instr)) {
      // SLL RD, RD, IMMEDIATE
      if (rdx) as.shl(x86::gpd(rdx), sa(instr));
      else as.shl(x86_spill(rd(instr)), sa(instr));
    } else {
      // SLL RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shl(x86::gpd(rdx), sa(instr));
      else as.shl(x86_spill(rd(instr)), sa(instr));
    }
  }

  void srl(uint32_t instr) {
    if (rd(instr) == 0 || sa(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SRL RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rd(instr) == rt(instr)) {
      // SRL RD, RD, IMMEDIATE
      if (rdx) as.shr(x86::gpd(rdx), sa(instr));
      else as.shr(x86_spill(rd(instr)), sa(instr));
    } else {
      // SRL RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.shr(x86::gpd(rdx), sa(instr));
      else as.shr(x86_spill(rd(instr)), sa(instr));
    }
  }

  void sra(uint32_t instr) {
    if (rd(instr) == 0 || sa(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // SLL RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spill(rd(instr)), 0);
    } else if (rd(instr) == rt(instr)) {
      // SLL RD, RD, IMMEDIATE
      if (rdx) as.sar(x86::gpd(rdx), sa(instr));
      else as.sar(x86_spill(rd(instr)), sa(instr));
    } else {
      // SLL RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) as.sar(x86::gpd(rdx), sa(instr));
      else as.sar(x86_spill(rd(instr)), sa(instr));
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

  uint32_t j(uint32_t instr, uint32_t pc) {
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86::edi, dst);
    return block_end;
  }

  uint32_t jal(uint32_t instr, uint32_t pc) {
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86_spill(31), pc + 4);
    as.mov(x86::edi, dst);
    ++call_stack;
    return block_end;
  }

  uint32_t jr(uint32_t instr) {
    uint32_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::edi, x86::gpd(rsx));
    else as.mov(x86::edi, x86_spill(rs(instr)));
    --call_stack;
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

  uint32_t blez(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rsx), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovg(x86::edi, x86::eax);
    return block_end;
  }

  uint32_t bgtz(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rsx), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    as.cmovle(x86::edi, x86::eax);
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

  uint32_t blezl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) {
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rsx), 0);
    as.mov(x86::edi, pc + 4);
    as.jg(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t bgtzl(uint32_t instr, uint32_t pc) {
    if (rs(instr) == 0) return pc + 4;
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rsx), 0);
    as.mov(x86::edi, pc + 4);
    as.jle(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  // subu, multu, mflo, addu, jr

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
      case 0x08: next_pc = jr(instr); break;
      case 0x20: addu(instr); break;
      case 0x21: addu(instr); break;
      case 0x24: and_(instr); break;
      case 0x25: or_(instr); break;
      case 0x2a: slt(instr); break;
      case 0x2b: sltu(instr); break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  void cop0(uint32_t instr) {
    printf("COP0 instruction %x\n", instr);
  }

  void jit_block(uint32_t pc) {
    end_label = as.newLabel();
    as.push(x86::rbp);
    as.mov(x86::rbp, reinterpret_cast<uint64_t>(&reg_array));

    for (uint32_t next_pc = pc + 4; pc != block_end;) {
      printf("%x\n", pc);
      uint32_t instr = mem_read(pc);
      pc = next_pc, next_pc += 4;
      switch (instr >> 26) {
        case 0x00: next_pc = special(instr, pc); break;
        case 0x02: next_pc = j(instr, pc); break;
        case 0x03: next_pc = jal(instr, pc); break;
        case 0x04: next_pc = beq(instr, pc); break;
        case 0x05: next_pc = bne(instr, pc); break;
        case 0x06: next_pc = blez(instr, pc); break;
        case 0x07: next_pc = bgtz(instr, pc); break;
        case 0x08: addiu(instr); break;
        case 0x09: addiu(instr); break;
        case 0x0a: slti(instr); break;
        case 0x0b: sltiu(instr); break;
        case 0x0c: andi(instr); break;
        case 0x0d: ori(instr); break;
        case 0x0e: xori(instr); break;
        case 0x0f: lui(instr); break;
        case 0x10: cop0(instr); break;
        case 0x14: next_pc = beql(instr, pc); break;
        case 0x15: next_pc = bnel(instr, pc); break;
        case 0x16: next_pc = blezl(instr, pc); break;
        case 0x17: next_pc = bgtzl(instr, pc); break;
        case 0x23: lw(instr); break;
        case 0x2b: sw(instr); break;
        case 0x2f: cache(instr); break;
        default: invalid(instr); break;
      }
    }

    as.bind(end_label);
    as.mov(x86::rax, x86::rdi);
    as.pop(x86::rbp);
    as.ret();
  }
};

int main(int argc, char* argv[]) {
  FILE *file = fopen(argv[1], "r");
  uint32_t pc = 0xa4000040;
  JitRuntime runtime;

  if (!file) {
    printf("error: can't open file %s", argv[1]);
    exit(1);
  }

  mem_array = static_cast<uint8_t*>(mmap(NULL, addr_mask,
      PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, 0, 0));
  printf("mem_array is at address: %p\n", static_cast<void*>(mem_array));
  // set sp to pif boot rom value
  reg_array[29] = 0xa4001ff0;
  // use last entry in reg_array for mem start addres
  reg_array[32] = reinterpret_cast<uint64_t>(mem_array);

  fread(mem_array + 0x04000040, 1, 4032, file);
  fclose(file);

  while (true) {
    Function &run_block = blocks[pc];
    if (!run_block) {
      CodeHolder code;
      FileLogger logger(stdout);

      code.init(runtime.codeInfo());
      code.setLogger(&logger);
      MipsJit cpu(code);

      code.init(runtime.codeInfo());
      cpu.jit_block(pc);
      runtime.add(&run_block, &code);
    }
    pc = run_block();
    printf("call stack: %d\n", call_stack);
    printf("$t0: %llx, $t3: %llx\n", reg_array[8], reg_array[11]);
    printf("Executed block %x\n", pc);
  }
}
