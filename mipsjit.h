#ifndef MIPSJIT_H
#define MIPSJIT_H

#include "r4300.h"
#include "asmjit/asmjit.h"

using namespace asmjit;

struct MipsJit {
  x86::Assembler as;
  Label end_label;
  static constexpr uint32_t block_end = 0x04ffffff;
  static constexpr uint8_t hi = R4300::hi, lo = R4300::lo;
  static constexpr uint8_t dev_cop0 = R4300::dev_cop0;

  MipsJit(CodeHolder &code) : as(&code) {}

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

  uint8_t x86_reg(uint8_t reg) {
    return 0;
  }

  const x86::Mem x86_spill(uint8_t reg) {
    return x86::dword_ptr(x86::rbp, reg << 3);
  }

  const x86::Mem x86_spilld(uint8_t reg) {
    return x86::qword_ptr(x86::rbp, reg << 3);
  }

  /* === Helper Pseudo-instructions === */

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

  /* === Instruction Translations === */

  template <typename T>
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
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    if (rtx) as.mov(x86::gpq(rtx), x86::rax);
    else as.mov(x86_spilld(rt(instr)), x86::rax);
    as.pop(x86::edi);
  }

  template <typename T>
  void sw(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  template <typename T, bool right>
  void lwl(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    // compute mask for loaded data
    as.mov(x86::ecx, x86::edi); as.and_(x86::ecx, Imm(sizeof(T) - 1));
    as.xor_(x86::rax, x86::rax); as.not_(x86::rax);
    if (right) as.sub(x86::edi, Imm(sizeof(T) - 1)), as.add(x86::ecx, 1);
    as.shl(x86::ecx, 3); as.shl(x86::rax, x86::cl);
    // read unaligned data from memory
    as.push(x86::rax);
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    as.pop(x86::rcx);
    // apply mask depending on direction
    if (right) {
      as.mov(x86::rdx, x86::rcx); as.not_(x86::rcx);
      as.cmp(x86::rcx, 0); as.cmove(x86::rcx, x86::rdx);
    }
    as.and_(x86::rax, x86::rcx); as.not_(x86::rcx);
    if (rtx) {
      as.movsxd(x86::rdx, x86::gpq(rtx));
      as.and_(x86::rdx, x86::rcx); as.or_(x86::rdx, x86::rax);
      as.mov(x86::gpq(rtx), x86::rdx);
    } else {
      as.movsxd(x86::rdx, x86_spilld(rt(instr)));
      as.and_(x86::rdx, x86::rcx); as.or_(x86::rdx, x86::rax);
      as.mov(x86_spilld(rt(instr)), x86::rdx);
    }
    as.pop(x86::edi);
  }

  template <typename T, bool right>
  void swl(uint32_t instr) {
    printf("SWL (unaligned store)\n");
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpq(rsx), imm(instr)));
    else {
      as.mov(x86::rax, x86_spilld(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::rax, imm(instr)));
    }
    // compute mask for loaded data
    as.mov(x86::ecx, x86::edi); as.and_(x86::ecx, Imm(sizeof(T) - 1));
    as.xor_(x86::rax, x86::rax); as.not_(x86::rax);
    if (right) as.sub(x86::edi, Imm(sizeof(T) - 1)), as.add(x86::ecx, 1);
    as.shl(x86::ecx, 3); as.shl(x86::rax, x86::cl);
    // read previous data from memory
    as.push(x86::rax);
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    as.pop(x86::rcx);
    // apply mask depending on direction
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    if (right) {
      as.mov(x86::rdx, x86::rcx); as.not_(x86::rcx);
      as.cmp(x86::rcx, 0); as.cmove(x86::rcx, x86::rdx);
    }
    as.and_(x86::rsi, x86::rcx); as.not_(x86::rcx);
    as.and_(x86::rax, x86::rcx); as.or_(x86::rsi, x86::rax);
    // write masked data to memory
    as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
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

  uint32_t jalr(uint32_t instr, uint32_t pc) {
    as.mov(x86_spill(31), pc + 4);
    uint32_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::edi, x86::gpd(rsx));
    else as.mov(x86::edi, x86_spill(rs(instr)));
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
    if (rd(instr) == 11) as.and_(x86_spill(13 + dev_cop0), ~0x8000);
    if (rt(instr) == 0) as.mov(x86_spill(rd(instr) + dev_cop0), 0);
    else move(rd(instr) + dev_cop0, rt(instr));
  }

  uint32_t eret() {
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
      case 0x09: next_pc = jalr(instr, pc); break;
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
          case 0x18: next_pc = eret(); break;
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

  uint32_t jit_block() {
    as.push(x86::rbp);
    as.mov(x86::rbp, reinterpret_cast<uint64_t>(R4300::reg_array));
    //as.push(x86::gpd(x86_reg(5)));
    //as.mov(x86::gpd(x86_reg(5)), x86_spill(5));
    //as.push(x86::gpd(x86_reg(6)));
    //as.mov(x86::gpd(x86_reg(6)), x86_spill(6));

    end_label = as.newLabel();
    uint32_t cycles = 0, pc = R4300::pc;
    for (uint32_t next_pc = pc + 4; pc != block_end; ++cycles) {
      uint32_t instr = R4300::fetch(pc);
      printf("%x: %x\n", pc, instr);
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
        case 0x1a: lwl<uint64_t, false>(instr); break;
        case 0x1b: lwl<uint64_t, true>(instr); break;
        case 0x20: lw<int8_t>(instr); break;
        case 0x21: lw<int16_t>(instr); break;
        case 0x22: lwl<int32_t, false>(instr); break;
        case 0x23: lw<int32_t>(instr); break;
        case 0x24: lw<uint8_t>(instr); break;
        case 0x25: lw<uint16_t>(instr); break;
        case 0x26: lwl<int32_t, true>(instr); break;
        case 0x27: lw<uint32_t>(instr); break;
        case 0x28: sw<uint8_t>(instr); break;
        case 0x29: sw<uint16_t>(instr); break;
        case 0x2a: swl<uint32_t, false>(instr); break;
        case 0x2b: sw<uint32_t>(instr); break;
        case 0x2c: swl<uint64_t, false>(instr); break;
        case 0x2d: swl<uint64_t, true>(instr); break;
        case 0x2e: swl<uint32_t, true>(instr); break;
        case 0x2f: cache(instr); break;
        case 0x31: cop1(instr); break; // LWC1
        case 0x35: cop1(instr); break; // LDC1
        case 0x37: lw<uint64_t>(instr); break;
        case 0x39: cop1(instr); break; // SWC1
        case 0x3d: cop1(instr); break; // SDC1
        case 0x3f: sw<uint64_t>(instr); break;
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

#endif
