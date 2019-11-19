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
  static constexpr uint8_t dev_cop1 = R4300::dev_cop1;
  enum class Dir { ll, rl, ra };
  enum class CC { gt, lt, ge, le };
  enum class Op { add, sub, mul, div };

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
    // 64 bit register to register
    if (dst == src) return;
    uint8_t dstx = x86_reg(dst), srcx = x86_reg(src);
    if (dstx) {
      if (srcx) as.mov(x86::gpq(dstx), x86::gpq(srcx));
      else as.mov(x86::gpq(dstx), x86_spilld(src));
    } else {
      if (srcx) as.mov(x86_spilld(dst), x86::gpq(srcx));
      else { 
        as.mov(x86::rax, x86_spilld(src));
        as.mov(x86_spilld(dst), x86::rax);
      }
    }
  }

  void to_eax(uint8_t src) {
    // 64 bit to 32 bit truncate
    uint8_t srcx = x86_reg(src);
    if (srcx) as.mov(x86::eax, x86::gpd(srcx));
    else as.mov(x86::eax, x86_spill(src));
  }

  void from_eax(uint8_t dst) {
    // 32 bit to 64 bit sign extend
    uint8_t dstx = x86_reg(dst);
    if (dstx) as.movsxd(x86::gpq(dstx), x86::eax);
    else {
      as.movsxd(x86::rax, x86::eax);
      as.mov(x86_spilld(dst), x86::rax);
    }
  }
  
  void compare(uint8_t reg1, uint8_t reg2) {
    // 64 bit register to register
    uint8_t reg1x = x86_reg(reg1), reg2x = x86_reg(reg2);
    if (reg2 == 0) {
      if (reg1x) as.cmp(x86::gpq(reg1x), 0);
      else as.cmp(x86_spilld(reg1), 0);
    } else {
      if (reg1x) {
        if (reg2x) as.cmp(x86::gpq(reg1x), x86::gpq(reg2x));
        else as.cmp(x86::gpq(reg1x), x86_spilld(reg2));
      } else {
        if (reg2x) as.cmp(x86_spilld(reg1), x86::gpq(reg2x));
        else {
          as.mov(x86::rax, x86_spilld(reg2));
          as.cmp(x86_spilld(reg1), x86::rax);
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
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
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
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  template <typename T, Dir dir>
  void lwl(uint32_t instr) {
    if (rt(instr) == 0) return;
    constexpr bool right = (dir != Dir::ll);
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
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
      as.and_(x86::gpq(rtx), x86::rcx);
      as.or_(x86::gpq(rtx), x86::rax);
    } else {
      as.and_(x86_spilld(rt(instr)), x86::rcx);
      as.or_(x86_spilld(rt(instr)), x86::rax);
    }
    if (sizeof(T) < 8) to_eax(rt(instr)), from_eax(rt(instr));
    as.pop(x86::edi);
  }

  template <typename T, Dir dir>
  void swl(uint32_t instr) {
    constexpr bool right = (dir != Dir::ll);
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    // compute mask for loaded data
    as.mov(x86::ecx, x86::edi); as.and_(x86::ecx, Imm(sizeof(T) - 1));
    as.xor_(x86::rax, x86::rax); as.not_(x86::rax);
    if (right) as.sub(x86::edi, Imm(sizeof(T) - 1)), as.add(x86::ecx, 1);
    as.shl(x86::ecx, 3); as.shl(x86::rax, x86::cl);
    // read previous data from memory
    as.push(x86::edi); as.push(x86::rax);
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    as.pop(x86::rcx); as.pop(x86::edi);
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
    // LUI RT, IMMEDIATE
    as.mov(x86::eax, uimm(instr) << 16);
    from_eax(rt(instr));
  }

  void addiu(uint32_t instr) {
    if (rt(instr) == 0) return;
    if (rs(instr) == 0) {
      // ADDIU RT, $0, IMMEDIATE
      as.mov(x86::eax, imm(instr));
      from_eax(rt(instr));
    } else {
      // ADDIU RT, RS, IMMEDIATE
      to_eax(rs(instr));
      as.add(x86::eax, imm(instr));
      from_eax(rt(instr));
    }
  }

  void addu(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rs(instr) == 0 && rt(instr) == 0) {
      // ADD RD, $0, $0 
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // ADD RD, $0, RT
      to_eax(rt(instr));
      from_eax(rd(instr));
    } else if (rt(instr) == 0) {
      // ADD RD, RS, $0
      to_eax(rs(instr));
      from_eax(rd(instr));
    } else {
      // ADD RD, RS, RT
      to_eax(rt(instr));
      uint8_t rsx = x86_reg(rs(instr));
      if (rsx) as.add(x86::eax, x86::gpd(rsx));
      else as.add(x86::eax, x86_spill(rs(instr)));
      from_eax(rd(instr));
    }
  }

  void subu(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rs(instr) == 0 && rt(instr) == 0) {
      // SUB RD, $0, $0 
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SUB RD, $0, RT
      to_eax(rt(instr));
      as.neg(x86::eax);
      from_eax(rd(instr));
    } else if (rt(instr) == 0) {
      // SUB RD, RS, $0
      to_eax(rs(instr));
      from_eax(rd(instr));
    } else {
      // SUB RD, RS, RT
      to_eax(rs(instr));
      uint8_t rtx = x86_reg(rt(instr));
      if (rtx) as.sub(x86::eax, x86::gpd(rtx));
      else as.sub(x86::eax, x86_spill(rt(instr)));
      from_eax(rd(instr));
    }
  }

  void ori(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // ORI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpq(rtx), uimm(instr));
      else as.mov(x86_spilld(rt(instr)), uimm(instr));
    } else {
      // ORI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.or_(x86::gpq(rtx), uimm(instr));
      else as.or_(x86_spilld(rt(instr)), uimm(instr));
    }
  }

  void or_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // OR RD, $0, $0 
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0 || rt(instr) == rs(instr)) {
      // OR RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // OR RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      // OR RD, RD, RT
      uint8_t rtx = x86_reg(rt(instr));
      if (rdx) {
        if (rtx) as.or_(x86::gpq(rdx), x86::gpq(rtx));
        else as.or_(x86::gpq(rdx), x86_spilld(rt(instr)));
      } else {
        if (rtx) as.or_(x86_spilld(rd(instr)), x86::gpq(rtx));
        else {
          as.mov(x86::rax, x86_spilld(rt(instr)));
          as.or_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    } else {
      // OR RD, RS, RT
      uint8_t rsx = x86_reg(rs(instr));
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.or_(x86::gpq(rdx), x86::gpq(rsx));
        else as.or_(x86::gpq(rdx), x86_spilld(rs(instr)));
      } else {
        if (rsx) as.or_(x86_spilld(rd(instr)), x86::gpq(rsx));
        else {
          as.mov(x86::rax, x86_spilld(rs(instr)));
          as.or_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    }
  }

  void andi(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0 || uimm(instr) == 0) {
      // ANDI RT, $0, IMMEDIATE
      if (rtx) as.xor_(x86::gpq(rtx), x86::gpq(rtx));
      else as.mov(x86_spilld(rt(instr)), 0);
    } else {
      // ANDI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.and_(x86::gpq(rtx), uimm(instr));
      else as.and_(x86_spilld(rt(instr)), uimm(instr));
    }
  }

  void and_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 || rt(instr) == 0) {
      // AND RD, $0, RT
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rt(instr) == rs(instr)) {
      // AND RD, RT, RT
      move(rd(instr), rt(instr));
    } else if (rd(instr) == rs(instr)) {
      // AND RD, RD, RT
      uint8_t rtx = x86_reg(rt(instr));
      if (rdx) {
        if (rtx) as.and_(x86::gpq(rdx), x86::gpq(rtx));
        else as.and_(x86::gpq(rdx), x86_spilld(rt(instr)));
      } else {
        if (rtx) as.and_(x86_spilld(rd(instr)), x86::gpq(rtx));
        else {
          as.mov(x86::rax, x86_spilld(rt(instr)));
          as.and_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    } else {
      // AND RD, RS, RT
      uint8_t rsx = x86_reg(rs(instr));
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.and_(x86::gpq(rdx), x86::gpq(rsx));
        else as.and_(x86::gpq(rdx), x86_spilld(rs(instr)));
      } else {
        if (rsx) as.and_(x86_spilld(rd(instr)), x86::gpq(rsx));
        else {
          as.mov(x86::rax, x86_spilld(rs(instr)));
          as.and_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    }
  }

  void xori(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // XORI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpq(rtx), uimm(instr));
      else as.mov(x86_spilld(rt(instr)), uimm(instr));
    } else {
      // XORI RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.xor_(x86::gpq(rtx), uimm(instr));
      else as.xor_(x86_spilld(rt(instr)), uimm(instr));
    }
  }

  void xor_(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == rt(instr)) {
      // XOR RD, RT, RT
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
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
        if (rtx) as.xor_(x86::gpq(rdx), x86::gpq(rtx));
        else as.xor_(x86::gpq(rdx), x86_spilld(rt(instr)));
      } else {
        if (rtx) as.xor_(x86_spilld(rd(instr)), x86::gpq(rtx));
        else {
          as.mov(x86::rax, x86_spilld(rt(instr)));
          as.xor_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    } else {
      // XOR RD, RS, RT
      uint8_t rsx = x86_reg(rs(instr));
      move(rd(instr), rt(instr));
      if (rdx) {
        if (rsx) as.xor_(x86::gpq(rdx), x86::gpq(rsx));
        else as.xor_(x86::gpq(rdx), x86_spilld(rs(instr)));
      } else {
        if (rsx) as.xor_(x86_spilld(rd(instr)), x86::gpq(rsx));
        else {
          as.mov(x86::rax, x86_spilld(rs(instr)));
          as.xor_(x86_spilld(rd(instr)), x86::rax);
        }
      }
    }
  }

  void nor(uint32_t instr) {
    if (rd(instr) == 0) return;
    or_(instr);
    uint8_t rdx = x86_reg(rd(instr));
    if (rdx) as.not_(x86::gpq(rdx));
    else as.not_(x86_spilld(rd(instr)));
  }

  template <Dir dir, bool add32>
  void dsll(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // DSLL32 RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // DSLL32 RD, RT, IMMEDIATE
      move(rd(instr), rt(instr));
      if (rdx) {
        switch (dir) {
          case Dir::ll: as.shl(x86::gpq(rdx), sa(instr) + 32 * add32); break;
          case Dir::rl: as.shr(x86::gpq(rdx), sa(instr) + 32 * add32); break;
          case Dir::ra: as.sar(x86::gpq(rdx), sa(instr) + 32 * add32); break;
        }
      } else {
        switch (dir) {
          case Dir::ll: as.shl(x86_spilld(rd(instr)), sa(instr) + 32 * add32); break;
          case Dir::rl: as.shr(x86_spilld(rd(instr)), sa(instr) + 32 * add32); break;
          case Dir::ra: as.sar(x86_spilld(rd(instr)), sa(instr) + 32 * add32); break;
        }
      }
    }
  }

  template <Dir dir>
  void sll(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rt(instr) == 0) {
      // SLL RD, $0, IMMEDIATE
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // SLL RD, RT, IMMEDIATE
      to_eax(rt(instr));
      switch (dir) {
        case Dir::ll: as.shl(x86::eax, sa(instr)); break;
        case Dir::rl: as.shr(x86::eax, sa(instr)); break;
        case Dir::ra: as.sar(x86::eax, sa(instr)); break;
      }
      from_eax(rd(instr));
    }
  }

  template <Dir dir>
  void sllv(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rt(instr) == 0) {
      // SLLV RD, $0, RS
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SLLV RD, RT, $0
      to_eax(rt(instr));
      from_eax(rd(instr));
    } else {
      // SLLV RD, RT, RS
      to_eax(rt(instr));
      uint8_t rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
      else as.mov(x86::ecx, x86_spill(rs(instr)));
      switch (dir) {
        case Dir::ll: as.shl(x86::eax, x86::cl); break;
        case Dir::rl: as.shr(x86::eax, x86::cl); break;
        case Dir::ra: as.sar(x86::eax, x86::cl); break;
      }
      from_eax(rd(instr));
    }
  }

  void slti(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint32_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // SLTI RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpq(rtx), 0 < imm(instr));
      else as.mov(x86_spilld(rt(instr)), 0 < imm(instr));
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      // SLTI RT, RS, IMMEDIATE
      if (rsx) as.cmp(x86::gpq(rsx), imm(instr));
      else as.cmp(x86_spilld(rs(instr)), imm(instr));
      as.setl(x86::al);
      if (rtx) as.movzx(x86::gpq(rtx), x86::al);
      else {
        as.movzx(x86::rax, x86::al);
        as.mov(x86_spill(rt(instr)), x86::rax);
      }
    }
  }

  void sltiu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint32_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // SLTIU RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpq(rtx), 0 != imm(instr));
      else as.mov(x86_spilld(rt(instr)), 0 != imm(instr));
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      // SLTIU RT, RS, IMMEDIATE
      if (rsx) as.cmp(x86::gpq(rsx), imm(instr));
      else as.cmp(x86_spilld(rs(instr)), imm(instr));
      as.setb(x86::al);
      if (rtx) as.movzx(x86::gpq(rtx), x86::al);
      else {
        as.movzx(x86::rax, x86::al);
        as.mov(x86_spill(rt(instr)), x86::rax);
      }
    }
  }

  void slt(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rt(instr) == rs(instr)) {
      // SLT RD, RT, RT
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // SLT RD, RS, RT
      compare(rs(instr), rt(instr));
      as.setl(x86::al);
      if (rdx) as.movzx(x86::gpq(rdx), x86::al);
      else {
        as.movzx(x86::rax, x86::al);
        as.mov(x86_spilld(rd(instr)), x86::rax);
      }
    }
  }

  void sltu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint32_t rdx = x86_reg(rd(instr));
    if (rt(instr) == rs(instr)) {
      // SLTU RD, RT, RT
      if (rdx) as.xor_(x86::gpq(rdx), x86::gpq(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // SLTU RD, RS, RT
      compare(rs(instr), rt(instr));
      as.setb(x86::al);
      if (rdx) as.movzx(x86::gpq(rdx), x86::al);
      else {
        as.movzx(x86::rax, x86::al);
        as.mov(x86_spilld(rd(instr)), x86::rax);
      }
    }
  }

  void mult(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spill(lo), 0);
      as.mov(x86_spill(hi), 0);
    } else {
      to_eax(rs(instr));
      uint32_t rtx = x86_reg(rt(instr));
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
      to_eax(rs(instr));
      uint32_t rtx = x86_reg(rt(instr));
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
      to_eax(rs(instr));
      as.cdq();
      Label after_div = as.newLabel();
      uint32_t rtx = x86_reg(rt(instr));
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
      to_eax(rs(instr));
      as.xor_(x86::edx, x86::edx);
      Label after_div = as.newLabel();
      uint32_t rtx = x86_reg(rt(instr));
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
    as.mov(x86::eax, x86_spill(hi));
    from_eax(rd(instr));
  }

  void mthi(uint32_t instr) {
    if (rd(instr) == 0) as.mov(x86_spill(hi), 0);
    to_eax(rd(instr));
    as.mov(x86_spill(hi), x86::eax);
  }

  void mflo(uint32_t instr) {
    if (rd(instr) == 0) return;
    as.mov(x86::eax, x86_spill(lo));
    from_eax(rd(instr));
  }

  void mtlo(uint32_t instr) {
    if (rd(instr) == 0) as.mov(x86_spill(lo), 0);
    to_eax(rd(instr));
    as.mov(x86_spill(lo), x86::eax);
  }

  uint32_t j(uint32_t instr, uint32_t pc) {
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86::edi, dst);
    return block_end;
  }

  uint32_t jal(uint32_t instr, uint32_t pc) {
    as.mov(x86::eax, pc + 4), from_eax(31);
    uint32_t dst = (pc & 0xf0000000) | target(instr);
    as.mov(x86::edi, dst);
    return block_end;
  }

  uint32_t jalr(uint32_t instr, uint32_t pc) {
    as.mov(x86::eax, pc + 4), from_eax(31);
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

  template <CC cond, bool link>
  uint32_t bltz(uint32_t instr, uint32_t pc) {
    if (link) as.mov(x86::eax, pc + 4), from_eax(31);
    if (rs(instr) == 0) {
      if (cond == CC::lt || cond == CC::gt) return pc;
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    switch (cond) {
      case CC::lt: as.cmovl(x86::edi, x86::eax); break;
      case CC::gt: as.cmovg(x86::edi, x86::eax); break;
      case CC::le: as.cmovle(x86::edi, x86::eax); break;
      case CC::ge: as.cmovge(x86::edi, x86::eax); break;
    }
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

  template <CC cond, bool link>
  uint32_t bltzl(uint32_t instr, uint32_t pc) {
    if (link) as.mov(x86::eax, pc + 4), from_eax(31);
    if (rs(instr) == 0) {
      if (cond == CC::lt || cond == CC::gt) return pc + 4;
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.cmp(x86::gpd(rsx), 0);
    else as.cmp(x86_spill(rs(instr)), 0);
    as.mov(x86::edi, pc + 4);
    switch (cond) {
      case CC::lt: as.jge(end_label); break;
      case CC::gt: as.jle(end_label); break;
      case CC::le: as.jg(end_label); break;
      case CC::ge: as.jl(end_label); break;
    }
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
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

  void mfc0(uint32_t instr) {
    printf("Read from COP0 reg %d\n", rd(instr));
    if (rt(instr) == 0) return;
    move(rt(instr), rd(instr) + dev_cop0);
  }

  void mtc0(uint32_t instr) {
    printf("Write to COP0 reg %d\n", rd(instr));
    if (rd(instr) == 11) as.and_(x86_spill(13 + dev_cop0), ~0x8000);
    if (rt(instr) == 0) as.mov(x86_spill(rd(instr) + dev_cop0), 0);
    else move(rd(instr) + dev_cop0, rt(instr));
  }

  template <Op operation>
  void add_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1);
    if (operation == Op::add) {
      if (rtx) as.addss(x86::xmm0, x86::xmm(rtx));
      else as.addss(x86::xmm0, x86_spill(rt(instr) + dev_cop1));
    } else if (operation == Op::sub) {
      if (rtx) as.subss(x86::xmm0, x86::xmm(rtx));
      else as.subss(x86::xmm0, x86_spill(rt(instr) + dev_cop1));
    } else if (operation == Op::mul) {
      if (rtx) as.mulss(x86::xmm0, x86::xmm(rtx));
      else as.mulss(x86::xmm0, x86_spill(rt(instr) + dev_cop1));
    } else if (operation == Op::div) {
      if (rtx) as.divss(x86::xmm0, x86::xmm(rtx));
      else as.divss(x86::xmm0, x86_spill(rt(instr) + dev_cop1));
    }
    uint8_t sax = x86_reg(sa(instr) + dev_cop1);
    if (sax) as.movss(x86::xmm(sax), x86::xmm0);
    else as.movss(x86_spill(sa(instr) + dev_cop1), x86::xmm0);
  }

  void c_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1);
    if (rtx) as.ucomiss(x86::xmm0, x86::xmm(rtx));
    else as.ucomiss(x86::xmm0, x86_spill(rt(instr) + dev_cop1));
    Label after_set = as.newLabel();
    as.mov(x86::eax, 0x1); // False
    switch (instr & 0x7) {
      case 0x1: as.setnp(x86::al); break; // Unordered
      case 0x2: as.jp(after_set); // Equal / Ordered
      case 0x3: as.setne(x86::al); break; // Equal / Unordered
      case 0x4: as.jp(after_set); // Less / Ordered
      case 0x5: as.setae(x86::al); break; // Less / Unordered
      case 0x6: as.jp(after_set); // Less / Equal / Ordered
      case 0x7: as.seta(x86::al); break;  // Less / Equal / Unordered
    }
    uint8_t cc = sa(instr) >> 2;
    uint32_t mask = (cc ? (0x1000 << cc) : 0x800);
    as.bind(after_set); as.sub(x86::eax, 1);
    as.and_(x86::eax, mask); as.and_(x86_spill(31 + dev_cop1), ~mask);
    as.or_(x86_spill(31 + dev_cop1), x86::eax);
  }

  template <bool cond>
  uint32_t bc1f(uint32_t instr, uint32_t pc) {
    uint8_t cc = rt(instr) >> 2;
    uint32_t mask = (cc ? 0x800 : (0x1000 << cc));
    as.and_(x86_spill(31 + dev_cop1), ~mask);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    if (cond) as.cmovnz(x86::edi, x86::eax);
    else as.cmovz(x86::edi, x86::eax);
    return block_end;
  }

  template <bool cond>
  uint32_t bc1fl(uint32_t instr, uint32_t pc) {
    uint8_t cc = rt(instr) >> 2;
    uint32_t mask = (cc ? 0x800 : (0x1000 << cc));
    as.and_(x86_spill(31 + dev_cop1), ~mask);
    as.mov(x86::edi, pc + 4);
    if (cond) as.jz(end_label);
    else as.jnz(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  void mov_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    uint8_t sax = x86_reg(sa(instr) + dev_cop1);
    if (sax) as.movss(x86::xmm(sax), x86::xmm0);
    else as.movss(x86_spill(sa(instr) + dev_cop1), x86::xmm0);
  }

  void mfc1(uint32_t instr) {
    printf("Read from COP1 reg %d\n", rd(instr));
    if (rt(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    as.movss(x86_spill(rt(instr)), x86::xmm0);
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1);
    if (rtx) as.mov(x86::gpd(rtx), x86_spill(rt(instr)));
  }

  void mtc1(uint32_t instr) {
    printf("Write to COP1 reg %d\n", rd(instr));
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86_spill(rt(instr)), x86::gpd(rtx));
    as.movss(x86::xmm0, x86_spill(rt(instr)));
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm(rdx), x86::xmm0);
    else as.movss(x86_spill(rd(instr) + dev_cop1), x86::xmm0);
  }

  template <typename T>
  void lwc1(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1), rsx = x86_reg(rs(instr));
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    as.mov(x86_spilld(rt(instr) + dev_cop1), x86::rax);
    if (rtx) as.movss(x86::xmm(rtx), x86_spilld(rt(instr) + dev_cop1));
    as.pop(x86::edi);
  }

  template <typename T>
  void swc1(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1), rsx = x86_reg(rs(instr));
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi);
    as.push(x86::esi);
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.movss(x86_spilld(rt(instr) + dev_cop1), x86::xmm(rtx));
    as.mov(x86::rsi, x86_spilld(rt(instr) + dev_cop1));
    as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    as.pop(x86::esi);
    as.pop(x86::edi);
  }

  void invalid(uint32_t instr) {
    printf("Unimplemented instruction %x\n", instr);
    exit(0);
  }

  /* === Basic Block Translation ==*/

  uint32_t special(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch (instr & 0x3f) {
      case 0x00: sll<Dir::ll>(instr); break;
      case 0x02: sll<Dir::rl>(instr); break;
      case 0x03: sll<Dir::ra>(instr); break;
      case 0x04: sllv<Dir::ll>(instr); break;
      case 0x06: sllv<Dir::rl>(instr); break;
      case 0x07: sllv<Dir::ra>(instr); break;
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
      case 0x38: dsll<Dir::ll, false>(instr); break;
      case 0x3a: dsll<Dir::rl, false>(instr); break;
      case 0x3b: dsll<Dir::ra, false>(instr); break;
      case 0x3c: dsll<Dir::ll, true>(instr); break;
      case 0x3e: dsll<Dir::rl, true>(instr); break;
      case 0x3f: dsll<Dir::ra, true>(instr); break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t regimm(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 16) & 0x1f) {
      case 0x00: next_pc = bltz<CC::lt, false>(instr, pc); break;
      case 0x01: next_pc = bltz<CC::ge, false>(instr, pc); break;
      case 0x02: next_pc = bltzl<CC::lt, false>(instr, pc); break;
      case 0x03: next_pc = bltzl<CC::ge, false>(instr, pc); break;
      case 0x10: next_pc = bltz<CC::lt, true>(instr, pc); break;
      case 0x11: next_pc = bltz<CC::ge, true>(instr, pc); break;
      case 0x12: next_pc = bltzl<CC::lt, true>(instr, pc); break;
      case 0x13: next_pc = bltzl<CC::ge, true>(instr, pc); break;
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

  uint32_t cop1(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 24) & 0x3) {
      case 0x0: // COP1/0
        switch (rs(instr)) {
          case 0x0: mfc1(instr); break;
          case 0x4: mtc1(instr); break;
          default: printf("COP1 instruction %x\n", instr); break;
        }
        break;
      case 0x1: // COP1/3
        switch (rt(instr) & 0x3) {
          case 0x0: next_pc = bc1f<false>(instr, pc); break;
          case 0x1: next_pc = bc1f<true>(instr, pc); break;
          case 0x2: next_pc = bc1fl<false>(instr, pc); break;
          case 0x3: next_pc = bc1fl<true>(instr, pc); break;
          default: invalid(instr); break;
        }
        break;
      case 0x2: // COP1/2
        switch (instr & 0x3f) {
          case 0x00: add_fmt<Op::add>(instr); break;
          case 0x01: add_fmt<Op::sub>(instr); break;
          case 0x02: add_fmt<Op::mul>(instr); break;
          case 0x03: add_fmt<Op::div>(instr); break;
          case 0x06: mov_fmt(instr); break;
          case 0x30: case 0x32: case 0x34: case 0x36:
          case 0x31: case 0x33: case 0x35: case 0x37:
            c_fmt(instr); break;
          case 0x38: case 0x3a: case 0x3c: case 0x3e:
          case 0x39: case 0x3b: case 0x3d: case 0x3f:
            c_fmt(instr); break;
          default: printf("COP1 instruction %x\n", instr); break;
        }
        break;
      default: printf("COP1 instruction %x\n", instr); break;
    }
    return next_pc;
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
      //printf("%x: %x\n", pc, instr);
      pc = next_pc, next_pc += 4;
      switch (instr >> 26) {
        case 0x00: next_pc = special(instr, pc); break;
        case 0x01: next_pc = regimm(instr, pc); break;
        case 0x02: next_pc = j(instr, pc); break;
        case 0x03: next_pc = jal(instr, pc); break;
        case 0x04: next_pc = beq(instr, pc); break;
        case 0x05: next_pc = bne(instr, pc); break;
        case 0x06: next_pc = bltz<CC::le, false>(instr, pc); break;
        case 0x07: next_pc = bltz<CC::gt, false>(instr, pc); break;
        case 0x08: addiu(instr); break; // ADDI
        case 0x09: addiu(instr); break;
        case 0x0a: slti(instr); break;
        case 0x0b: sltiu(instr); break;
        case 0x0c: andi(instr); break;
        case 0x0d: ori(instr); break;
        case 0x0e: xori(instr); break;
        case 0x0f: lui(instr); break;
        case 0x10: next_pc = cop0(instr, pc); break;
        case 0x11: next_pc = cop1(instr, pc); break;
        case 0x14: next_pc = beql(instr, pc); break;
        case 0x15: next_pc = bnel(instr, pc); break;
        case 0x16: next_pc = bltzl<CC::le, false>(instr, pc); break;
        case 0x17: next_pc = bltzl<CC::gt, false>(instr, pc); break;
        case 0x1a: lwl<uint64_t, Dir::ll>(instr); break;
        case 0x1b: lwl<uint64_t, Dir::rl>(instr); break;
        case 0x20: lw<int8_t>(instr); break;
        case 0x21: lw<int16_t>(instr); break;
        case 0x22: lwl<int32_t, Dir::ll>(instr); break;
        case 0x23: lw<int32_t>(instr); break;
        case 0x24: lw<uint8_t>(instr); break;
        case 0x25: lw<uint16_t>(instr); break;
        case 0x26: lwl<int32_t, Dir::rl>(instr); break;
        case 0x27: lw<uint32_t>(instr); break;
        case 0x28: sw<uint8_t>(instr); break;
        case 0x29: sw<uint16_t>(instr); break;
        case 0x2a: swl<uint32_t, Dir::ll>(instr); break;
        case 0x2b: sw<uint32_t>(instr); break;
        case 0x2c: swl<uint64_t, Dir::ll>(instr); break;
        case 0x2d: swl<uint64_t, Dir::rl>(instr); break;
        case 0x2e: swl<uint32_t, Dir::rl>(instr); break;
        case 0x2f: cache(instr); break;
        case 0x31: lwc1<uint32_t>(instr); break;
        case 0x35: lwc1<uint64_t>(instr); break;
        case 0x37: lw<uint64_t>(instr); break;
        case 0x39: swc1<uint32_t>(instr); break;
        case 0x3d: swc1<uint64_t>(instr); break;
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
