#ifndef MIPSJIT_H
#define MIPSJIT_H

#include "r4300.h"
#include "asmjit/asmjit.h"

using namespace asmjit;

enum class Device { r4300, rsp };

template <Device device>
struct MipsJit {
  enum class Dir { ll, rl, ra };
  enum class CC { gt, lt, ge, le, eq, ne };
  enum class Mul { frac, high, midm, midn, low };
  enum class Op {
    add, sub, mul, div, sqrt, abs, mov, neg,
    and_, or_, xor_, addc, subc
  };

  x86::Assembler as;
  Label end_label;
  static constexpr uint32_t block_end = 0x04ffffff;

  const bool is_rsp = (device == Device::rsp);
  const uint8_t hi = R4300::hi, lo = R4300::lo;
  const uint8_t dev_cop0 = (is_rsp ? RSP::dev_cop0 : R4300::dev_cop0);
  const uint8_t dev_cop1 = R4300::dev_cop1, dev_cop2 = RSP::dev_cop2;

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

  constexpr uint8_t x86_reg(uint8_t reg) {
    // allocate cop1 and cop2 xmm regs too?
    switch (reg) {
      case 2: return 12; // $v0 = r12
      case 4: return 13; // $a0 = r13
      case 5: return 14; // $a1 = r14
      case 6: return 15; // $a2 = r15
      case 18: return 3; // $s2 = rbx
      case 3: return 8;  // $v1 = r8
      case 7: return 9;  // $a3 = r9
      case 8: return 10; // $t0 = r10
      case 9: return 11; // $t1 = r11
      default: return 0;
    }
  }

  constexpr x86::Mem x86_spill(uint8_t reg) {
    return x86::dword_ptr(x86::rbp, reg << 3);
  }

  constexpr x86::Mem x86_spilld(uint8_t reg) {
    return x86::qword_ptr(x86::rbp, reg << 3);
  }

  constexpr x86::Mem x86_spillq(uint8_t reg) {
    return x86::dqword_ptr(x86::rbp, reg << 3);
  }

  void x86_load_acc() {
    // the only saved xmm registers are RSP accumulators
    if (!is_rsp) return;
    as.movdqa(x86::xmm13, x86_spillq(32 * 2 + dev_cop2));
    as.movdqa(x86::xmm14, x86_spillq(33 * 2 + dev_cop2));
    as.movdqa(x86::xmm15, x86_spillq(34 * 2 + dev_cop2));
  }

  void x86_store_acc() {
    if (!is_rsp) return;
    as.movdqa(x86_spillq(32 * 2 + dev_cop2), x86::xmm13);
    as.movdqa(x86_spillq(33 * 2 + dev_cop2), x86::xmm14);
    as.movdqa(x86_spillq(34 * 2 + dev_cop2), x86::xmm15);
  }

  void x86_load_all() {
    x86_load_acc();
    for (uint8_t i = 0x0; i < 0x20; ++i) {
      if (x86_reg(i) == 0) continue;
      as.push(x86::gpq(x86_reg(i)));
      as.mov(x86::gpq(x86_reg(i)), x86_spilld(i));
    }
  }

  void x86_store_all() {
    x86_store_acc();
    for (uint8_t i = 0x20; i != 0; --i) {
      if (x86_reg(i) == 0) continue;
      as.mov(x86_spilld(i), x86::gpq(x86_reg(i)));
      as.pop(x86::gpq(x86_reg(i)));
    }
  }

  void x86_store_caller() {
    x86_store_acc();
    for (uint8_t i = 0x20; i != 0; --i) {
      if (x86_reg(i) < 8 || x86_reg(i) >= 12) continue;
      as.push(x86::gpq(x86_reg(i)));
    }
  }

  void x86_load_caller() {
    x86_load_acc();
    for (uint8_t i = 0x0; i < 0x20; ++i) {
      if (x86_reg(i) < 8 || x86_reg(i) >= 12) continue;
      as.pop(x86::gpq(x86_reg(i)));
    }
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

  void elem_spec(uint8_t e) {
    // creates scalars from xmm15, according to element specifier
    uint64_t base; uint8_t offset;
    switch (__builtin_clz(e & 0xf)) {
      case 0: base = 0x0000000000000000; offset = (e & 0x7) * 2; break;
      case 1: base = 0x0808080800000000; offset = (e & 0x3) * 2; break;
      case 2: base = 0x0c0c080804040000; offset = (e & 0x1) * 2; break;
      default: return;
    }
    as.mov(x86::rax, base); as.movq(x86::xmm0, x86::rax);
    if (offset) {
      as.mov(x86::eax, offset); as.movq(x86::xmm2, x86::rax);
      as.paddb(x86::xmm0, x86::xmm2);
    }
    as.movdqa(x86::xmm1, x86::xmm0); as.pcmpeqd(x86::xmm2, x86::xmm2);
    as.psubb(x86::xmm1, x86::xmm2); as.punpcklbw(x86::xmm0, x86::xmm1);
    as.pshufb(x86::xmm15, x86::xmm0);
  }

  void update_acc() {
    // assuming old accumulator values stored in spillq
    // adds them to new accumulator values in xmm13-15
    as.pxor(x86::xmm1, x86::xmm1);
    // calc lower accumulator overflow mask
    as.movdqa(x86::xmm0, x86_spillq(34 * 2 + dev_cop2));
    as.paddusw(x86::xmm0, x86::xmm15);
    as.paddw(x86::xmm15, x86_spillq(34 * 2 + dev_cop2));
    // add carry to mid if overflow
    as.pcmpeqw(x86::xmm0, x86::xmm15);
    as.pcmpeqw(x86::xmm0, x86::xmm1);
    as.psubw(x86::xmm14, x86::xmm0);
    // calc middle accumulator overflow mask
    as.movdqa(x86::xmm0, x86_spillq(33 * 2 + dev_cop2));
    as.paddusw(x86::xmm0, x86::xmm14);
    as.paddw(x86::xmm14, x86_spillq(33 * 2 + dev_cop2));
    // add carry to high if overflow
    as.pcmpeqw(x86::xmm0, x86::xmm14);
    as.pcmpeqw(x86::xmm0, x86::xmm1);
    as.paddw(x86::xmm13, x86_spillq(32 * 2 + dev_cop2));
    as.psubw(x86::xmm13, x86::xmm0);
  }

  /* === Instruction Translations === */

  template <typename T>
  void lw(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (is_rsp) as.call(reinterpret_cast<uint64_t>(RSP::read<T>));
    else as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    x86_load_caller(); as.pop(x86::edi);
    if (rtx) as.mov(x86::gpq(rtx), x86::rax);
    else as.mov(x86_spilld(rt(instr)), x86::rax);
  }

  template <typename T>
  void sw(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    if (is_rsp) as.call(reinterpret_cast<uint64_t>(RSP::write<T>));
    else as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    x86_load_caller(); as.pop(x86::edi);
  }

  template <typename T, Dir dir>
  void lwl(uint32_t instr) {
    if (rt(instr) == 0) return;
    constexpr bool right = (dir != Dir::ll);
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // LWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
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
    if (is_rsp) as.call(reinterpret_cast<uint64_t>(RSP::read<T>));
    else as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    as.pop(x86::rcx);
    // apply mask depending on direction
    if (right) {
      as.mov(x86::rdx, x86::rcx); as.not_(x86::rcx);
      as.cmp(x86::rcx, 0); as.cmove(x86::rcx, x86::rdx);
    }
    as.and_(x86::rax, x86::rcx); as.not_(x86::rcx);
    x86_load_caller(); as.pop(x86::edi);
    if (rtx) {
      as.and_(x86::gpq(rtx), x86::rcx);
      as.or_(x86::gpq(rtx), x86::rax);
    } else {
      as.and_(x86_spilld(rt(instr)), x86::rcx);
      as.or_(x86_spilld(rt(instr)), x86::rax);
    }
    if (sizeof(T) < 8) to_eax(rt(instr)), from_eax(rt(instr));
  }

  template <typename T, Dir dir>
  void swl(uint32_t instr) {
    constexpr bool right = (dir != Dir::ll);
    uint8_t rtx = x86_reg(rt(instr)), rsx = x86_reg(rs(instr));
    // SWL BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
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
    if (is_rsp) as.call(reinterpret_cast<uint64_t>(RSP::read<T>));
    else as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
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
    if (is_rsp) as.call(reinterpret_cast<uint64_t>(RSP::write<T>));
    else as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    x86_load_caller(); as.pop(x86::edi);
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
    as.and_(x86_spilld(12 + dev_cop0), ~0x2);
    as.mov(x86::edi, x86_spill(14 + dev_cop0));
    as.jmp(end_label);
    return block_end;
  }

  void cache(uint32_t instr) {
    printf("CACHE instruction %x\n", instr);
  }

  uint32_t break_(uint32_t instr, uint32_t pc) {
    if (!is_rsp) invalid(instr);
    as.or_(x86_spilld(4 + dev_cop0), 0x3);
    as.mov(x86::edi, pc + 4);
    return block_end;
  }

  void mfc0(uint32_t instr) {
    printf("Read from COP0 reg %d\n", rd(instr));
    if (rt(instr) == 0) return;
    move(rt(instr), rd(instr) + dev_cop0);
  }

  void mtc0(uint32_t instr) {
    printf("Write to COP0 reg %d\n", rd(instr));
    if (is_rsp && rd(instr) == 4) {
      as.push(x86::edi); x86_store_caller(); as.mov(x86::edi, rt(instr));
      as.call(reinterpret_cast<uint64_t>(RSP::set_status));
      x86_load_caller(); as.pop(x86::edi); return;
    } else if (!is_rsp && rd(instr) == 11) {
      as.and_(x86_spilld(13 + dev_cop0), ~0x8000);
    }
    if (rt(instr) == 0) as.mov(x86_spilld(rd(instr) + dev_cop0), 0);
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
    } else if (operation == Op::sqrt) {
      as.sqrtss(x86::xmm0, x86::xmm0);
    } else if (operation == Op::abs) {
      as.xorps(x86::xmm1, x86::xmm1);
      as.subss(x86::xmm1, x86::xmm0);
      as.maxss(x86::xmm0, x86::xmm1);
    } else if (operation == Op::neg) {
      as.xorps(x86::xmm1, x86::xmm1);
      as.subss(x86::xmm1, x86::xmm0);
      as.movss(x86::xmm0, x86::xmm1);
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

  void mfc1(uint32_t instr) {
    printf("Read from COP1 reg %d\n", rd(instr));
    if (rt(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    as.movss(x86_spill(rt(instr)), x86::xmm0);
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86::gpd(rtx), x86_spill(rt(instr)));
    to_eax(rt(instr)); from_eax(rt(instr));
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
    // LWC1 BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(R4300::read<T>));
    x86_load_caller(); as.pop(x86::edi);
    as.mov(x86_spilld(rt(instr) + dev_cop1), x86::rax);
    if (rtx) as.movss(x86::xmm(rtx), x86_spilld(rt(instr) + dev_cop1));
  }

  template <typename T>
  void swc1(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1), rsx = x86_reg(rs(instr));
    // SWC1 BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    if (rtx) as.movss(x86_spilld(rt(instr) + dev_cop1), x86::xmm(rtx));
    as.mov(x86::rsi, x86_spilld(rt(instr) + dev_cop1));
    as.call(reinterpret_cast<uint64_t>(R4300::write<T>));
    x86_load_caller(); as.pop(x86::edi);
  }

  template <uint8_t round_mode>
  void round_w_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.roundss(x86::xmm0, x86::xmm(rdx), round_mode);
    else as.roundss(x86::xmm0, x86_spill(rd(instr) + dev_cop1), round_mode);
    as.cvtss2si(x86::eax, x86::xmm0);
    as.mov(x86_spill(sa(instr) + dev_cop1), x86::eax);
    uint8_t sax = x86_reg(sa(instr) + dev_cop1);
    if (sax) as.movss(x86::xmm(sax), x86_spill(sa(instr) + dev_cop1));
  }

  void cvt_fmt_w(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.movss(x86_spill(rd(instr) + dev_cop1), x86::xmm(rdx));
    as.cvtsi2ss(x86::xmm0, x86_spill(rd(instr) + dev_cop1));
    uint8_t sax = x86_reg(sa(instr) + dev_cop1);
    if (sax) as.movss(x86::xmm(sax), x86::xmm0);
    else as.movss(x86_spill(sa(instr) + dev_cop1), x86::xmm0);
  }

  void cvt_w_fmt(uint32_t instr) {
    uint8_t round_mode = 0; // read from FCSR
    uint8_t rdx = x86_reg(rd(instr) + dev_cop1);
    if (rdx) as.roundss(x86::xmm0, x86::xmm(rdx), round_mode);
    else as.roundss(x86::xmm0, x86_spill(rd(instr) + dev_cop1), round_mode);
    as.cvtss2si(x86::eax, x86::xmm0);
    as.mov(x86_spill(sa(instr) + dev_cop1), x86::eax);
    uint8_t sax = x86_reg(sa(instr) + dev_cop1);
    if (sax) as.movss(x86::xmm(sax), x86_spill(sa(instr) + dev_cop1));
  }

  void invalid(uint32_t instr) {
    printf("Unimplemented instruction %x\n", instr);
    exit(0);
  }

  template <Mul mul_type, bool accumulate>
  void vmudn(uint32_t instr) {
    // add rounding value
    if (mul_type == Mul::frac && !accumulate) {
      as.pcmpeqd(x86::xmm15, x86::xmm15); as.psllw(x86::xmm15, 15);
      as.pxor(x86::xmm14, x86::xmm14); as.pxor(x86::xmm13, x86::xmm13);
    }
    // save old accumulator values
    if (accumulate || mul_type == Mul::frac) x86_store_acc();
    // move vt into accumulator
    uint8_t rtx = x86_reg(rt(instr) + dev_cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + dev_cop2));
    elem_spec(rs(instr)); as.movdqa(x86::xmm14, x86::xmm15);
    // move vs into xmm temp register
    uint8_t rdx = x86_reg(rd(instr) * 2 + dev_cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + dev_cop2));
    if (mul_type == Mul::high) {
      // multiply signed vt by signed vs
      as.pmullw(x86::xmm15, x86::xmm0);
      as.pmulhw(x86::xmm14, x86::xmm0);
      // shift product up by 16 bits
      as.movdqa(x86::xmm13, x86::xmm14);
      as.movdqa(x86::xmm14, x86::xmm15);
      as.pxor(x86::xmm15, x86::xmm15);
    } else if (mul_type == Mul::low) {
      // multiply unsigned vt by unsigned vs
      as.pmulhuw(x86::xmm15, x86::xmm0);
      as.pxor(x86::xmm14, x86::xmm14);
      as.pxor(x86::xmm13, x86::xmm13);
    } else if (mul_type == Mul::frac) {
      // multiply signed vt by signed vs
      as.pmullw(x86::xmm15, x86::xmm0);
      as.pmulhw(x86::xmm14, x86::xmm0);
      // shift product up by 1 bit
      as.movdqa(x86::xmm13, x86::xmm14); as.psraw(x86::xmm13, 15);
      as.psllw(x86::xmm14, 1);
      as.movdqa(x86::xmm0, x86::xmm15); as.psrlw(x86::xmm0, 15);
      as.por(x86::xmm14, x86::xmm0);
      as.psllw(x86::xmm15, 1);
    } else {
      // save sign of vt, to fix unsigned multiply
      as.movdqa(x86::xmm1, x86::xmm15); as.psraw(x86::xmm1, 15);
      // multiply unsigned vt by unsigned vs
      as.pmullw(x86::xmm15, x86::xmm0);
      as.pmulhuw(x86::xmm14, x86::xmm0);
      // subtract vs where vt was negative
      as.pand(x86::xmm0, x86::xmm1);
      as.psubw(x86::xmm14, x86::xmm0);
      // sign extend to upper accumulator
      as.movdqa(x86::xmm13, x86::xmm14); as.psraw(x86::xmm13, 15);
    }
    if (accumulate || mul_type == Mul::frac) update_acc();
    // saturate signed value
    if (mul_type == Mul::midn || mul_type == Mul::low) {
      as.movdqa(x86::xmm0, x86::xmm15);
    } else {
      as.movdqa(x86::xmm0, x86::xmm14); as.movdqa(x86::xmm1, x86::xmm14);
      as.punpcklwd(x86::xmm0, x86::xmm13); as.punpckhwd(x86::xmm1, x86::xmm13);
      as.packssdw(x86::xmm0, x86::xmm1);
    }
    // move accumulator section into vd
    uint8_t sax = x86_reg(sa(instr) * 2 + dev_cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm0);
    else as.movdqa(x86_spillq(sa(instr) * 2 + dev_cop2), x86::xmm0);
  }

  template <Op operation>
  void vadd(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + dev_cop2));
    uint8_t rdx = x86_reg(rd(instr) * 2 + dev_cop2);
    elem_spec(rs(instr));
    if (operation == Op::abs) {
      if (rdx) as.psignw(x86::xmm15, x86::xmm(rdx));
      else as.psignw(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::add) { // doesn't handle VCO carry-in/clear
      if (rdx) as.paddsw(x86::xmm15, x86::xmm(rdx));
      else as.paddsw(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::addc) { // doesn't handle VCO carry-out
      if (rdx) as.paddw(x86::xmm15, x86::xmm(rdx));
      else as.paddw(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::sub) {
      if (rdx) as.psubsw(x86::xmm15, x86::xmm(rdx));
      else as.psubsw(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::subc) {
      if (rdx) as.psubw(x86::xmm15, x86::xmm(rdx));
      else as.psubw(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    }
    uint8_t sax = x86_reg(sa(instr) * 2 + dev_cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + dev_cop2), x86::xmm15);
  }

  template <bool eq, bool invert>
  void veq(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + dev_cop2));
    uint8_t rdx = x86_reg(rd(instr) * 2 + dev_cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + dev_cop2));
    as.movdqa(x86::xmm1, x86::xmm15);
    if (eq) as.pcmpeqw(x86::xmm15, x86::xmm0);
    else as.pcmpgtw(x86::xmm15, x86::xmm0);
    if (invert) as.pand(x86::xmm1, x86::xmm15), as.pandn(x86::xmm15, x86::xmm0);
    else as.pand(x86::xmm0, x86::xmm15), as.pandn(x86::xmm15, x86::xmm1);
    auto result = (invert ? x86::xmm1 : x86::xmm0); as.por(x86::xmm15, result);
    uint8_t sax = x86_reg(sa(instr) * 2 + dev_cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + dev_cop2), x86::xmm15);
  }

  template <Op operation, bool invert>
  void vand(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) * 2 + dev_cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + dev_cop2));
    uint8_t rdx = x86_reg(rd(instr) * 2 + dev_cop2);
    elem_spec(rs(instr));
    if (operation == Op::and_) {
      if (rdx) as.pand(x86::xmm15, x86::xmm(rdx));
      else as.pand(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::or_) {
      if (rdx) as.por(x86::xmm15, x86::xmm(rdx));
      else as.por(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    } else if (operation == Op::xor_) {
      if (rdx) as.pxor(x86::xmm15, x86::xmm(rdx));
      else as.pxor(x86::xmm15, x86_spillq(rd(instr) * 2 + dev_cop2));
    }
    if (invert) as.pcmpeqd(x86::xmm0, x86::xmm0), as.pxor(x86::xmm15, x86::xmm0);
    uint8_t sax = x86_reg(sa(instr) * 2 + dev_cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + dev_cop2), x86::xmm15);
  }

  void vsar(uint32_t instr) {
    uint8_t acc = (rs(instr) & 0x3) + 13;
    uint8_t sax = x86_reg(sa(instr) * 2 + dev_cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm(acc));
    else as.movdqa(x86_spillq(sa(instr) * 2 + dev_cop2), x86::xmm(acc));
  }

  template <typename T>
  void ldv(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1), rsx = x86_reg(rs(instr));
    // LDV BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    as.call(reinterpret_cast<uint64_t>(RSP::read<T>));
    x86_load_caller(); as.pop(x86::edi);
    auto result = (rtx ? x86::xmm(rtx) : x86::xmm0);
    if (!rtx) as.movdqa(x86::xmm0, x86_spillq(rt(instr) * 2 + dev_cop2));
    switch (sizeof(T)) {
      case 1: as.pinsrb(result, x86::rax, sa(instr)); break;
      case 2: as.pinsrw(result, x86::rax, sa(instr) >> 1); break;
      case 3: as.pinsrd(result, x86::rax, sa(instr) >> 2); break;
      case 4: as.pinsrq(result, x86::rax, sa(instr) >> 3); break;
    }
    if (!rtx) as.movdqa(x86_spillq(rt(instr) * 2 + dev_cop2), x86::xmm0);
  }

  template <typename T>
  void sdv(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) + dev_cop1), rsx = x86_reg(rs(instr));
    // SDV BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), imm(instr)));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, imm(instr)));
    }
    auto result = (rtx ? x86::xmm(rtx) : x86::xmm0);
    if (!rtx) as.movdqa(x86::xmm0, x86_spillq(rt(instr) * 2 + dev_cop2));
    switch (sizeof(T)) {
      case 1: as.pextrb(x86::rsi, result, sa(instr)); break;
      case 2: as.pextrw(x86::rsi, result, sa(instr) >> 1); break;
      case 3: as.pextrd(x86::rsi, result, sa(instr) >> 2); break;
      case 4: as.pextrq(x86::rsi, result, sa(instr) >> 3); break;
    }
    as.call(reinterpret_cast<uint64_t>(RSP::write<T>));
    x86_load_caller(); as.pop(x86::edi);
  }

  void lqv(uint32_t instr) {
    // only handles 128-bit aligned
    uint8_t rtx = x86_reg(rt(instr) * 2 + dev_cop2), rsx = x86_reg(rs(instr));
    uint32_t offset = (instr & 0x7) << 4;
    // LQV BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), offset));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, offset));
    }
    as.push(x86::edi);
    as.call(reinterpret_cast<uint64_t>(RSP::read<uint64_t>));
    as.mov(x86_spilld(rt(instr) * 2 + 1 + dev_cop2), x86::rax);
    as.pop(x86::edi); as.add(x86::edi, 8);
    as.call(reinterpret_cast<uint64_t>(RSP::read<uint64_t>));
    as.mov(x86_spilld(rt(instr) * 2 + dev_cop2), x86::rax);
    x86_load_caller(); as.pop(x86::edi);
    if (rtx) as.movdqa(x86::xmm(rtx), x86_spillq(rt(instr) * 2 + dev_cop2));
  }

  void sqv(uint32_t instr) {
    // only handles 128-bit aligned
    uint8_t rtx = x86_reg(rt(instr) * 2 + dev_cop2), rsx = x86_reg(rs(instr));
    uint32_t offset = (instr & 0x7) << 4;
    // SQV BASE(RS), RT, OFFSET(IMMEDIATE)
    as.push(x86::edi); x86_store_caller();
    if (rsx) as.lea(x86::edi, x86::dword_ptr(x86::gpd(rsx), offset));
    else {
      as.mov(x86::eax, x86_spill(rs(instr)));
      as.lea(x86::edi, x86::dword_ptr(x86::eax, offset));
    }
    if (rtx) as.movdqa(x86_spillq(rt(instr) * 2 + dev_cop2), x86::xmm(rtx));
    as.mov(x86::rsi, x86_spilld(rt(instr) * 2 + 1 + dev_cop2));
    as.push(x86::edi);
    as.call(reinterpret_cast<uint64_t>(RSP::write<uint64_t>));
    as.mov(x86::rsi, x86_spilld(rt(instr) * 2 + dev_cop2));
    as.pop(x86::edi); as.add(x86::edi, 8);
    as.call(reinterpret_cast<uint64_t>(RSP::write<uint64_t>));
    x86_load_caller(); as.pop(x86::edi);
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
      case 0x0d: next_pc = break_(instr, pc); break;
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
    // only handles single-precision float (no fixed points or doubles)
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
          case 0x04: add_fmt<Op::sqrt>(instr); break;
          case 0x05: add_fmt<Op::abs>(instr); break;
          case 0x06: add_fmt<Op::mov>(instr); break;
          case 0x07: add_fmt<Op::neg>(instr); break;
          case 0x0c: round_w_fmt<0>(instr); break;
          case 0x0d: round_w_fmt<3>(instr); break;
          case 0x0e: round_w_fmt<2>(instr); break;
          case 0x0f: round_w_fmt<1>(instr); break;
          case 0x20: cvt_fmt_w(instr); break;
          case 0x21: cvt_fmt_w(instr); break;
          case 0x24: cvt_w_fmt(instr); break;
          case 0x25: cvt_w_fmt(instr); break;
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

  uint32_t cop2(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    if (((instr >> 25) & 0x1) != 1) {
      printf("COP2 instruction %x\n", instr); return next_pc;
    }
    switch (instr & 0x3f) {
      case 0x00: vmudn<Mul::frac, false>(instr); break; // VMULF (buggy)
      case 0x01: vmudn<Mul::frac, false>(instr); break; // VMULU (buggy)
      case 0x04: vmudn<Mul::low, false>(instr); break; // VMUDL
      case 0x05: vmudn<Mul::midm, false>(instr); break; // VMUDM
      case 0x06: vmudn<Mul::midn, false>(instr); break; // VMUDN
      case 0x07: vmudn<Mul::high, false>(instr); break; // VMUDH
      case 0x08: vmudn<Mul::frac, true>(instr); break; // VMACF (buggy)
      case 0x09: vmudn<Mul::frac, true>(instr); break; // VMACU (buggy)
      case 0x0c: vmudn<Mul::low, true>(instr); break; // VMADL
      case 0x0d: vmudn<Mul::midm, true>(instr); break; // VMADM
      case 0x0e: vmudn<Mul::midn, true>(instr); break; // VMADN
      case 0x0f: vmudn<Mul::high, true>(instr); break; // VMADH
      case 0x10: vadd<Op::add>(instr); break;
      case 0x11: vadd<Op::sub>(instr); break;
      case 0x13: vadd<Op::abs>(instr); break;
      case 0x14: vadd<Op::addc>(instr); break;
      case 0x15: vadd<Op::subc>(instr); break;
      case 0x1d: vsar(instr); break;
      case 0x20: veq<false, false>(instr); break;
      case 0x21: veq<true, false>(instr); break;
      case 0x22: veq<true, true>(instr); break;
      case 0x23: veq<false, true>(instr); break;
      case 0x28: vand<Op::and_, false>(instr); break;
      case 0x29: vand<Op::and_, true>(instr); break;
      case 0x2a: vand<Op::or_, false>(instr); break;
      case 0x2b: vand<Op::or_, true>(instr); break;
      case 0x2c: vand<Op::xor_, false>(instr); break;
      case 0x2d: vand<Op::xor_, true>(instr); break;
      default: printf("COP2 instruction %x\n", instr); break;
    }
    return next_pc;
  }

  uint32_t jit_block() {
    as.push(x86::rbp);
    if (is_rsp) as.mov(x86::rbp, reinterpret_cast<uint64_t>(&RSP::reg_array));
    else as.mov(x86::rbp, reinterpret_cast<uint64_t>(&R4300::reg_array));
    x86_load_all();

    uint32_t cycles = 0, pc = (is_rsp ? RSP::pc : R4300::pc);
    end_label = as.newLabel();
    for (uint32_t next_pc = pc + 4; pc != block_end; ++cycles) {
      uint32_t instr = (is_rsp ? RSP::fetch(pc) : R4300::fetch(pc));
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
        case 0x12: next_pc = cop2(instr, pc); break;
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
        //case 0x2c: swl<uint64_t, Dir::ll>(instr); break;
        //case 0x2d: swl<uint64_t, Dir::rl>(instr); break;
        case 0x2e: swl<uint32_t, Dir::rl>(instr); break;
        case 0x2f: cache(instr); break;
        case 0x30: lw<int32_t>(instr); break; // LL
        case 0x31: lwc1<int32_t>(instr); break;
        case 0x32: lqv(instr); break;
        case 0x35: lwc1<uint64_t>(instr); break;
        case 0x37: lw<uint64_t>(instr); break;
        case 0x38: sw<uint32_t>(instr); break; // SC
        case 0x39: swc1<uint32_t>(instr); break;
        case 0x3a: sqv(instr); break;
        case 0x3d: swc1<uint64_t>(instr); break;
        case 0x3f: sw<uint64_t>(instr); break;
        default: invalid(instr); break;
      }
    }

    as.bind(end_label);
    x86_store_all();
    as.pop(x86::rbp);
    as.mov(x86::eax, x86::edi);
    as.ret();
    return cycles;
  }
};

#endif
