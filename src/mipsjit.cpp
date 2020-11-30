#include "asmjit/asmjit.h"
#include "nmulator.h"
#include "recip_rom.h"

using namespace asmjit;

struct MipsJit {
  MipsConfig &cfg;
  x86::Assembler as;

  Label end_label, exc_label;
  bool cop1_checked;
  static constexpr uint32_t block_end = 0x04ffffff;

  MipsJit(MipsConfig *cfg_, CodeHolder &code)
    : cfg(*cfg_), as(&code) {}

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

  constexpr x86::Mem x86_spillh(uint8_t reg) {
    uint32_t fpr = ((reg & ~0x1) + cfg.cop1) << 3;
    return x86::dword_ptr(x86::rbp, fpr + ((reg & 0x1) << 2));
  }

  enum COP2_Reg {
    ACC_HI = 64, ACC_MD = 66, ACC_LO = 68,
    VCO_LO = 70, VCO_HI = 72, VCC_LO = 74,
    VCC_HI = 76, VCE_LO = 78, VCE_HI = 80,
    DIV_IN = 82, DIV_OUT = 83, DIV_H = 84,
  };

  void x86_load_acc() {
    // the only saved xmm registers are RSP accumulators
    if (!cfg.cop2) return;
    as.movdqa(x86::xmm13, x86_spillq(ACC_HI + cfg.cop2));
    as.movdqa(x86::xmm14, x86_spillq(ACC_MD + cfg.cop2));
    as.movdqa(x86::xmm15, x86_spillq(ACC_LO + cfg.cop2));
  }

  void x86_store_acc() {
    if (!cfg.cop2) return;
    as.movdqa(x86_spillq(ACC_HI + cfg.cop2), x86::xmm13);
    as.movdqa(x86_spillq(ACC_MD + cfg.cop2), x86::xmm14);
    as.movdqa(x86_spillq(ACC_LO + cfg.cop2), x86::xmm15);
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

  // arg0 rcx preserved, arg1 rsi clobbered
  // rbx, rbp, rdi, rsp, r12-r15 preserved
  void x86_call(uint64_t func) {
#ifdef _WIN32
    as.push(x86::rcx), as.sub(x86::rsp, 32);
    as.mov(x86::rdx, x86::rsi), as.call(func);
    as.add(x86::rsp, 32), as.pop(x86::rcx);
#else
    as.push(x86::rdi), as.push(x86::rdi), as.push(x86::rcx);
    as.mov(x86::rdi, x86::rcx), as.call(func);
    as.pop(x86::rcx), as.pop(x86::rdi), as.pop(x86::rdi);
#endif
  }

  enum FN_Type {
    FN_READ, FN_WRITE, FN_TLB,
    FN_EXIT, FN_ENTER, FN_EXC
  };

  void emit_funcs() {
    // MMIO read handler
    cfg.fn[FN_READ] = (uint32_t)as.offset();
    x86_store_caller(), as.push(x86::rdi);
    x86_call((uint64_t)cfg.read);
    as.pop(x86::rdi), x86_load_caller();
    as.ret();

    // MMIO write handler
    cfg.fn[FN_WRITE] = (uint32_t)as.offset();
    x86_store_caller(), as.push(x86::rdi);
    x86_call((uint64_t)cfg.write);
    as.pop(x86::rdi), x86_load_caller();
    as.ret();

    // TLB miss handler
    // assumes not in branch delay, EXL = 0
    cfg.fn[FN_TLB] = (uint32_t)as.offset();
    // Cause
    uint32_t reg = (13 + cfg.cop0) << 3;
    as.mov(x86::byte_ptr(x86::rbp, reg), x86::eax);
    // Status, EPC
    as.or_(x86_spill(12 + cfg.cop0), 0x2);
    as.mov(x86_spill(14 + cfg.cop0), x86::edi);
    Label valid = as.newLabel();
    as.bt(x86::ecx, 29), as.mov(x86::edi, 0x80000000);
    as.jnc(valid), as.add(x86::edi, 0x180), as.bind(valid);
    // BadVAddr
    as.and_(x86::ecx, 0xfff), as.mov(x86::eax, x86::edx);
    as.shl(x86::eax, 12), as.or_(x86::ecx, x86::eax);
    as.mov(x86_spill(8 + cfg.cop0), x86::ecx);
    // Context
    as.mov(x86::eax, x86_spill(4 + cfg.cop0));
    as.and_(x86::eax, 0xff000000), as.and_(x86::edx, 0xffffe);
    as.shl(x86::edx, 3), as.or_(x86::eax, x86::edx);
    as.mov(x86_spill(4 + cfg.cop0), x86::eax);
    // EntryHi
    as.shl(x86::edx, 9);
    as.mov(x86_spill(10 + cfg.cop0), x86::edx);

    // JIT return handler
    cfg.fn[FN_EXIT] = (uint32_t)as.offset();
    x86_store_all(), as.mov(x86::eax, x86::edi);
#ifdef _WIN32
    as.pop(x86::rsi), as.pop(x86::rdi);
#endif
    as.pop(x86::rbp), as.ret();

    // JIT enter handler
    cfg.fn[FN_ENTER] = (uint32_t)as.offset();
    as.push(x86::rbp);
#ifdef _WIN32
    as.push(x86::rdi), as.push(x86::rsi);
    as.mov(x86::rdi, x86::rcx);
#endif
    as.mov(x86::rbp, (uint64_t)cfg.regs);
    x86_load_all(), as.jmp(x86::rdi);
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
    // 32 bit register to register
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

  void elem_spec(uint8_t e) {
    e &= 0xf; if (e <= 1) return; // element scalars from xmm15
    as.pshufb(x86::xmm15, x86_spillq(14 + e * 2 + cfg.pool));
  }

  void update_acc(bool high, bool frac) {
    // assuming old accumulator values stored in spillq
    // adds them to new accumulator values in xmm13-15
    as.pxor(x86::xmm1, x86::xmm1);
    if (high) as.movdqa(x86::xmm15, x86_spillq(ACC_LO + cfg.cop2));
    else {
      // calc lower accumulator overflow mask
      as.movdqa(x86::xmm0, x86_spillq(ACC_LO + cfg.cop2));
      as.paddusw(x86::xmm0, x86::xmm15);
      as.paddw(x86::xmm15, x86_spillq(ACC_LO + cfg.cop2));
      // add carry to mid if overflow
      as.pcmpeqw(x86::xmm0, x86::xmm15);
      as.pcmpeqw(x86::xmm0, x86::xmm1);
      as.psubw(x86::xmm14, x86::xmm0);
      as.movdqa(x86::xmm13, x86::xmm14);
      as.psraw(x86::xmm13, 15);
      if (frac) as.pxor(x86::xmm13, x86::xmm2);
    }
    // calc middle accumulator overflow mask
    as.movdqa(x86::xmm0, x86_spillq(ACC_MD + cfg.cop2));
    as.paddusw(x86::xmm0, x86::xmm14);
    as.paddw(x86::xmm14, x86_spillq(ACC_MD + cfg.cop2));
    // add carry to high if overflow
    as.pcmpeqw(x86::xmm0, x86::xmm14);
    as.pcmpeqw(x86::xmm0, x86::xmm1);
    as.paddw(x86::xmm13, x86_spillq(ACC_HI + cfg.cop2));
    as.psubw(x86::xmm13, x86::xmm0);
  }

  uint32_t check_breaks(uint32_t pc, uint32_t next_pc) {
    if (!cfg.stop_at(pc)) return next_pc;
    // ensure it actually returns to C++
    if (next_pc != block_end) as.mov(x86::edi, pc), as.jmp(end_label);
    return block_end;
  }

  /* === Instruction Translations === */

  inline void x86_paddr(Label miss) {
    as.mov(x86::edx, x86::ecx), as.shr(x86::edx, 12);
    as.mov(x86::rax, (uint64_t)cfg.pages);
    auto off = x86::dword_ptr(x86::rax, x86::rdx, 2);
    as.sub(x86::ecx, off), as.js(miss);
  }

  // fallback to MMIO or jump to TLB miss
  // paddr in ecx, vpage in edx, pc in edi
  inline void x86_miss(Label miss, uint32_t func) {
    Label after = as.newLabel();
    as.jmp(after), as.bind(miss);
    as.mov(x86::eax, func == FN_WRITE ? 12 : 8);
    as.bt(x86::ecx, 30), as.jc(cfg.fn[FN_TLB]);
    as.call(cfg.fn[func]), as.bind(after);
  }

  template <typename T>
  inline void x86_read(uint32_t pc) {
    bool early = pc != block_end && cfg.pages;
    if (early) as.mov(x86::edi, pc);
    // translate virtual address
    Label miss = as.newLabel();
    if (cfg.pages) x86_paddr(miss);
    else as.and_(x86::ecx, 0xfff);
    // loads unsigned rax from paddr ecx
    as.mov(x86::rax, (uint64_t)cfg.mem);
    if (sizeof(T) == 8) {
      as.movbe(x86::rax, x86::qword_ptr(x86::rax, x86::rcx));
    } else if (sizeof(T) == 4) {
      as.movbe(x86::eax, x86::dword_ptr(x86::rax, x86::rcx));
    } else if (sizeof(T) == 2) {
      as.movbe(x86::ax, x86::word_ptr(x86::rax, x86::rcx));
      as.movzx(x86::eax, x86::ax);
    } else if (sizeof(T) == 1) {
      as.movzx(x86::eax, x86::byte_ptr(x86::rax, x86::rcx));
    }
    // translation miss handler
    if (cfg.pages) x86_miss(miss, FN_READ);
  }

  template <typename T>
  inline void x86_read_s(uint32_t pc) {
    bool early = pc != block_end && cfg.pages;
    if (early) as.mov(x86::edi, pc);
    // translate virtual address
    Label miss = as.newLabel();
    if (cfg.pages) x86_paddr(miss);
    else as.and_(x86::ecx, 0xfff);
    // loads signed rax from paddr ecx
    as.mov(x86::rax, (uint64_t)cfg.mem);
    if (sizeof(T) == 8) {
      as.movbe(x86::rax, x86::qword_ptr(x86::rax, x86::rcx));
    } else if (sizeof(T) == 4) {
      as.movbe(x86::eax, x86::dword_ptr(x86::rax, x86::rcx));
      as.movsxd(x86::rax, x86::eax);
    } else if (sizeof(T) == 2) {
      as.movbe(x86::ax, x86::word_ptr(x86::rax, x86::rcx));
      as.movsx(x86::rax, x86::ax);
    } else if (sizeof(T) == 1) {
      as.movsx(x86::rax, x86::byte_ptr(x86::rax, x86::rcx));
    }
    // translation miss handler
    if (cfg.pages) x86_miss(miss, FN_READ);
  }

  template <typename T, bool phys=false>
  inline void x86_write(uint32_t pc) {
    bool early = pc != block_end && cfg.pages;
    if (early && !phys) as.mov(x86::edi, pc);
    // translate virtual address
    Label miss = as.newLabel();
    if (cfg.pages && !phys) x86_paddr(miss);
    if (cfg.pages && phys) as.cmp(x86::ecx, 0), as.js(miss);
    if (!cfg.pages) as.and_(x86::ecx, 0xfff);
    // writes rsi to paddr ecx
    as.mov(x86::rax, (uint64_t)cfg.mem);
    if (sizeof(T) == 8) {
      as.movbe(x86::qword_ptr(x86::rax, x86::rcx), x86::rsi);
    } else if (sizeof(T) == 4) {
      as.movbe(x86::dword_ptr(x86::rax, x86::rcx), x86::esi);
    } else if (sizeof(T) == 2) {
      as.movbe(x86::word_ptr(x86::rax, x86::rcx), x86::si);
    } else if (sizeof(T) == 1) {
      as.mov(x86::byte_ptr(x86::rax, x86::rcx), x86::sil);
    }
    // translation miss handler
    if (cfg.pages) x86_miss(miss, FN_WRITE);
  }

  template <typename T>
  void lw(uint32_t instr, uint32_t pc) {
    if (rt(instr) == 0) return;
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, imm(instr));
    // load byte-swapped data from address
    constexpr bool sign = T(-1) < T(0);
    sign ? x86_read_s<T>(pc) : x86_read<T>(pc);
    // move data into register
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86::gpq(rtx), x86::rax);
    else as.mov(x86_spilld(rt(instr)), x86::rax);
  }

  template <typename T>
  void sw(uint32_t instr, uint32_t pc) {
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, imm(instr));
    // store byte-swapped register
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86::rsi, x86::gpq(rtx));
    else as.mov(x86::rsi, x86_spilld(rt(instr)));
    x86_write<T>(pc);
  }

  template <typename T, bool right>
  void lwl(uint32_t instr, uint32_t pc) {
    if (rt(instr) == 0) return;
    // LWL BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = right * ((int32_t)sizeof(T) - 1);
    as.add(x86::ecx, imm(instr) - off);
    // load byte-swapped data from address
    constexpr bool sign = T(-1) < T(0);
    sign ? x86_read_s<T>(pc) : x86_read<T>(pc);
    if (right) as.dec(x86::ecx);
    as.and_(x86::ecx, (uint8_t)sizeof(T) - 1);
    uint8_t rtx = x86_reg(rt(instr));
    // mask according to alignment
    if (rtx) as.xor_(x86::rax, x86::gpq(rtx));
    else as.xor_(x86::rax, x86_spilld(rt(instr)));
    if (right) as.xor_(x86::ecx, 7); as.shl(x86::ecx, 3);
    right ? as.shl(x86::rax, x86::cl) : as.shr(x86::rax, x86::cl);
    right ? as.shr(x86::rax, x86::cl) : as.shl(x86::rax, x86::cl);
    if (rtx) as.xor_(x86::gpq(rtx), x86::rax);
    else as.xor_(x86_spilld(rt(instr)), x86::rax);
    if (sizeof(T) < 8) to_eax(rt(instr)), from_eax(rt(instr));
  }

  template <typename T, bool right>
  void swl(uint32_t instr, uint32_t pc) {
    // SWL BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = right * ((int32_t)sizeof(T) - 1);
    as.add(x86::ecx, imm(instr) - off);
    // load byte-swapped data from address
    constexpr bool sign = T(-1) < T(0);
    sign ? x86_read_s<T>(pc) : x86_read<T>(pc);
    as.mov(x86::rsi, x86::rax), as.mov(x86::edx, x86::ecx);
    if (right) as.dec(x86::ecx);
    as.and_(x86::ecx, (uint8_t)sizeof(T) - 1);
    uint8_t rtx = x86_reg(rt(instr));
    // mask according to alignment
    if (rtx) as.xor_(x86::rsi, x86::gpq(rtx));
    else as.xor_(x86::rsi, x86_spilld(rt(instr)));
    if (right) as.xor_(x86::ecx, 7); as.shl(x86::ecx, 3);
    right ? as.shl(x86::rsi, x86::cl) : as.shr(x86::rsi, x86::cl);
    right ? as.shr(x86::rsi, x86::cl) : as.shl(x86::rsi, x86::cl);
    as.xor_(x86::rsi, x86::rax), as.mov(x86::ecx, x86::edx);
    x86_write<T, true>(0);
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

  void daddiu(uint32_t instr) {
    if (rt(instr) == 0) return;
    uint8_t rtx = x86_reg(rt(instr));
    if (rs(instr) == 0) {
      // DADDIU RT, $0, IMMEDIATE
      if (rtx) as.mov(x86::gpq(rtx), imm(instr));
      else as.mov(x86_spilld(rt(instr)), imm(instr));
    } else {
      // DADDIU RT, RS, IMMEDIATE
      move(rt(instr), rs(instr));
      if (rtx) as.add(x86::gpq(rtx), imm(instr));
      else as.add(x86_spilld(rt(instr)), imm(instr));
    }
  }

  void addu(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rs(instr) == 0 && rt(instr) == 0) {
      // ADD RD, $0, $0
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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

  void daddu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // DADD RD, $0, $0
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // DADD RD, $0, RT
      move(rd(instr), rt(instr));
    } else if (rt(instr) == 0) {
      // DADD RD, RS, $0
      move(rd(instr), rs(instr));
    } else if (rd(instr) == rs(instr)) {
      // DADD RD, RD, RT
      uint8_t rtx = x86_reg(rt(instr));
      if (rdx) {
        if (rtx) as.add(x86::gpq(rdx), x86::gpq(rtx));
        else as.add(x86::gpq(rdx), x86_spilld(rt(instr)));
      } else {
        if (rtx) as.add(x86_spilld(rd(instr)), x86::gpq(rtx));
        else {
          as.mov(x86::rax, x86_spilld(rt(instr)));
          as.add(x86_spilld(rd(instr)), x86::rax);
        }
      }
    } else {
      // DADD RD, RS, RT
      move(rd(instr), rt(instr));
      uint8_t rsx = x86_reg(rs(instr));
      if (rdx) {
        if (rsx) as.add(x86::gpq(rdx), x86::gpq(rsx));
        else as.add(x86::gpq(rdx), x86_spilld(rs(instr)));
      } else {
        if (rsx) as.add(x86_spilld(rd(instr)), x86::gpq(rsx));
        else {
          as.mov(x86::rax, x86_spilld(rs(instr)));
          as.add(x86_spilld(rd(instr)), x86::rax);
        }
      }
    }
  }

  void subu(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rs(instr) == 0 && rt(instr) == 0) {
      // SUB RD, $0, $0
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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

  void dsubu(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rs(instr) == 0 && rt(instr) == 0) {
      // DSUB RD, $0, $0
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // DSUB RD, $0, RT
      move(rd(instr), rt(instr));
      if (rdx) as.neg(x86::gpq(rdx));
      else as.neg(x86_spilld(rd(instr)));
    } else if (rt(instr) == 0) {
      // DSUB RD, RS, $0
      move(rd(instr), rs(instr));
    } else {
      // DSUB RD, RS, RT
      uint8_t rsx = x86_reg(rs(instr)), rtx = x86_reg(rt(instr));
      if (rsx) as.mov(x86::rax, x86::gpq(rsx));
      else as.mov(x86::rax, x86_spilld(rs(instr)));
      if (rtx) as.sub(x86::rax, x86::gpq(rtx));
      else as.sub(x86::rax, x86_spilld(rt(instr)));
      if (rdx) as.mov(x86::gpq(rdx), x86::rax);
      else as.mov(x86_spilld(rd(instr)), x86::rax);
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
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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
      if (rtx) as.xor_(x86::gpd(rtx), x86::gpd(rtx));
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
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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

  enum SLL_Type { SLL, SRL, SRA };

  template <SLL_Type type, bool add32>
  void dsll(uint32_t instr) {
    if (rd(instr) == 0) return;
    uint8_t rdx = x86_reg(rd(instr));
    if (rt(instr) == 0) {
      // DSLL32 RD, $0, IMMEDIATE
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // DSLL32 RD, RT, IMMEDIATE
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) as.mov(x86::rax, x86::gpq(rtx));
      else as.mov(x86::rax, x86_spilld(rt(instr)));
      if (type == SLL) as.shl(x86::rax, sa(instr) + 32 * add32);
      if (type == SRL) as.shr(x86::rax, sa(instr) + 32 * add32);
      if (type == SRA) as.sar(x86::rax, sa(instr) + 32 * add32);
      if (rdx) as.mov(x86::gpq(rdx), x86::rax);
      else as.mov(x86_spilld(rd(instr)), x86::rax);
    }
  }

  template <SLL_Type type>
  void sll(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rt(instr) == 0) {
      // SLL RD, $0, IMMEDIATE
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else {
      // SLL RD, RT, IMMEDIATE
      to_eax(rt(instr));
      if (type == SLL) as.shl(x86::eax, sa(instr));
      if (type == SRL) as.shr(x86::eax, sa(instr));
      if (type == SRA) as.sar(x86::eax, sa(instr));
      from_eax(rd(instr));
    }
  }

  template <SLL_Type type>
  void sllv(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rt(instr) == 0) {
      // SLLV RD, $0, RS
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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
      if (type == SLL) as.shl(x86::eax, x86::cl);
      if (type == SRL) as.shr(x86::eax, x86::cl);
      if (type == SRA) as.sar(x86::eax, x86::cl);
      from_eax(rd(instr));
    }
  }

  template <SLL_Type type>
  void dsllv(uint32_t instr) {
    if (rd(instr) == 0) return;
    if (rt(instr) == 0) {
      // SLLV RD, $0, RS
      uint8_t rdx = x86_reg(rd(instr));
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
      else as.mov(x86_spilld(rd(instr)), 0);
    } else if (rs(instr) == 0) {
      // SLLV RD, RT, $0
      move(rd(instr), rt(instr));
    } else {
      // SLLV RD, RT, RS
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) as.mov(x86::rax, x86::gpq(rtx));
      else as.mov(x86::rax, x86_spilld(rt(instr)));
      uint8_t rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::rcx, x86::gpq(rsx));
      else as.mov(x86::rcx, x86_spilld(rs(instr)));
      if (type == SLL) as.shl(x86::rax, x86::cl);
      if (type == SRL) as.shr(x86::rax, x86::cl);
      if (type == SRA) as.sar(x86::rax, x86::cl);
      uint32_t rdx = x86_reg(rd(instr));
      if (rdx) as.mov(x86::gpq(rdx), x86::rax);
      else as.mov(x86_spilld(rd(instr)), x86::rax);
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
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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
      if (rdx) as.xor_(x86::gpd(rdx), x86::gpd(rdx));
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

  enum GPR_Reg { GPR_HI = 32, GPR_LO = 33 };

  template <bool sgn>
  void mult(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spilld(GPR_HI), 0);
      as.mov(x86_spilld(GPR_LO), 0);
    } else {
      to_eax(rs(instr));
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) {
        if (sgn) as.imul(x86::gpd(rtx));
        else as.mul(x86::gpd(rtx));
      } else {
        if (sgn) as.imul(x86_spill(rt(instr)));
        else as.mul(x86_spill(rt(instr)));
      }
      as.movsxd(x86::rax, x86::eax);
      as.movsxd(x86::rdx, x86::edx);
      as.mov(x86_spilld(GPR_HI), x86::rdx);
      as.mov(x86_spilld(GPR_LO), x86::rax);
    }
  }

  template <bool sgn>
  void dmult(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spilld(GPR_HI), 0);
      as.mov(x86_spilld(GPR_LO), 0);
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::rax, x86::gpq(rsx));
      else as.mov(x86::rax, x86_spilld(rs(instr)));
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) {
        if (sgn) as.imul(x86::gpq(rtx));
        else as.mul(x86::gpq(rtx));
      } else {
        if (sgn) as.imul(x86_spilld(rt(instr)));
        else as.mul(x86_spilld(rt(instr)));
      }
      as.mov(x86_spilld(GPR_HI), x86::rdx);
      as.mov(x86_spilld(GPR_LO), x86::rax);
    }
  }

  template <bool sgn>
  void div(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spilld(GPR_HI), 0);
      as.mov(x86_spilld(GPR_LO), 0);
    } else {
      to_eax(rs(instr));
      Label before_div = as.newLabel(), after_div = as.newLabel();
      as.cmp(x86::eax, 0x1 << 31), as.jne(before_div);
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) {
        as.cmp(x86::gpd(rtx), -1), as.je(after_div);
        as.bind(before_div);
        as.test(x86::gpd(rtx), x86::gpd(rtx)), as.je(after_div);
        if (sgn) as.cdq(), as.idiv(x86::gpd(rtx));
        else as.xor_(x86::edx, x86::edx), as.div(x86::gpd(rtx));
      } else {
        as.cmp(x86_spill(rt(instr)), -1), as.je(after_div);
        as.bind(before_div);
        as.cmp(x86_spill(rt(instr)), 0), as.je(after_div);
        if (sgn) as.cdq(), as.idiv(x86_spill(rt(instr)));
        else as.xor_(x86::edx, x86::edx), as.div(x86_spill(rt(instr)));
      }
      as.bind(after_div);
      as.movsxd(x86::rax, x86::eax);
      as.movsxd(x86::rdx, x86::edx);
      as.mov(x86_spilld(GPR_HI), x86::rdx);
      as.mov(x86_spilld(GPR_LO), x86::rax);
    }
  }

  template <bool sgn>
  void ddiv(uint32_t instr) {
    if (rs(instr) == 0 || rt(instr) == 0) {
      as.mov(x86_spilld(GPR_HI), 0);
      as.mov(x86_spilld(GPR_LO), 0);
    } else {
      uint32_t rsx = x86_reg(rs(instr));
      if (rsx) as.mov(x86::rax, x86::gpq(rsx));
      else as.mov(x86::rax, x86_spilld(rs(instr)));
      Label before_div = as.newLabel();
      Label after_div = as.newLabel();
      as.mov(x86::rdx, 0x80), as.shl(x86::rdx, 56);
      as.cmp(x86::rax, x86::rdx), as.jne(before_div);
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) {
        as.cmp(x86::gpq(rtx), -1), as.je(after_div);
        as.bind(before_div);
        as.test(x86::gpq(rtx), x86::gpq(rtx)), as.je(after_div);
        if (sgn) as.cqo(), as.idiv(x86::gpq(rtx));
        else as.xor_(x86::edx, x86::edx), as.div(x86::gpq(rtx));
      } else {
        as.cmp(x86_spilld(rt(instr)), -1), as.je(after_div);
        as.bind(before_div);
        as.cmp(x86_spilld(rt(instr)), 0), as.je(after_div);
        if (sgn) as.cqo(), as.idiv(x86_spilld(rt(instr)));
        else as.xor_(x86::edx, x86::edx), as.div(x86_spilld(rt(instr)));
      }
      as.bind(after_div);
      as.mov(x86_spilld(GPR_HI), x86::rdx);
      as.mov(x86_spilld(GPR_LO), x86::rax);
    }
  }

  template <GPR_Reg reg>
  void mfhi(uint32_t instr) {
    if (rd(instr) == 0) return;
    move(rd(instr), reg);
  }

  template <GPR_Reg reg>
  void mthi(uint32_t instr) {
    if (rd(instr) == 0) {
      uint8_t regx = x86_reg(reg);
      if (regx) as.xor_(x86::gpd(regx), x86::gpd(regx));
      else as.mov(x86_spilld(reg), 0);
    }
    move(reg, rd(instr));
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

  enum BLTZ_Type { BLTZ, BGTZ, BLEZ, BGEZ };

  template <BLTZ_Type type, bool link>
  uint32_t bltz(uint32_t instr, uint32_t pc) {
    if (link) as.mov(x86::eax, pc + 4), from_eax(31);
    if (rs(instr) == 0) {
      if (type == BLTZ || type == BGTZ) return pc;
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    compare(rs(instr), 0);
    as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    if (type == BLTZ) as.cmovl(x86::edi, x86::eax);
    if (type == BGTZ) as.cmovg(x86::edi, x86::eax);
    if (type == BLEZ) as.cmovle(x86::edi, x86::eax);
    if (type == BGEZ) as.cmovge(x86::edi, x86::eax);
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

  template <BLTZ_Type type, bool link>
  uint32_t bltzl(uint32_t instr, uint32_t pc) {
    if (link) as.mov(x86::eax, pc + 4), from_eax(31);
    if (rs(instr) == 0) {
      if (type == BLTZ || type == BGTZ) return pc + 4;
      as.mov(x86::edi, pc + (imm(instr) << 2));
      return block_end;
    }
    compare(rs(instr), 0);
    as.mov(x86::edi, pc + 4);
    if (type == BLTZ) as.jge(end_label);
    if (type == BGTZ) as.jle(end_label);
    if (type == BLEZ) as.jg(end_label);
    if (type == BGEZ) as.jl(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  uint32_t break_(uint32_t pc) {
    as.or_(x86_spilld(4 + cfg.cop0), 0x3);
    as.mov(x86::edi, pc + 4);
    return block_end;
  }

  void tlbr() {
    constexpr uint8_t entry_reg[4] = {5, 10, 2, 3};
    as.mov(x86::rsi, (uint64_t)cfg.tlb);
    as.mov(x86::eax, x86_spill(cfg.cop0));
    as.shl(x86::eax, 4), as.add(x86::rsi, x86::rax);
    for (uint8_t i = 0; i < 4; ++i) {
      as.mov(x86::eax, x86::dword_ptr(x86::rsi, i * 4));
      as.mov(x86_spill(entry_reg[i] + cfg.cop0), x86::eax);
    }
  }

  template <bool rand>
  void tlbwi() {
    x86_store_caller(), as.mov(x86::ecx, x86_spill(rand + cfg.cop0));
    if (rand) {
      as.mov(x86::ecx, 32), as.sub(x86::ecx, x86_spill(6 + cfg.cop0));
      as.crc32(x86::eax, x86_spill(9 + cfg.cop0)), as.mul(x86::ecx);
      as.mov(x86::ecx, x86::edx), as.xor_(x86::ecx, 0x1f);
    }
    as.and_(x86::ecx, 0x1f), x86_call((uint64_t)cfg.tlbwi);
    x86_load_caller();
  }

  void tlbp() {
    as.mov(x86::rsi, (uint64_t)cfg.tlb);
    as.mov(x86::ecx, x86_spill(10 + cfg.cop0)), as.xor_(x86::eax, x86::eax);
    as.mov(x86::edx, x86_spill(cfg.cop0)), as.or_(x86::edx, 0x80000000);
    for (uint8_t i = 0; i < 32; ++i) {
      as.cmp(x86::ecx, x86::dword_ptr(x86::rsi, i * 16 + 4));
      as.cmove(x86::edx, x86::eax), as.inc(x86::eax);
    }
    as.mov(x86_spill(cfg.cop0), x86::edx);
  }

  uint32_t eret() {
    as.and_(x86_spill(12 + cfg.cop0), ~0x2);
    as.mov(x86::edi, x86_spill(14 + cfg.cop0));
    as.jmp(end_label);
    return block_end;
  }

  void mfc0(uint32_t instr) {
    if (rt(instr) == 0) return;
    move(rt(instr), rd(instr) + cfg.cop0);
    if (cfg.is_rsp && rt(instr) == 7) {
      as.mov(x86_spilld(rd(instr) + cfg.cop0), 1);
    }
  }

  void mtc0(uint32_t instr) {
    bool call = cfg.mtc0_mask & (1 << rd(instr));
    if (!call) return move(rd(instr) + cfg.cop0, rt(instr));
    // read register value parameter
    if (rt(instr) != 0) {
      uint32_t rtx = x86_reg(rt(instr));
      if (rtx) as.movsxd(x86::rsi, x86::gpd(rtx));
      else as.movsxd(x86::rsi, x86_spill(rt(instr)));
    } else as.xor_(x86::esi, x86::esi);
    // pass cop0 index to callback
    x86_store_caller(), as.mov(x86::ecx, rd(instr));
    x86_call((uint64_t)cfg.mtc0), x86_load_caller();
  }

  enum ADD_FMT_Type {
    ADD_FMT, SUB_FMT, MUL_FMT, DIV_FMT,
    SQR_FMT, ABS_FMT, MOV_FMT, NEG_FMT
  };

  template <ADD_FMT_Type type>
  void add_fmt_s(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
    else as.movss(x86::xmm0, x86_spill(rd(instr) + cfg.cop1));
    uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
    if (type == ADD_FMT) {
      if (rtx) as.addss(x86::xmm0, x86::xmm(rtx));
      else as.addss(x86::xmm0, x86_spill(rt(instr) + cfg.cop1));
    } else if (type == SUB_FMT) {
      if (rtx) as.subss(x86::xmm0, x86::xmm(rtx));
      else as.subss(x86::xmm0, x86_spill(rt(instr) + cfg.cop1));
    } else if (type == MUL_FMT) {
      if (rtx) as.mulss(x86::xmm0, x86::xmm(rtx));
      else as.mulss(x86::xmm0, x86_spill(rt(instr) + cfg.cop1));
    } else if (type == DIV_FMT) {
      if (rtx) as.divss(x86::xmm0, x86::xmm(rtx));
      else as.divss(x86::xmm0, x86_spill(rt(instr) + cfg.cop1));
    } else if (type == SQR_FMT) {
      as.sqrtss(x86::xmm0, x86::xmm0);
    } else if (type == ABS_FMT) {
      as.xorps(x86::xmm1, x86::xmm1);
      as.subss(x86::xmm1, x86::xmm0);
      as.maxss(x86::xmm0, x86::xmm1);
    } else if (type == NEG_FMT) {
      as.xorps(x86::xmm1, x86::xmm1);
      as.subss(x86::xmm1, x86::xmm0);
      as.movss(x86::xmm0, x86::xmm1);
    }
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movss(x86::xmm(sax), x86::xmm0);
    else as.movss(x86_spill(sa(instr) + cfg.cop1), x86::xmm0);
  }

  template <ADD_FMT_Type type>
  void add_fmt_d(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rdx) as.movsd(x86::xmm0, x86::xmm(rdx));
    else as.movsd(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1));
    uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
    if (type == ADD_FMT) {
      if (rtx) as.addsd(x86::xmm0, x86::xmm(rtx));
      else as.addsd(x86::xmm0, x86_spilld(rt(instr) + cfg.cop1));
    } else if (type == SUB_FMT) {
      if (rtx) as.subsd(x86::xmm0, x86::xmm(rtx));
      else as.subsd(x86::xmm0, x86_spilld(rt(instr) + cfg.cop1));
    } else if (type == MUL_FMT) {
      if (rtx) as.mulsd(x86::xmm0, x86::xmm(rtx));
      else as.mulsd(x86::xmm0, x86_spilld(rt(instr) + cfg.cop1));
    } else if (type == DIV_FMT) {
      if (rtx) as.divsd(x86::xmm0, x86::xmm(rtx));
      else as.divsd(x86::xmm0, x86_spilld(rt(instr) + cfg.cop1));
    } else if (type == SQR_FMT) {
      as.sqrtsd(x86::xmm0, x86::xmm0);
    } else if (type == ABS_FMT) {
      as.xorpd(x86::xmm1, x86::xmm1);
      as.subsd(x86::xmm1, x86::xmm0);
      as.maxsd(x86::xmm0, x86::xmm1);
    } else if (type == NEG_FMT) {
      as.xorpd(x86::xmm1, x86::xmm1);
      as.subsd(x86::xmm1, x86::xmm0);
      as.movsd(x86::xmm0, x86::xmm1);
    }
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movsd(x86::xmm(sax), x86::xmm0);
    else as.movsd(x86_spilld(sa(instr) + cfg.cop1), x86::xmm0);
  }

  template <ADD_FMT_Type type>
  void add_fmt(uint32_t instr) {
    if (rs(instr) == 16)
      return add_fmt_s<type>(instr);
    else if (rs(instr) == 17)
      return add_fmt_d<type>(instr);
    else invalid(instr);
  }

  void c_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rs(instr) == 16) {
      if (rdx) as.movss(x86::xmm0, x86::xmm(rdx));
      else as.movss(x86::xmm0, x86_spill(rd(instr) + cfg.cop1));
      uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
      if (rtx) as.ucomiss(x86::xmm0, x86::xmm(rtx));
      else as.ucomiss(x86::xmm0, x86_spill(rt(instr) + cfg.cop1));
    } else if (rs(instr) == 17) {
      if (rdx) as.movsd(x86::xmm0, x86::xmm(rdx));
      else as.movsd(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1));
      uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
      if (rtx) as.ucomisd(x86::xmm0, x86::xmm(rtx));
      else as.ucomisd(x86::xmm0, x86_spilld(rt(instr) + cfg.cop1));
    } else invalid(instr);
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
    as.and_(x86::eax, mask); as.and_(x86_spill(32 + cfg.cop1), ~mask);
    as.or_(x86_spill(32 + cfg.cop1), x86::eax);
  }

  template <bool cond>
  uint32_t bc1t(uint32_t instr, uint32_t pc) {
    uint8_t cc = rt(instr) >> 2;
    uint32_t mask = (cc ? (0x1000 << cc) : 0x800);
    as.mov(x86::eax, x86_spill(32 + cfg.cop1));
    as.and_(x86::eax, mask), as.mov(x86::edi, pc + 4);
    as.mov(x86::eax, pc + (imm(instr) << 2));
    if (cond) as.cmovnz(x86::edi, x86::eax);
    else as.cmovz(x86::edi, x86::eax);
    return block_end;
  }

  template <bool cond>
  uint32_t bc1tl(uint32_t instr, uint32_t pc) {
    uint8_t cc = rt(instr) >> 2;
    uint32_t mask = (cc ? (0x1000 << cc) : 0x800);
    as.mov(x86::eax, x86_spill(32 + cfg.cop1));
    as.and_(x86::eax, mask), as.mov(x86::edi, pc + 4);
    if (cond) as.jz(end_label);
    else as.jnz(end_label);
    as.mov(x86::edi, pc + (imm(instr) << 2));
    return block_end;
  }

  template <bool dword>
  void mfc1(uint32_t instr) {
    if (rt(instr) == 0) return;
    bool fr = cfg.regs[12 + cfg.cop0] & 0x4000000;
    if (!dword && !fr) {
      uint8_t rdx = x86_reg((rd(instr) & ~0x1) + cfg.cop1);
      if (rdx) as.insertps(x86::xmm0, x86::xmm(rdx), (rd(instr) & 0x1) << 6);
      else as.movss(x86::xmm0, x86_spillh(rd(instr)));
    } else {
      uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
      if (rdx) as.movsd(x86::xmm0, x86::xmm(rdx));
      else as.movsd(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1));
    }
    as.movsd(x86_spilld(rt(instr)), x86::xmm0);
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86::gpq(rtx), x86_spilld(rt(instr)));
    if (!dword) to_eax(rt(instr)), from_eax(rt(instr));
  }

  template <bool dword>
  void mtc1(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.mov(x86_spilld(rt(instr)), x86::gpq(rtx));
    as.movsd(x86::xmm0, x86_spilld(rt(instr)));
    bool fr = cfg.regs[12 + cfg.cop0] & 0x4000000;
    if (!dword && !fr) {
      uint8_t rdx = x86_reg((rd(instr) & ~0x1) + cfg.cop1);
      if (rdx) as.insertps(x86::xmm(rdx), x86::xmm0, (rd(instr) & 0x1) << 4);
      else as.movss(x86_spillh(rd(instr)), x86::xmm0);
    } else {
      uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
      if (rdx) as.movsd(x86::xmm(rdx), x86::xmm0);
      else as.movsd(x86_spilld(rd(instr) + cfg.cop1), x86::xmm0);
    }
  }

  void cfc1(uint32_t instr) {
    if (rt(instr) == 0) return;
    if (rd(instr) == 31) move(rt(instr), 32 + cfg.cop1);
  }

  void ctc1(uint32_t instr) {
    if (rd(instr) != 31) return;
    if (rt(instr) == 0) as.mov(x86_spilld(32 + cfg.cop1), 0);
    else move(32 + cfg.cop1, rt(instr));
  }

  template <typename T>
  void lwc1(uint32_t instr, uint32_t pc) {
    if (!cop1_checked) check_cop1(pc);
    // LW BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, imm(instr));
    // load byte-swapped data from address
    constexpr bool sign = T(-1) < T(0);
    sign ? x86_read_s<T>(pc) : x86_read<T>(pc);
    // move data into register
    bool fr = cfg.regs[12 + cfg.cop0] & 0x4000000;
    if (sizeof(T) < 8 && !fr) {
      uint8_t rtx = x86_reg((rt(instr) & ~0x1) + cfg.cop1);
      as.mov(x86_spillh(rt(instr)), x86::eax);
      if (rtx) as.pinsrd(x86::xmm(rtx), x86_spillh(rt(instr)), rt(instr) & 0x1);
    } else {
      uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
      as.mov(x86_spilld(rt(instr) + cfg.cop1), x86::rax);
      if (rtx) as.movsd(x86::xmm(rtx), x86_spilld(rt(instr) + cfg.cop1));
    }
  }

  template <typename T>
  void swc1(uint32_t instr, uint32_t pc) {
    if (!cop1_checked) check_cop1(pc);
    // SW BASE(RS), RT, OFFSET(IMMEDIATE)
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, imm(instr));
    // store byte-swapped register
    bool fr = cfg.regs[12 + cfg.cop0] & 0x4000000;
    if (sizeof(T) < 8 && !fr) {
      uint8_t rtx = x86_reg((rt(instr) & ~0x1) + cfg.cop1);
      if (rtx) as.pextrd(x86_spillh(rt(instr)), x86::xmm(rtx), rt(instr) & 0x1);
      as.mov(x86::rsi, x86_spillh(rt(instr)));
    } else {
      uint8_t rtx = x86_reg(rt(instr) + cfg.cop1);
      if (rtx) as.movsd(x86_spilld(rt(instr) + cfg.cop1), x86::xmm(rtx));
      as.mov(x86::rsi, x86_spilld(rt(instr) + cfg.cop1));
    }
    x86_write<T>(pc);
  }

  enum ROUND_FMT_Type { RN_FMT, RM_FMT, RP_FMT, RZ_FMT };

  template <bool dword, ROUND_FMT_Type type>
  void round_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rs(instr) == 16) {
      if (rdx) as.roundss(x86::xmm0, x86::xmm(rdx), type);
      else as.roundss(x86::xmm0, x86_spill(rd(instr) + cfg.cop1), type);
      as.cvtss2si(x86::rax, x86::xmm0);
    } else if (rs(instr) == 17) {
      if (rdx) as.roundsd(x86::xmm0, x86::xmm(rdx), type);
      else as.roundsd(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1), type);
      as.cvtsd2si(x86::rax, x86::xmm0);
    } else invalid(instr);
    as.mov(x86_spilld(sa(instr) + cfg.cop1), (dword ? x86::rax : x86::eax));
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movsd(x86::xmm(sax), x86_spilld(sa(instr) + cfg.cop1));
  }

  void cvt_s_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rdx) as.movsd(x86_spilld(rd(instr) + cfg.cop1), x86::xmm(rdx));
    if (rs(instr) == 17) as.cvtsd2ss(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1));
    else as.cvtsi2ss(x86::xmm0, x86_spill(rd(instr) + cfg.cop1));
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movss(x86::xmm(sax), x86::xmm0);
    else as.movss(x86_spill(sa(instr) + cfg.cop1), x86::xmm0);
  }

  void cvt_d_fmt(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rdx) as.movsd(x86_spilld(rd(instr) + cfg.cop1), x86::xmm(rdx));
    if (rs(instr) == 16) as.cvtss2sd(x86::xmm0, x86_spill(rd(instr) + cfg.cop1));
    else as.cvtsi2sd(x86::xmm0, x86_spill(rd(instr) + cfg.cop1));
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movsd(x86::xmm(sax), x86::xmm0);
    else as.movsd(x86_spilld(sa(instr) + cfg.cop1), x86::xmm0);
  }

  template <bool dword>
  void cvt_w_fmt(uint32_t instr) {
    uint8_t round_mode = 0; // read from FCSR
    uint8_t rdx = x86_reg(rd(instr) + cfg.cop1);
    if (rs(instr) == 16) {
      if (rdx) as.roundss(x86::xmm0, x86::xmm(rdx), round_mode);
      else as.roundss(x86::xmm0, x86_spill(rd(instr) + cfg.cop1), round_mode);
      as.cvtss2si(x86::eax, x86::xmm0);
    } else if (rs(instr) == 17) {
      if (rdx) as.roundsd(x86::xmm0, x86::xmm(rdx), round_mode);
      else as.roundsd(x86::xmm0, x86_spilld(rd(instr) + cfg.cop1), round_mode);
      as.cvtsd2si(x86::rax, x86::xmm0);
    } else invalid(instr);
    as.mov(x86_spilld(sa(instr) + cfg.cop1), (dword ? x86::rax : x86::eax));
    uint8_t sax = x86_reg(sa(instr) + cfg.cop1);
    if (sax) as.movsd(x86::xmm(sax), x86_spilld(sa(instr) + cfg.cop1));
  }

  void invalid(uint32_t instr) {
    const char *name = cfg.is_rsp ? "RSP" : "R4300";
    printf("Invalid %s instruction %x\n", name, instr), exit(1);
  }

  enum VMULF_Type { VMULF, VMULU };

  template <VMULF_Type type, bool acc>
  void vmulf(uint32_t instr) {
    if (acc) x86_store_acc();
    // move vt into accumulator
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)), as.movdqa(x86::xmm14, x86::xmm15);
    // move vs into temp register
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    // multiply signed vt by signed vs
    as.movdqa(x86::xmm1, x86::xmm15);
    as.pmullw(x86::xmm15, x86::xmm0);
    as.pmulhw(x86::xmm14, x86::xmm0);
    // shift product up by 1 bit
    as.movdqa(x86::xmm0, x86::xmm15); as.psrlw(x86::xmm0, 15);
    as.paddw(x86::xmm14, x86::xmm14); as.por(x86::xmm14, x86::xmm0);
    as.paddw(x86::xmm15, x86::xmm15);
    if (!acc) {
      // add rounding value
      as.movdqa(x86::xmm0, x86::xmm15); as.psrlw(x86::xmm0, 15);
      as.paddw(x86::xmm14, x86::xmm0); as.pcmpeqd(x86::xmm0, x86::xmm0);
      as.psllw(x86::xmm0, 15); as.pxor(x86::xmm15, x86::xmm0);
      // handle overflow case
      as.pcmpeqw(x86::xmm1, x86::xmm0);
      if (rdx) as.pcmpeqw(x86::xmm0, x86::xmm(rdx));
      else as.pcmpeqw(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
      as.pand(x86::xmm0, x86::xmm1); as.movdqa(x86::xmm13, x86::xmm14);
      as.psraw(x86::xmm13, 15); as.pxor(x86::xmm13, x86::xmm0);
    } else {
      // handle overflow case
      as.pcmpeqd(x86::xmm2, x86::xmm2); as.psllw(x86::xmm2, 15);
      as.pcmpeqw(x86::xmm1, x86::xmm2);
      if (rdx) as.pcmpeqw(x86::xmm2, x86::xmm(rdx));
      else as.pcmpeqw(x86::xmm2, x86_spillq(rd(instr) * 2 + cfg.cop2));
      as.pand(x86::xmm2, x86::xmm1); update_acc(false, true);
    }
    // saturate signed value
    as.movdqa(x86::xmm0, x86::xmm14); as.punpcklwd(x86::xmm0, x86::xmm13);
    as.movdqa(x86::xmm1, x86::xmm14); as.punpckhwd(x86::xmm1, x86::xmm13);
    as.packssdw(x86::xmm0, x86::xmm1);
    // handle unsigned clamping
    if (type == VMULU) {
      as.movdqa(x86::xmm1, x86::xmm0); as.movdqa(x86::xmm2, x86::xmm0);
      as.pcmpgtw(x86::xmm2, x86::xmm14); as.psraw(x86::xmm0, 15);
      as.pandn(x86::xmm0, x86::xmm1); as.por(x86::xmm0, x86::xmm2);
    }
    // move accumulator section into vd
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm0);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm0);
  }

  enum VMUDN_Type { VMUDL, VMUDM, VMUDN, VMUDH };

  template <VMUDN_Type type, bool acc>
  void vmudn(uint32_t instr) {
    if (acc) x86_store_acc();
    // move vt into accumulator
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)); as.movdqa(x86::xmm14, x86::xmm15);
    // move vs into xmm temp register
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    if (type == VMUDH) {
      // multiply signed vt by signed vs
      as.movdqa(x86::xmm13, x86::xmm14);
      as.pmullw(x86::xmm14, x86::xmm0);
      as.pmulhw(x86::xmm13, x86::xmm0);
      if (!acc) as.pxor(x86::xmm15, x86::xmm15);
    } else if (type == VMUDL) {
      // multiply unsigned vt by unsigned vs
      as.pmulhuw(x86::xmm15, x86::xmm0);
      as.pxor(x86::xmm14, x86::xmm14);
      if (!acc) as.pxor(x86::xmm13, x86::xmm13);
    } else if (type == VMUDM || type == VMUDN) {
      // multiply unsigned vt by unsigned vs
      as.movdqa(x86::xmm1, x86::xmm15);
      as.pmullw(x86::xmm15, x86::xmm0);
      as.pmulhuw(x86::xmm14, x86::xmm0);
      // subtract vs where vt was negative
      if (type == VMUDN) as.psraw(x86::xmm1, 15);
      else as.psraw(x86::xmm0, 15);
      as.pand(x86::xmm1, x86::xmm0);
      as.psubw(x86::xmm14, x86::xmm1);
      if (!acc) as.movdqa(x86::xmm13, x86::xmm14);
      if (!acc) as.psraw(x86::xmm13, 15);
    }
    if (acc) update_acc(type == VMUDH, false);
    if (type == VMUDN || type == VMUDL) {
      // saturate unsigned value
      as.movdqa(x86::xmm0, x86::xmm14); as.psraw(x86::xmm0, 15);
      as.movdqa(x86::xmm1, x86::xmm13); as.psraw(x86::xmm1, 15);
      as.pxor(x86::xmm2, x86::xmm2); as.pcmpeqw(x86::xmm2, x86::xmm1);
      as.pcmpeqw(x86::xmm0, x86::xmm13); as.movdqa(x86::xmm1, x86::xmm15);
      as.pand(x86::xmm1, x86::xmm0); as.pandn(x86::xmm0, x86::xmm2);
      as.por(x86::xmm0, x86::xmm1);
    } else if (type == VMUDH || type == VMUDM) {
      // saturate signed value
      as.movdqa(x86::xmm0, x86::xmm14); as.punpcklwd(x86::xmm0, x86::xmm13);
      as.movdqa(x86::xmm1, x86::xmm14); as.punpckhwd(x86::xmm1, x86::xmm13);
      as.packssdw(x86::xmm0, x86::xmm1);
    }
    // move accumulator section into vd
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm0);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm0);
  }

  enum VADD_Type { VABS, VADD, VADDC, VSUB, VSUBC };

  template <VADD_Type type>
  void vadd(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr));
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rtx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    printf("COP2 ADD of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    if (type == VABS) {
      as.psignw(x86::xmm15, x86::xmm0);
    } else if (type == VADD) {
      as.movdqa(x86::xmm1, x86::xmm15); as.pmaxsw(x86::xmm15, x86::xmm0);
      as.pminsw(x86::xmm0, x86::xmm1); as.movdqa(x86::xmm1, x86::xmm15);
      as.psubw(x86::xmm1, x86_spillq(VCO_LO + cfg.cop2));
      as.paddw(x86::xmm1, x86::xmm0);
      as.psubsw(x86::xmm0, x86_spillq(VCO_LO + cfg.cop2));
      as.paddsw(x86::xmm15, x86::xmm0); as.pxor(x86::xmm0, x86::xmm0);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
      as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
    } else if (type == VADDC) {
      as.movdqa(x86::xmm1, x86::xmm15); as.paddw(x86::xmm15, x86::xmm0);
      as.paddusw(x86::xmm1, x86::xmm0); as.pcmpeqw(x86::xmm1, x86::xmm15);
      as.pxor(x86::xmm0, x86::xmm0);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
      as.pcmpeqw(x86::xmm0, x86::xmm1);
      as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
    } else if (type == VSUB) {
      as.movdqa(x86::xmm1, x86::xmm0); as.movdqa(x86::xmm2, x86::xmm15);
      as.psubw(x86::xmm15, x86_spillq(VCO_LO + cfg.cop2));
      as.psubsw(x86::xmm2, x86_spillq(VCO_LO + cfg.cop2));
      as.psubw(x86::xmm1, x86::xmm15); as.psubsw(x86::xmm0, x86::xmm2);
      as.pcmpgtw(x86::xmm2, x86::xmm15); as.paddsw(x86::xmm0, x86::xmm2);
      as.movdqa(x86::xmm15, x86::xmm0); as.pxor(x86::xmm0, x86::xmm0);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
      as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
    } else if (type == VSUBC) {
      as.movdqa(x86::xmm1, x86::xmm0); as.psubusw(x86::xmm1, x86::xmm15);
      as.movdqa(x86::xmm2, x86::xmm0); as.pcmpeqw(x86::xmm2, x86::xmm15);
      as.psubw(x86::xmm0, x86::xmm15); as.movdqa(x86::xmm15, x86::xmm0);
      as.pxor(x86::xmm0, x86::xmm0); as.pcmpeqw(x86::xmm2, x86::xmm0);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm2);
      as.pcmpeqw(x86::xmm0, x86::xmm1); as.pand(x86::xmm2, x86::xmm0);
      as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm2);
    }
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
    if (type == VADD || type == VSUB) as.movdqa(x86::xmm15, x86::xmm1);
  }

  enum VMOV_Type { VMOV, VRCP, VRSQ };

  template <VMOV_Type type, bool low>
  void vmov(uint32_t instr) {
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    uint8_t se = rs(instr) & 0x7, de = rd(instr) & 0x7;
    uint8_t elem = 7 - (type == VMOV && low ? de : se);
    elem_spec(rs(instr)), as.pextrw(x86::eax, x86::xmm15, elem);
    printf("COP2 MOV of $%d to $%d\n", rt(instr), sa(instr));
    if (type == VRCP || type == VRSQ) {
      printf("VRCP/VRSQ Operation\n");
      Label after_sext = as.newLabel();
      if (low) {
        as.or_(x86::eax, x86_spill(DIV_IN + cfg.cop2));
        as.test(x86_spill(DIV_H + cfg.cop2), 1), as.jnz(after_sext);
      }
      as.movsx(x86::eax, x86::ax); as.bind(after_sext);
      as.mov(x86_spill(DIV_H + cfg.cop2), 0);
      Label after_recip = as.newLabel();
      // check for special cases, absolute value
      as.mov(x86::ecx, 0xffff0000); as.cmp(x86::eax, 0xffff8000);
      as.cmove(x86::eax, x86::ecx); as.je(after_recip);
      as.cdq(); as.xor_(x86::eax, x86::edx); as.sub(x86::eax, x86::edx);
      as.mov(x86::ecx, 0x7fffffff); as.test(x86::eax, x86::eax);
      as.cmove(x86::eax, x86::ecx); as.je(after_recip);
      // calculate index into reciprocal rom, shift result
      as.bsr(x86::ecx, x86::eax); as.xor_(x86::ecx, 0x1f);
      as.shl(x86::eax, x86::cl); as.shr(x86::eax, 22);
      if (type == VRCP) as.and_(x86::eax, 0x1ff), as.xor_(x86::ecx, 0x1f);
      else {
        as.and_(x86::eax, 0x1fe); as.mov(x86::esi, x86::ecx);
        as.and_(x86::esi, 1); as.or_(x86::eax, x86::esi);
        as.or_(x86::eax, 0x200); as.xor_(x86::ecx, 0x1f); as.shr(x86::ecx, 1);
      }
      as.mov(x86::rsi, (uint64_t)recip_rom);
      as.mov(x86::ax, x86::word_ptr(x86::rsi, x86::eax, 1));
      as.or_(x86::eax, 0x10000); as.shl(x86::eax, 14); as.sar(x86::eax, x86::cl);
      as.xor_(x86::eax, x86::edx); as.bind(after_recip);
      as.mov(x86_spill(DIV_OUT + cfg.cop2), x86::eax);
    } else if (!low) {
      as.shl(x86::eax, 16); as.mov(x86_spill(DIV_IN + cfg.cop2), x86::eax);
      as.mov(x86::eax, x86_spill(DIV_OUT + cfg.cop2));
      as.sar(x86::eax, 16); as.mov(x86_spill(DIV_H + cfg.cop2), 1);
    }
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    auto result = (sax ? x86::xmm(sax) : x86::xmm0);
    if (!sax) as.movdqa(x86::xmm0, x86_spillq(sa(instr) * 2 + cfg.cop2));
    as.pinsrw(result, x86::eax, 7 - de);
    if (!sax) as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm0);
  }

  template <bool eq, bool invert>
  void veq(uint32_t instr) {
    printf("COP2 VEQ of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)), as.movdqa(x86::xmm1, x86::xmm15);
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    if (!eq) {
      as.movdqa(x86::xmm2, x86::xmm15); as.pcmpeqw(x86::xmm15, x86::xmm0);
      as.pand(x86::xmm15, x86_spillq(VCO_LO + cfg.cop2));
      as.pand(x86::xmm15, x86_spillq(VCO_HI + cfg.cop2));
      as.pcmpgtw(x86::xmm2, x86::xmm0); as.por(x86::xmm15, x86::xmm2);
    } else {
      as.pcmpeqd(x86::xmm2, x86::xmm2), as.pcmpeqw(x86::xmm15, x86::xmm0);
      as.pxor(x86::xmm2, x86_spillq(VCO_HI + cfg.cop2));
      as.pand(x86::xmm15, x86::xmm2);
    }
    if (invert) as.pcmpeqd(x86::xmm2, x86::xmm2), as.pxor(x86::xmm15, x86::xmm2);
    as.movdqa(x86_spillq(VCC_LO + cfg.cop2), x86::xmm15);
    as.pand(x86::xmm0, x86::xmm15), as.pandn(x86::xmm15, x86::xmm1);
    as.por(x86::xmm15, x86::xmm0); as.pxor(x86::xmm0, x86::xmm0);
    as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
    as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
    as.movdqa(x86_spillq(VCC_HI + cfg.cop2), x86::xmm0);
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
  }

  template <bool vcr>
  void vch(uint32_t instr) {
    printf("COP2 VCH of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)), as.movdqa(x86::xmm1, x86::xmm15);
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    // xmm0 = vs, xmm1 = vt
    as.pxor(x86::xmm15, x86::xmm15), as.pxor(x86::xmm0, x86::xmm1);
    as.pcmpgtw(x86::xmm15, x86::xmm0), as.pxor(x86::xmm0, x86::xmm1);
    as.pxor(x86::xmm1, x86::xmm15); if (!vcr) as.psubw(x86::xmm1, x86::xmm15);
    as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm15);
    // xmm1/vts = neg ? -vt : vt, xmm15/neg/vco_lo = (vs ^ vt) < 0
    if (!vcr) {
      as.movdqa(x86::xmm2, x86::xmm0), as.psubw(x86::xmm2, x86::xmm1);
      as.pxor(x86::xmm3, x86::xmm3), as.pcmpeqw(x86::xmm3, x86::xmm2);
      as.pcmpeqw(x86::xmm2, x86::xmm15), as.pand(x86::xmm2, x86::xmm15);
      as.por(x86::xmm3, x86::xmm2), as.movdqa(x86_spillq(VCE_LO + cfg.cop2), x86::xmm2);
      as.pcmpeqd(x86::xmm2, x86::xmm2), as.pxor(x86::xmm2, x86::xmm3);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm2);
    }
    // vce = neg && vs == vts - 1, neq/vco_hi = vts != vs && !vce
    as.movdqa(x86::xmm2, x86::xmm0), as.pcmpgtw(x86::xmm2, x86::xmm1);
    as.movdqa(x86::xmm3, x86::xmm1), as.pcmpgtw(x86::xmm3, x86::xmm0);
    // xmm2 = vs > vts, xmm3 = vts > vs
    as.pxor(x86::xmm4, x86::xmm4), as.pcmpgtw(x86::xmm4, x86::xmm0);
    as.pand(x86::xmm4, x86::xmm15), as.pand(x86::xmm2, x86::xmm15);
    as.pandn(x86::xmm15, x86::xmm3), as.por(x86::xmm4, x86::xmm15);
    as.pcmpeqd(x86::xmm3, x86::xmm3), as.pxor(x86::xmm4, x86::xmm3);
    as.movdqa(x86_spillq(VCC_HI + cfg.cop2), x86::xmm4);
    // vcc_hi = neg ? vs >= 0 : vs >= vts
    as.movdqa(x86::xmm4, x86::xmm0), as.pcmpgtw(x86::xmm4, x86::xmm3);
    as.movdqa(x86::xmm3, x86_spillq(VCO_LO + cfg.cop2));
    as.pandn(x86::xmm3, x86::xmm4), as.por(x86::xmm3, x86::xmm2);
    as.pcmpeqd(x86::xmm4, x86::xmm4), as.pxor(x86::xmm4, x86::xmm3);
    as.movdqa(x86_spillq(VCC_LO + cfg.cop2), x86::xmm4);
    // vcc_lo = neg ? vs <= vts : vs <= 0
    as.por(x86::xmm15, x86::xmm2), as.pand(x86::xmm0, x86::xmm15);
    as.pandn(x86::xmm15, x86::xmm1), as.por(x86::xmm15, x86::xmm0);
    // xmm15 = (neg ? vs > vts : vts > vs) ? vs : vts
    if (vcr) {
      as.pxor(x86::xmm0, x86::xmm0);
      as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
      as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
      as.movdqa(x86_spillq(VCE_LO + cfg.cop2), x86::xmm0);
    }
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
  }

  void vcl(uint32_t instr) {
    printf("COP2 VCL of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)), as.movdqa(x86::xmm1, x86::xmm15);
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    auto neg = x86_spillq(VCO_LO + cfg.cop2), neq = x86_spillq(VCO_HI + cfg.cop2);
    auto gte = x86_spillq(VCC_HI + cfg.cop2), lte = x86_spillq(VCC_LO + cfg.cop2);
    as.movdqa(x86::xmm15, neg), as.movdqa(x86::xmm2, gte);
    as.pxor(x86::xmm1, x86::xmm15), as.psubw(x86::xmm1, x86::xmm15);
    // xmm0 = vs, xmm1/vts = neg ? -vt : vt, xmm15/neg = vco_lo
    as.movdqa(x86::xmm3, x86::xmm15), as.por(x86::xmm3, neq);
    as.movdqa(x86::xmm4, x86::xmm1), as.psubusw(x86::xmm4, x86::xmm0);
    as.pcmpeqw(x86::xmm4, x86::xmm3), as.pand(x86::xmm2, x86::xmm3);
    as.pandn(x86::xmm3, x86::xmm4), as.por(x86::xmm3, x86::xmm2);
    // gte/vcc_hi = (neg || neq) ? vcc_hi : vs >= vts (unsigned)
    as.movdqa(gte, x86::xmm3), as.movdqa(x86::xmm2, x86::xmm0);
    as.pcmpeqw(x86::xmm2, x86::xmm1), as.movdqa(x86::xmm3, x86::xmm0);
    as.psubusw(x86::xmm3, x86::xmm1), as.pxor(x86::xmm3, x86::xmm15);
    as.pcmpeqw(x86::xmm3, x86::xmm15);
    // eq/xmm2 = vs == vts, ncarry/xmm3 = vts >= vs (unsigned)
    as.movdqa(x86::xmm4, x86::xmm2), as.por(x86::xmm2, x86::xmm3);
    as.pand(x86::xmm3, x86::xmm4), as.movdqa(x86::xmm4, x86_spillq(VCE_LO + cfg.cop2));
    as.pand(x86::xmm2, x86::xmm4), as.pandn(x86::xmm4, x86::xmm3);
    as.por(x86::xmm4, x86::xmm2), as.movdqa(x86::xmm3, neq);
    // compare = vce ? (eq || ncarry) : (eq && ncarry)
    as.pcmpeqd(x86::xmm2, x86::xmm2), as.pxor(x86::xmm3, x86::xmm2);
    as.pand(x86::xmm3, x86::xmm15), as.pand(x86::xmm4, x86::xmm3);
    as.pandn(x86::xmm3, lte), as.por(x86::xmm3, x86::xmm4);
    // lte/vcc_lo = (neg && !neq) ? compare : vcc_lo
    as.movdqa(lte, x86::xmm3), as.pand(x86::xmm3, x86::xmm15);
    as.pandn(x86::xmm15, gte), as.por(x86::xmm15, x86::xmm3);
    as.pand(x86::xmm1, x86::xmm15), as.pandn(x86::xmm15, x86::xmm0);
    as.por(x86::xmm15, x86::xmm1), as.pxor(x86::xmm0, x86::xmm0);
    // xmm15 = (neg ? lte : gte) ? vts : vs
    as.movdqa(neg, x86::xmm0), as.movdqa(neq, x86::xmm0);
    as.movdqa(x86_spillq(VCE_LO + cfg.cop2), x86::xmm0);
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
  }

  void vmrg(uint32_t instr) {
    printf("COP2 VMRG of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr)), as.movdqa(x86::xmm1, x86::xmm15);
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (rdx) as.movdqa(x86::xmm0, x86::xmm(rdx));
    else as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    as.movdqa(x86::xmm15, x86_spillq(VCC_LO + cfg.cop2));
    as.pand(x86::xmm0, x86::xmm15), as.pandn(x86::xmm15, x86::xmm1);
    as.por(x86::xmm15, x86::xmm0);
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
    as.pxor(x86::xmm0, x86::xmm0);
    as.movdqa(x86_spillq(VCO_HI + cfg.cop2), x86::xmm0);
    as.movdqa(x86_spillq(VCO_LO + cfg.cop2), x86::xmm0);
  }

  enum VAND_Type { VAND, VOR, VXOR };

  template <VAND_Type type, bool invert>
  void vand(uint32_t instr) {
    printf("COP2 VAND of $%d and $%d to $%d\n", rt(instr), rd(instr), sa(instr));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm15, x86::xmm(rtx));
    else as.movdqa(x86::xmm15, x86_spillq(rt(instr) * 2 + cfg.cop2));
    elem_spec(rs(instr));
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    if (type == VAND) {
      if (rdx) as.pand(x86::xmm15, x86::xmm(rdx));
      else as.pand(x86::xmm15, x86_spillq(rd(instr) * 2 + cfg.cop2));
    } else if (type == VOR) {
      if (rdx) as.por(x86::xmm15, x86::xmm(rdx));
      else as.por(x86::xmm15, x86_spillq(rd(instr) * 2 + cfg.cop2));
    } else if (type == VXOR) {
      if (rdx) as.pxor(x86::xmm15, x86::xmm(rdx));
      else as.pxor(x86::xmm15, x86_spillq(rd(instr) * 2 + cfg.cop2));
    }
    if (invert) as.pcmpeqd(x86::xmm0, x86::xmm0), as.pxor(x86::xmm15, x86::xmm0);
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm15);
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm15);
  }

  void vsar(uint32_t instr) {
    printf("COP2 VSAR into $%d\n", sa(instr));
    uint8_t acc = (rs(instr) & 0x3) + 13;
    uint8_t sax = x86_reg(sa(instr) * 2 + cfg.cop2);
    if (sax) as.movdqa(x86::xmm(sax), x86::xmm(acc));
    else as.movdqa(x86_spillq(sa(instr) * 2 + cfg.cop2), x86::xmm(acc));
  }

  void mfc2(uint32_t instr) {
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2);
    auto result = (rdx ? x86::xmm(rdx) : x86::xmm0);
    if (!rdx) as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    if (sa(instr) & 0x2) as.palignr(result, result, 15);
    as.pextrw(x86::eax, result, 7 - (sa(instr) >> 2));
    if (sa(instr) & 0x2) as.palignr(result, result, 1);
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.movsx(x86::gpq(rtx), x86::ax);
    else {
      as.movsx(x86::rax, x86::ax);
      as.mov(x86_spilld(rt(instr)), x86::rax);
    }
  }

  void mtc2(uint32_t instr) {
    printf("COP2 MTC2 into $%d\n", rd(instr));
    uint8_t rdx = x86_reg(rd(instr) * 2 + cfg.cop2), rtx = x86_reg(rt(instr));
    auto result = (rdx ? x86::xmm(rdx) : x86::xmm0);
    if (!rdx) as.movdqa(x86::xmm0, x86_spillq(rd(instr) * 2 + cfg.cop2));
    if (sa(instr) & 0x2) as.palignr(result, result, 15);
    if (rtx) as.pinsrw(result, x86::gpd(rtx), 7 - (sa(instr) >> 2));
    else as.pinsrw(result, x86_spill(rt(instr)), 7 - (sa(instr) >> 2));
    if (sa(instr) & 0x2) as.palignr(result, result, 1);
    if (!rdx) as.movdqa(x86_spillq(rd(instr) * 2 + cfg.cop2), x86::xmm0);
  }

  void cfc2(uint32_t instr) {
    if (rt(instr) == 0) return;
    as.mov(x86::rax, (uint64_t)0x01030507090b0d0f); as.movq(x86::xmm1, x86::rax);
    as.movdqa(x86::xmm0, x86_spillq((rd(instr) & 0x3) * 4 + VCO_LO + cfg.cop2));
    as.pshufb(x86::xmm0, x86::xmm1); as.pmovmskb(x86::ecx, x86::xmm0);
    as.movdqa(x86::xmm0, x86_spillq((rd(instr) & 0x3) * 4 + VCO_HI + cfg.cop2));
    as.pshufb(x86::xmm0, x86::xmm1); as.pmovmskb(x86::eax, x86::xmm0);
    as.and_(x86::ecx, 0xff); as.shl(x86::eax, 8); as.or_(x86::eax, x86::ecx);
    uint8_t rtx = x86_reg(rt(instr));
    if (rtx) as.movsx(x86::gpq(rtx), x86::ax);
    else {
      as.movsx(x86::rax, x86::ax);
      as.mov(x86_spilld(rt(instr)), x86::rax);
    }
  }

  void ctc2(uint32_t instr) {
    for (uint8_t i = 0; i < 2; ++i) {
      as.pxor(x86::xmm0, x86::xmm0);
      if (rt(instr) != 0) {
        as.mov(x86::rax, (uint64_t)0x0102040810204080); as.movq(x86::xmm1, x86::rax);
        auto mask = (i ? x86::xmm0 : x86::xmm1), val = (i ? x86::xmm1 : x86::xmm0);
        uint8_t rtx = x86_reg(rt(instr)); as.punpcklbw(mask, val);
        rtx ? as.movd(val, x86::gpd(rtx)) : as.movd(val, x86_spill(rt(instr)));
        as.pshuflw(val, val, 0); as.pshufd(val, val, 0), as.pand(val, mask);
        as.pcmpeqw(x86::xmm0, x86::xmm1);
      }
      uint32_t off = i * 2 + VCO_LO + cfg.cop2;
      as.movdqa(x86_spillq((rd(instr) & 0x3) * 4 + off), x86::xmm0);
      if (rd(instr) & 0x2) return;
    }
  }

  inline int32_t sext(uint32_t val, uint32_t bits=32) {
    if (bits >= 32) return val;
    uint32_t mask = (1 << bits) - 1, sign = 1 << (bits - 1);
    return ((val & mask) ^ sign) - sign;
  }

  template <typename T>
  void ldv(uint32_t instr) {
    uint8_t rsx = x86_reg(rs(instr));
    int32_t off = sext(instr, 7) << __builtin_ctz(sizeof(T));
    // LDV BASE(RS), RT, OFFSET(IMMEDIATE)
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, off), x86_read<T>(0);
    // align old register values
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    auto result = (rtx ? x86::xmm(rtx) : x86::xmm0);
    if (!rtx) as.movdqa(x86::xmm0, x86_spillq(rt(instr) * 2 + cfg.cop2));
    uint8_t elem = (sa(instr) >> 1) & (sizeof(T) - 1);
    if (elem) as.palignr(result, result, 16 - elem), as.movdqa(x86::xmm1, result);
    // load new register values
    if (sizeof(T) == 1) as.pinsrb(result, x86::rax, 0xf - (sa(instr) >> 1));
    if (sizeof(T) == 2) as.pinsrw(result, x86::rax, 0x7 - (sa(instr) >> 2));
    if (sizeof(T) == 4) as.pinsrd(result, x86::rax, 0x3 - (sa(instr) >> 3));
    if (sizeof(T) == 8) as.pinsrq(result, x86::rax, 0x1 - (sa(instr) >> 4));
    if (elem) as.palignr(x86::xmm1, result, elem), as.movdqa(result, x86::xmm1);
    if (!rtx) as.movdqa(x86_spillq(rt(instr) * 2 + cfg.cop2), x86::xmm0);
  }

  template <typename T>
  void sdv(uint32_t instr) {
    uint8_t rsx = x86_reg(rs(instr));
    int32_t off = sext(instr, 7) << __builtin_ctz(sizeof(T));
    // SDV BASE(RS), RT, OFFSET(IMMEDIATE)
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    as.add(x86::ecx, off);
    // align old register values
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    auto result = (rtx ? x86::xmm(rtx) : x86::xmm0);
    if (!rtx) as.movdqa(x86::xmm0, x86_spillq(rt(instr) * 2 + cfg.cop2));
    uint8_t elem = (sa(instr) >> 1) & (sizeof(T) - 1);
    if (elem) as.palignr(result, result, 16 - elem);
    // load new register values
    if (sizeof(T) == 1) as.pextrb(x86::rsi, result, 0xf - (sa(instr) >> 1));
    if (sizeof(T) == 2) as.pextrw(x86::rsi, result, 0x7 - (sa(instr) >> 2));
    if (sizeof(T) == 4) as.pextrd(x86::rsi, result, 0x3 - (sa(instr) >> 3));
    if (sizeof(T) == 8) as.pextrq(x86::rsi, result, 0x1 - (sa(instr) >> 4));
    if (elem) as.palignr(result, result, elem);
    x86_write<T>(0);
  }

  template <bool right>
  void lqv(uint32_t instr) {
    // compute load address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * 16;
    as.add(x86::ecx, off), as.and_(x86::ecx, 0xfff);
    // load byte-swapped data from address
    as.mov(x86::rax, (uint64_t)cfg.mem - right * 16);
    as.movdqu(x86::xmm1, x86::dqword_ptr(x86::rax, x86::rcx));
    as.pshufb(x86::xmm1, x86_spillq(cfg.pool));
    // mask loaded data based on alignment
    as.and_(x86::ecx, 0xf), as.neg(x86::rcx);
    uint32_t mask = cfg.pool * 8 + (3 - right) * 16 + (sa(instr) >> 1);
    as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rbp, x86::rcx, 0, mask));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    // merge with old register values (assumes e=0)
    if (sa(instr)) as.psrldq(x86::xmm1, sa(instr) >> 1);
    if (rtx) as.pblendvb(x86::xmm1, x86::xmm(rtx));
    else as.pblendvb(x86::xmm1, x86_spillq(rt(instr) * 2 + cfg.cop2));
    if (rtx) as.movdqa(x86::xmm(rtx), x86::xmm1);
    else as.movdqa(x86_spillq(rt(instr) * 2 + cfg.cop2), x86::xmm1);
  }

  template <bool right>
  void sqv(uint32_t instr) {
    // compute store address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * 16;
    as.add(x86::ecx, off), as.and_(x86::ecx, 0xfff);
    as.mov(x86::rsi, x86::rcx), as.and_(x86::ecx, 0xf);
    // load data from address
    as.mov(x86::rax, (uint64_t)cfg.mem - right * 16);
    as.movdqu(x86::xmm1, x86::dqword_ptr(x86::rax, x86::rsi));
    // mask loaded data based on alignment
    uint32_t mask = cfg.pool * 8 + (2 - right) * 16;
    as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rbp, x86::rcx, 0, mask));
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    // merge with byte-swapped register values (assumes e=0)
    if (rtx) as.movdqa(x86::xmm2, x86::xmm(rtx));
    else as.movdqa(x86::xmm2, x86_spillq(rt(instr) * 2 + cfg.cop2));
    as.pshufb(x86::xmm2, x86_spillq(cfg.pool));
    if (sa(instr)) as.palignr(x86::xmm2, x86::xmm2, sa(instr) >> 1);
    as.pblendvb(x86::xmm1, x86::xmm2);
    as.movdqu(x86::dqword_ptr(x86::rax, x86::rsi), x86::xmm1);
  }

  enum LPV_Type { LPV, LUV, LHV, LFV };

  template <LPV_Type type>
  void lpv(uint32_t instr) {
    // compute load address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * (type < 2 ? 8 : 16);
    int32_t elem  = (sa(instr) >> 1) & 0xf;
    as.add(x86::ecx, off), as.mov(x86::esi, x86::ecx);
    as.and_(x86::ecx, 0xff8), as.and_(x86::esi, 0x7);
    // load byte-swapped data from address
    as.mov(x86::edx, elem), as.sub(x86::edx, x86::esi);
    as.and_(x86::edx, 0xf), as.mov(x86::rax, (uint64_t)cfg.mem);
    as.movdqu(x86::xmm1, x86::dqword_ptr(x86::rbp, x86::rdx, 0, cfg.pool * 8));
    as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rax, x86::rcx));
    as.pshufb(x86::xmm0, x86::xmm1);
    // load into lanes depending on stride
    constexpr uint8_t masks[] = { 8, 8, 10, 12 };
    as.pshufb(x86::xmm0, x86_spillq(cfg.pool + masks[type]));
    if (type != LPV) as.psrlw(x86::xmm0, 1);
    // merge with old register if LFV
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (type == LFV) {
      uint32_t base = rt(instr) * 16 + cfg.cop2 * 8;
      for (int32_t e = elem; e < elem + 8 && e < 16; ++e) {
        as.pextrb(x86::al, x86::xmm0, 15 - e);
        if (rtx) as.pinsrb(x86::xmm(rtx), x86::al, 15 - e);
        else as.mov(x86::byte_ptr(x86::rbp, base + 15 - e), x86::al);
      }
    } else {
      if (rtx) as.movdqa(x86::xmm(rtx), x86::xmm0);
      else as.movdqa(x86_spillq(rt(instr) * 2 + cfg.cop2), x86::xmm0);
    }
  }

  template <LPV_Type type>
  void spv(uint32_t instr) {
    // compute store address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * (type < 2 ? 8 : 16);
    int32_t elem = (sa(instr) >> 1) & 0xf;
    // copy register values
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm1, x86::xmm(rtx));
    else as.movdqa(x86::xmm1, x86_spillq(rt(instr) * 2 + cfg.cop2));
    as.mov(x86::rax, (uint64_t)cfg.mem);
    if (type == LPV || type == LUV) {
      // unpack depending on stride
      as.add(x86::ecx, off), as.and_(x86::ecx, 0xfff);
      auto temp = type ^ (elem < 8) ? x86::xmm1 : x86::xmm2;
      as.movdqa(x86::xmm2, x86::xmm1), as.psllw(temp, 1);
      as.palignr(x86::xmm2, x86::xmm1, 16 - ((elem * 2) & 0xf));
      as.pshufb(x86::xmm2, x86_spillq(14 + cfg.pool));
      as.pextrq(x86::rsi, x86::xmm2, 0);
      as.mov(x86::qword_ptr(x86::rax, x86::rcx), x86::rsi);
    } else if (type == LHV) {
      // compute store address
      as.add(x86::ecx, off), as.mov(x86::esi, x86::ecx);
      as.and_(x86::ecx, 0xff9), as.and_(x86::esi, 0x6);
      as.mov(x86::edx, elem), as.sub(x86::edx, x86::esi);
      // left shift register values
      as.and_(x86::edx, 0xf), as.movdqa(x86::xmm2, x86::xmm1);
      as.psllq(x86::xmm1, 1), as.pslldq(x86::xmm2, 8);
      as.psrlq(x86::xmm2, 63), as.por(x86::xmm1, x86::xmm2);
      // unpack depending on stride
      as.movdqu(x86::xmm2, x86::dqword_ptr(x86::rax, x86::rcx));
      as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rbp, x86::rdx, 0, cfg.pool * 8));
      as.pshufb(x86::xmm1, x86::xmm0), as.pcmpeqw(x86::xmm0, x86::xmm0);
      as.psllw(x86::xmm0, 8), as.pblendvb(x86::xmm1, x86::xmm2);
      as.movdqu(x86::dqword_ptr(x86::rax, x86::rcx), x86::xmm1);
    } else if (type == LFV) {
      as.add(x86::ecx, off), as.mov(x86::esi, x86::ecx);
      as.and_(x86::ecx, 0xff8), as.and_(x86::esi, 0x7);
      as.psllq(x86::xmm1, 1), as.add(x86::rax, x86::rcx);
      as.pshufb(x86::xmm1, x86_spillq(16 + cfg.pool));
      for (int32_t i = 0; i < 4; ++i) {
        uint8_t e = elem + elem / 8 + i * 4;
        as.pextrb(x86::cl, x86::xmm1, e & 0xf);
        as.mov(x86::byte_ptr(x86::rax, x86::rsi), x86::cl);
        as.add(x86::esi, 4), as.and_(x86::esi, 0xf);
      }
    }
  }

  void ltv(uint32_t instr) {
    // compute load address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * 16;
    as.add(x86::ecx, off), as.and_(x86::ecx, 0xff8);
    // load byte-swapped data from address
    as.mov(x86::rax, (uint64_t)cfg.mem);
    as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rax, x86::rcx));
    as.and_(x86::ecx, 0x8), as.sub(x86::ecx, sa(instr) >> 1); as.and_(x86::ecx, 0xf);
    as.movdqu(x86::xmm1, x86::dqword_ptr(x86::rbp, x86::rcx, 0, cfg.pool * 8));
    as.pshufb(x86::xmm0, x86::xmm1);
    uint8_t base = (rt(instr) & 0x18) * 2;
    uint8_t elem = (sa(instr) >> 2) & 0x7;
    // insert data into correct lanes
    for (uint8_t i = 0; i < 8; ++i) {
      uint8_t rti = base + ((elem + i) & 0x7) * 2;
      uint8_t rtx = x86_reg(rti + cfg.cop2);
      uint32_t dst = (rti + cfg.cop2) * 8 + (7 - i) * 2;
      as.pextrw(x86::dx, x86::xmm0, 7 - i);
      if (rtx) as.pinsrw(x86::xmm(rtx), x86::dx, 7 - i);
      else as.mov(x86::word_ptr(x86::rbp, dst), x86::dx);
    }
  }

  void stv(uint32_t instr) {
    // compute store address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * 16;
    as.add(x86::ecx, off), as.and_(x86::ecx, 0xff8);
    // find affected registers and lanes
    as.mov(x86::rax, (uint64_t)cfg.mem);
    uint8_t base = (rt(instr) & ~0x7) * 2;
    uint8_t elem = (sa(instr) >> 2) & 0x7;
    for (uint8_t i = 0; i < 8; ++i) {
      // extract data from correct lanes
      uint8_t rti = base + ((elem + i) & 0x7) * 2;
      uint8_t rtx = x86_reg(rti + cfg.cop2);
      uint32_t src = (rti + cfg.cop2) * 8 + (7 - i) * 2;
      if (rtx) as.pextrw(x86::dx, x86::xmm(rtx), 7 - i);
      else as.mov(x86::dx, x86::word_ptr(x86::rbp, src));
      // store byte-swapped data from address
      as.movbe(x86::word_ptr(x86::rax, x86::rcx, 0, i * 2), x86::dx);
    }
  }

  void swv(uint32_t instr) {
    // compute store address
    uint8_t rsx = x86_reg(rs(instr));
    if (rsx) as.mov(x86::ecx, x86::gpd(rsx));
    else as.mov(x86::ecx, x86_spill(rs(instr)));
    int32_t off = sext(instr, 7) * 16;
    as.add(x86::ecx, off), as.mov(x86::esi, x86::ecx);
    as.and_(x86::ecx, 0xff8), as.and_(x86::esi, 0x7);
    // copy register values
    uint8_t rtx = x86_reg(rt(instr) * 2 + cfg.cop2);
    if (rtx) as.movdqa(x86::xmm1, x86::xmm(rtx));
    else as.movdqa(x86::xmm1, x86_spillq(rt(instr) * 2 + cfg.cop2));
    // rotate based on alignment
    as.mov(x86::edx, sa(instr) >> 1); as.sub(x86::edx, x86::esi);
    as.and_(x86::edx, 0xf), as.mov(x86::rax, (uint64_t)cfg.mem);
    as.movdqu(x86::xmm0, x86::dqword_ptr(x86::rbp, x86::rdx, 0, cfg.pool * 8));
    as.pshufb(x86::xmm1, x86::xmm0); as.mov(x86::rax, (uint64_t)cfg.mem);
    as.movdqu(x86::dqword_ptr(x86::rax, x86::rcx), x86::xmm1);
  }

  void lwc2(uint32_t instr) {
    switch (rd(instr)) {
      case 0x0: ldv<uint8_t>(instr); break;
      case 0x1: ldv<uint16_t>(instr); break;
      case 0x2: ldv<uint32_t>(instr); break;
      case 0x3: ldv<uint64_t>(instr); break;
      case 0x4: lqv<false>(instr); break;
      case 0x5: lqv<true>(instr); break;
      case 0x6: lpv<LPV>(instr); break;
      case 0x7: lpv<LUV>(instr); break;
      case 0x8: lpv<LHV>(instr); break;
      case 0x9: lpv<LFV>(instr); break;
      case 0xb: ltv(instr); break;
      default: invalid(instr); break;
    }
  }

  void swc2(uint32_t instr) {
    switch (rd(instr)) {
      case 0x0: sdv<uint8_t>(instr); break;
      case 0x1: sdv<uint16_t>(instr); break;
      case 0x2: sdv<uint32_t>(instr); break;
      case 0x3: sdv<uint64_t>(instr); break;
      case 0x4: sqv<false>(instr); break;
      case 0x5: sqv<true>(instr); break;
      case 0x6: spv<LPV>(instr); break;
      case 0x7: spv<LUV>(instr); break;
      case 0x8: spv<LHV>(instr); break;
      case 0x9: spv<LFV>(instr); break;
      case 0xa: swv(instr); break;
      case 0xb: stv(instr); break;
      default: invalid(instr); break;
    }
  }

  /* === Basic Block Translation ==*/

  uint32_t special(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch (instr & 0x3f) {
      case 0x00: sll<SLL>(instr); break;
      case 0x02: sll<SRL>(instr); break;
      case 0x03: sll<SRA>(instr); break;
      case 0x04: sllv<SLL>(instr); break;
      case 0x06: sllv<SRL>(instr); break;
      case 0x07: sllv<SRA>(instr); break;
      case 0x08: next_pc = jr(instr); break;
      case 0x09: next_pc = jalr(instr, pc); break;
      case 0x0d: next_pc = break_(pc); break;
      case 0x0f: printf("SYNC\n"); break;
      case 0x10: mfhi<GPR_HI>(instr); break;
      case 0x11: mthi<GPR_HI>(instr); break;
      case 0x12: mfhi<GPR_LO>(instr); break;
      case 0x13: mthi<GPR_LO>(instr); break;
      case 0x14: dsllv<SLL>(instr); break;
      case 0x16: dsllv<SRL>(instr); break;
      case 0x17: dsllv<SRA>(instr); break;
      case 0x18: mult<true>(instr); break;
      case 0x19: mult<false>(instr); break;
      case 0x1a: div<true>(instr); break;
      case 0x1b: div<false>(instr); break;
      case 0x1c: dmult<true>(instr); break;
      case 0x1d: dmult<false>(instr); break;
      case 0x1e: ddiv<true>(instr); break;
      case 0x1f: ddiv<false>(instr); break;
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
      case 0x2c: daddu(instr); break; // DADD
      case 0x2d: daddu(instr); break;
      case 0x2e: dsubu(instr); break; // DSUB
      case 0x2f: dsubu(instr); break;
      case 0x30: case 0x31: case 0x32: case 0x33:
      case 0x34: case 0x36: printf("TRAP\n"); break;
      case 0x38: dsll<SLL, false>(instr); break;
      case 0x3a: dsll<SRL, false>(instr); break;
      case 0x3b: dsll<SRA, false>(instr); break;
      case 0x3c: dsll<SLL, true>(instr); break;
      case 0x3e: dsll<SRL, true>(instr); break;
      case 0x3f: dsll<SRA, true>(instr); break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t regimm(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 16) & 0x1f) {
      case 0x00: next_pc = bltz<BLTZ, false>(instr, pc); break;
      case 0x01: next_pc = bltz<BGEZ, false>(instr, pc); break;
      case 0x02: next_pc = bltzl<BLTZ, false>(instr, pc); break;
      case 0x03: next_pc = bltzl<BGEZ, false>(instr, pc); break;
      case 0x10: next_pc = bltz<BLTZ, true>(instr, pc); break;
      case 0x11: next_pc = bltz<BGEZ, true>(instr, pc); break;
      case 0x12: next_pc = bltzl<BLTZ, true>(instr, pc); break;
      case 0x13: next_pc = bltzl<BGEZ, true>(instr, pc); break;
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
          case 0x1: mfc0(instr); break;
          case 0x4: mtc0(instr); break;
          case 0x5: mtc0(instr); break;
          default: invalid(instr); break;
        }
        break;
      case 0x2: // COP0/2
        switch (instr & 0x3f) {
          case 0x01: tlbr(); break;
          case 0x02: tlbwi<false>(); break;
          case 0x06: tlbwi<true>(); break;
          case 0x08: tlbp(); break;
          case 0x18: next_pc = eret(); break;
          default: invalid(instr); break;
        }
        break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  void check_cop1(uint32_t pc) {
    if (!cfg.cop1 || pc == block_end) return;
    Label cont = as.newLabel();
    as.bsr(x86::eax, x86_spill(12 + cfg.cop0)), as.cmp(x86::eax, 29);
    as.jae(cont), as.and_(x86_spill(13 + cfg.cop0), ~0xff);
    as.or_(x86_spill(13 + cfg.cop0), 0x1000002c);
    as.mov(x86_spill(14 + cfg.cop0), pc - 4);
    as.mov(x86::edi, 0x80000180), as.jmp(exc_label);
    as.bind(cont), cop1_checked = true;
  }

  uint32_t cop1(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    if (!cop1_checked) check_cop1(pc);
    switch ((instr >> 24) & 0x3) {
      case 0x0: // COP1/0
        switch (rs(instr)) {
          case 0x0: mfc1<false>(instr); break;
          case 0x1: mfc1<true>(instr); break;
          case 0x2: cfc1(instr); break;
          case 0x4: mtc1<false>(instr); break;
          case 0x5: mtc1<true>(instr); break;
          case 0x6: ctc1(instr); break;
          default: invalid(instr); break;
        }
        break;
      case 0x1: // COP1/3
        switch (rt(instr) & 0x3) {
          case 0x0: next_pc = bc1t<false>(instr, pc); break;
          case 0x1: next_pc = bc1t<true>(instr, pc); break;
          case 0x2: next_pc = bc1tl<false>(instr, pc); break;
          case 0x3: next_pc = bc1tl<true>(instr, pc); break;
          default: invalid(instr); break;
        }
        break;
      case 0x2: // COP1/2
        switch (instr & 0x3f) {
          case 0x00: add_fmt<ADD_FMT>(instr); break;
          case 0x01: add_fmt<SUB_FMT>(instr); break;
          case 0x02: add_fmt<MUL_FMT>(instr); break;
          case 0x03: add_fmt<DIV_FMT>(instr); break;
          case 0x04: add_fmt<SQR_FMT>(instr); break;
          case 0x05: add_fmt<ABS_FMT>(instr); break;
          case 0x06: add_fmt<MOV_FMT>(instr); break;
          case 0x07: add_fmt<NEG_FMT>(instr); break;
          case 0x08: round_fmt<true, RN_FMT>(instr); break;
          case 0x09: round_fmt<true, RZ_FMT>(instr); break;
          case 0x0a: round_fmt<true, RP_FMT>(instr); break;
          case 0x0b: round_fmt<true, RM_FMT>(instr); break;
          case 0x0c: round_fmt<true, RN_FMT>(instr); break;
          case 0x0d: round_fmt<true, RZ_FMT>(instr); break;
          case 0x0e: round_fmt<true, RP_FMT>(instr); break;
          case 0x0f: round_fmt<true, RM_FMT>(instr); break;
          case 0x20: cvt_s_fmt(instr); break;
          case 0x21: cvt_d_fmt(instr); break;
          case 0x24: cvt_w_fmt<false>(instr); break;
          case 0x25: cvt_w_fmt<true>(instr); break;
          case 0x30: case 0x32: case 0x34: case 0x36:
          case 0x31: case 0x33: case 0x35: case 0x37:
            c_fmt(instr); break;
          case 0x38: case 0x3a: case 0x3c: case 0x3e:
          case 0x39: case 0x3b: case 0x3d: case 0x3f:
            c_fmt(instr); break;
          default: invalid(instr); break;
        }
        break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t cop2(uint32_t instr, uint32_t pc) {
    uint32_t next_pc = pc + 4;
    switch ((instr >> 24) & 0x3) {
      case 0x0: // COP2/0
        switch (rs(instr)) {
          case 0x0: mfc2(instr); break;
          case 0x2: cfc2(instr); break;
          case 0x4: mtc2(instr); break;
          case 0x6: ctc2(instr); break;
          default: invalid(instr); break;
        }
        break;
      case 0x2: case 0x3: // COP2/2
        switch (instr & 0x3f) {
          case 0x00: vmulf<VMULF, false>(instr); break;
          case 0x01: vmulf<VMULU, false>(instr); break;
          case 0x04: vmudn<VMUDL, false>(instr); break;
          case 0x05: vmudn<VMUDM, false>(instr); break;
          case 0x06: vmudn<VMUDN, false>(instr); break;
          case 0x07: vmudn<VMUDH, false>(instr); break;
          case 0x08: vmulf<VMULF, true>(instr); break;
          case 0x09: vmulf<VMULU, true>(instr); break;
          case 0x0c: vmudn<VMUDL, true>(instr); break;
          case 0x0d: vmudn<VMUDM, true>(instr); break;
          case 0x0e: vmudn<VMUDN, true>(instr); break;
          case 0x0f: vmudn<VMUDH, true>(instr); break;
          case 0x10: vadd<VADD>(instr); break;
          case 0x11: vadd<VSUB>(instr); break;
          case 0x13: vadd<VABS>(instr); break;
          case 0x14: vadd<VADDC>(instr); break;
          case 0x15: vadd<VSUBC>(instr); break;
          case 0x1d: vsar(instr); break;
          case 0x20: veq<false, false>(instr); break;
          case 0x21: veq<true, false>(instr); break;
          case 0x22: veq<true, true>(instr); break;
          case 0x23: veq<false, true>(instr); break;
          case 0x24: vcl(instr); break;
          case 0x25: vch<false>(instr); break;
          case 0x26: vch<true>(instr); break;
          case 0x27: vmrg(instr); break;
          case 0x28: vand<VAND, false>(instr); break;
          case 0x29: vand<VAND, true>(instr); break;
          case 0x2a: vand<VOR, false>(instr); break;
          case 0x2b: vand<VOR, true>(instr); break;
          case 0x2c: vand<VXOR, false>(instr); break;
          case 0x2d: vand<VXOR, true>(instr); break;
          case 0x30: vmov<VRCP, false>(instr); break;  // VRCP
          case 0x31: vmov<VRCP, true>(instr); break;   // VRCPL
          case 0x32: vmov<VMOV, false>(instr); break;  // VRCPH
          case 0x33: vmov<VMOV, true>(instr); break;   // VMOV
          case 0x34: vmov<VRSQ, false>(instr); break;  // VRSQ
          case 0x35: vmov<VRSQ, true>(instr); break;   // VRSQL
          case 0x36: vmov<VMOV, false>(instr); break;  // VRSQH
          default: printf("COP2 instruction %x\n", instr); break;
        }
        break;
      default: invalid(instr); break;
    }
    return next_pc;
  }

  uint32_t jit_block(uint32_t pc) {
    uint32_t cycles = 0;
    end_label = as.newLabel(), exc_label = as.newLabel();
    cop1_checked = false;
    for (uint32_t next_pc = pc + 4; pc != block_end; ++cycles) {
      uint32_t instr = cfg.fetch(pc);
      pc = cycles ? check_breaks(pc, next_pc) : next_pc;
      switch (next_pc += 4, instr >> 26) {
        case 0x00: next_pc = special(instr, pc); break;
        case 0x01: next_pc = regimm(instr, pc); break;
        case 0x02: next_pc = j(instr, pc); break;
        case 0x03: next_pc = jal(instr, pc); break;
        case 0x04: next_pc = beq(instr, pc); break;
        case 0x05: next_pc = bne(instr, pc); break;
        case 0x06: next_pc = bltz<BLEZ, false>(instr, pc); break;
        case 0x07: next_pc = bltz<BGTZ, false>(instr, pc); break;
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
        case 0x16: next_pc = bltzl<BLEZ, false>(instr, pc); break;
        case 0x17: next_pc = bltzl<BGTZ, false>(instr, pc); break;
        case 0x18: daddiu(instr); break; // DADDI
        case 0x19: daddiu(instr); break;
        case 0x1a: lwl<uint64_t, false>(instr, pc); break;
        case 0x1b: lwl<uint64_t, true>(instr, pc); break;
        case 0x20: lw<int8_t>(instr, pc); break;
        case 0x21: lw<int16_t>(instr, pc); break;
        case 0x22: lwl<int32_t, false>(instr, pc); break;
        case 0x23: lw<int32_t>(instr, pc); break;
        case 0x24: lw<uint8_t>(instr, pc); break;
        case 0x25: lw<uint16_t>(instr, pc); break;
        case 0x26: lwl<int32_t, true>(instr, pc); break;
        case 0x27: lw<uint32_t>(instr, pc); break;
        case 0x28: sw<uint8_t>(instr, pc); break;
        case 0x29: sw<uint16_t>(instr, pc); break;
        case 0x2a: swl<uint32_t, false>(instr, pc); break;
        case 0x2b: sw<uint32_t>(instr, pc); break;
        case 0x2c: swl<uint64_t, false>(instr, pc); break;
        case 0x2d: swl<uint64_t, true>(instr, pc); break;
        case 0x2e: swl<uint32_t, true>(instr, pc); break;
        case 0x2f: printf("CACHE instruction %x\n", instr); break;
        case 0x30: lw<int32_t>(instr, pc); break; // LL
        case 0x31: lwc1<int32_t>(instr, pc); break;
        case 0x32: lwc2(instr); break;
        case 0x35: lwc1<uint64_t>(instr, pc); break;
        case 0x37: lw<uint64_t>(instr, pc); break;
        case 0x38: sw<uint32_t>(instr, pc); break; // SC
        case 0x39: swc1<uint32_t>(instr, pc); break;
        case 0x3a: swc2(instr); break;
        case 0x3d: swc1<uint64_t>(instr, pc); break;
        case 0x3f: sw<uint64_t>(instr, pc); break;
        default: invalid(instr); break;
      }
    }

    as.bind(end_label);
    if (!cfg.is_rsp) {
      as.add(x86_spill(9 + cfg.cop0), cycles / 2);
      Label cont_label = as.newLabel();
      // check cause and status registers
      as.mov(x86::eax, x86_spill(12 + cfg.cop0));
      as.mov(x86::ecx, x86::eax); as.and_(x86::ecx, 0x3);
      as.cmp(x86::ecx, 0x1); as.jne(cont_label);
      as.and_(x86::eax, x86_spill(13 + cfg.cop0));
      as.and_(x86::eax, 0xff00); as.jz(cont_label);
      // set interrupt pc, status
      as.mov(x86_spill(14 + cfg.cop0), x86::edi);
      as.mov(x86::edi, 0x80000180), as.bind(exc_label);
      as.or_(x86_spill(12 + cfg.cop0), 0x2);
      as.bind(cont_label);
    }

    // only check interrupts when status or
    // cause registers change - eret, mtc0, set_irqs
    //if (cfg.cop0[12] & 0x3 != 0x1) continue;
    // as.test(x86_spill(12 + cfg.cop0), 0x2);
    // as.jc(cfg.thunks[4]);
    // check jumps to funcs are actually are rip-relative

    if (cfg.pages) {
      // translate to physical address
      as.mov(x86::ecx, x86::edi);
      as.mov(x86::edx, x86::edi), as.shr(x86::edx, 12);
      as.mov(x86::rax, (uint64_t)cfg.pages);
      auto off = x86::dword_ptr(x86::rax, x86::rdx, 2);
      as.sub(x86::ecx, off), as.mov(x86::eax, 8);
      as.bt(x86::ecx, 30), as.jc(cfg.fn[FN_TLB]);
      // get function pointer from table
      as.mov(x86::rax, (uint64_t)cfg.lookup);
      as.mov(x86::rdx, x86::qword_ptr(x86::rax, x86::rcx, 1));
    } else {
      as.and_(x86::edi, 0xffc);
      as.mov(x86::rax, (uint64_t)cfg.lookup);
      as.mov(x86::rdx, x86::qword_ptr(x86::rax, x86::rdi, 1));
    }

    // check still_top (no intervening events)
    as.mov(x86::rax, (uint64_t)&Sched::until);
    uint32_t time = cfg.is_rsp ? cycles * 2 : cycles;
    as.sub(x86::qword_ptr(x86::rax), time), as.jl(cfg.fn[FN_EXIT]);

    as.cmp(x86::rdx, 0), as.je(cfg.fn[FN_EXIT]);
    as.jmp(x86::rdx);
    // fill lookup table with offsets to exit thunk
    //as.jmp(cfg.fn[FN_EXIT]);
    return cycles;
  }
};

/* === Wrapper interface === */

typedef uint32_t (*RunPtr)(CodePtr ptr);
static JitRuntime runtime;

static const uint16_t pool[23 * 8] = {
  // unaligned load/store
  0x0e0f, 0x0c0d, 0x0a0b, 0x0809, 0x0607, 0x0405, 0x0203, 0x0001,
  0x0e0f, 0x0c0d, 0x0a0b, 0x0809, 0x0607, 0x0405, 0x0203, 0x0001,
  0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  // packed load
  0x08ff, 0x09ff, 0x0aff, 0x0bff, 0x0cff, 0x0dff, 0x0eff, 0x0fff,
  0x01ff, 0x03ff, 0x05ff, 0x07ff, 0x09ff, 0x0bff, 0x0dff, 0x0fff,
  0x0bff, 0x0fff, 0x03ff, 0x07ff, 0x03ff, 0x07ff, 0x0bff, 0x0fff,
  // packed store
  0x0d0f, 0x090b, 0x0507, 0x0103, 0xffff, 0xffff, 0xffff, 0xffff,
  0x030f, 0xffff, 0x010d, 0xffff, 0x070b, 0xffff, 0x0509, 0xffff,
  // scalar quarter
  0x0302, 0x0302, 0x0706, 0x0706, 0x0b0a, 0x0b0a, 0x0f0e, 0x0f0e,
  0x0100, 0x0100, 0x0504, 0x0504, 0x0908, 0x0908, 0x0d0c, 0x0d0c,
  // scalar half
  0x0706, 0x0706, 0x0706, 0x0706, 0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e,
  0x0504, 0x0504, 0x0504, 0x0504, 0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c,
  0x0302, 0x0302, 0x0302, 0x0302, 0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a,
  0x0100, 0x0100, 0x0100, 0x0100, 0x0908, 0x0908, 0x0908, 0x0908,
  // scalar whole
  0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e, 0x0f0e,
  0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c, 0x0d0c,
  0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a, 0x0b0a,
  0x0908, 0x0908, 0x0908, 0x0908, 0x0908, 0x0908, 0x0908, 0x0908,
  0x0706, 0x0706, 0x0706, 0x0706, 0x0706, 0x0706, 0x0706, 0x0706,
  0x0504, 0x0504, 0x0504, 0x0504, 0x0504, 0x0504, 0x0504, 0x0504,
  0x0302, 0x0302, 0x0302, 0x0302, 0x0302, 0x0302, 0x0302, 0x0302,
  0x0100, 0x0100, 0x0100, 0x0100, 0x0100, 0x0100, 0x0100, 0x0100,
};

uint32_t Mips::jit(MipsConfig *cfg, uint32_t pc, CodePtr *ptr) {
  CodeHolder code;
  code.init(runtime.codeInfo());
  MipsJit jit(cfg, code);
  uint32_t len = jit.jit_block(pc);
  runtime.add(ptr, &code);
  return len;
}

uint32_t Mips::run(MipsConfig *cfg, CodePtr block) {
  RunPtr run = (RunPtr)cfg->fn[MipsJit::FN_ENTER];
  return run(block);
}

void Mips::init(MipsConfig *cfg) {
  void *ptr = cfg->regs + cfg->pool;
  if (cfg->pool) memcpy(ptr, pool, sizeof(pool));
  CodeHolder code;
  code.init(runtime.codeInfo());
  MipsJit jit(cfg, code);

  jit.emit_funcs();
  runtime.add(&ptr, &code);
  for (uint32_t i = 0; i < 5; ++i)
    cfg->fn[i] += (uint64_t)ptr;
}
