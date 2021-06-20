#include <stdio.h>
#include <nmmintrin.h>
#include "nmulator.h"

#include <vector>
#include "robin_hood.h"

static uint8_t *imem;
static uint64_t regs[150 + 46];

namespace RSP {
  uint8_t *mem = NULL;
  uint64_t *const cop0 = regs + 32;
  uint32_t pc = 0x0;
}

/* === Code change detection === */

static uint8_t code_mask[0x1000];
static CodePtr lookup[0x1000 / 4];

static void unprotect(uint32_t addr, uint8_t *src, uint32_t len) {
  for (uint32_t i = 0; i < len; i += 8) {
    // memcmp src and dst at bitmasked addresses
    uint64_t new_ = *(uint64_t*)(src + i);
    uint64_t old_ = *(uint64_t*)(RSP::mem + addr + i);
    uint64_t mask = *(uint64_t*)(code_mask + ((addr + i) & 0xfff));

    // clear lookup table if change detected
    if (!((old_ ^ new_) & mask)) continue;
    uint32_t pg = (addr + i) & 0xf00;
    memset(code_mask + pg, 0, 0x100);
    memset(lookup + pg / 4, 0, 0x200);
    i = ((addr + i) & ~0xff) + 0xf8 - addr;
  }
}

/* === RSP memory access === */

void RSP::dma(uint32_t val, bool to_ram) {
  // 64-bit align DMA params, limit length to within DMEM/IMEM
  uint32_t skip = (val >> 20) & 0xff8, count = (val >> 12) & 0xff;
  uint32_t len = (val & 0xff8) + 8, max = 0x1000 - (cop0[0] & 0xff8);
  len = (len > max ? max : len), cop0[0] &= 0x1ff8, cop0[1] &= 0x7ffff8;

  // repeatedly DMA bytes from RDRAM to DMEM/IMEM, or vice-versa
  for (uint32_t i = 0; i <= count; ++i) {
    uint8_t *src = to_ram ? mem + cop0[0] : R4300::ram + cop0[1];
    uint8_t *dst = to_ram ? R4300::ram + cop0[1] : mem + cop0[0];
    if (!to_ram && (cop0[0] >> 12)) unprotect(cop0[0], src, len);
    memcpy(dst, src, len), cop0[0] += len, cop0[1] += len + skip;
  }
}

void RSP::set_status(uint32_t val) {
  // update status flags, unhalt RSP
  if (cop0[4] & val & 0x1) Sched::add(TASK_RSP, 0);
  cop0[4] &= ~(val & 0x1), cop0[4] |= (val >> 1) & 0x2;
  cop0[4] &= ~(val & 0x4) >> 1;
  cop0[4] &= ~(pext(val >> 5, 0x3ff) << 5);
  cop0[4] |= pext(val >> 6, 0x3ff) << 5;

  // update MI RSP interrupt
  R4300::unset_irqs((val & 0x8) >> 3);
  R4300::set_irqs((val & 0x10) >> 4);
}

// read instruction, mark code address
static uint32_t fetch(uint32_t addr) {
  write32(code_mask + addr, 0xffffffff);
  return read32(imem + addr);
}

/* === Recompiler config === */

// read CPU-RSP semaphore
static int64_t mfc0(uint32_t) {
  return RSP::cop0[7] ? 1 : RSP::cop0[7]++;
}

// modify RSP and RDP status regs
static void mtc0(uint32_t idx, uint32_t val) {
  switch (idx &= 0x1f) {
    default: return;
    case 0: RSP::cop0[0] = val & 0x1fff; return;
    case 1: RSP::cop0[1] = val & 0xffffff; return;
    case 2: RSP::dma(val, false); return;
    case 3: RSP::dma(val, true); return;
    case 4: RSP::set_status(val); return;
    case 8: RSP::cop0[10] = RSP::cop0[8] = val; return;
    case 9: Sched::move(TASK_RDP, 0), RSP::cop0[9] = val; return;
    case 11: RSP::cop0[11] &= ~pext(val >> 0, 0x7);
      RSP::cop0[11] |= pext(val >> 1, 0x7); return;
  }
}

static int64_t stop_at(uint32_t addr) {
  if (!(addr & 0xff)) return true;
  return false;  // true to single-step
}

static MipsConfig cfg = {
  .regs = regs, .cop0 = 32,
  .cop2 = 64, .pool = 150,
  .lookup = lookup, .fetch = fetch,
  .stop_at = stop_at, .step = NULL,
  .mfc0 = mfc0, .mfc0_mask = 0x0080,
  .mtc0 = mtc0, .mtc0_mask = 0xeb7f,
};

robin_hood::unordered_map<uint32_t, std::vector<Block>> backups;

static uint32_t crc32(uint8_t *bytes, uint32_t len) {
  uint32_t crc = 0, *msg = (uint32_t*)bytes;
  for (uint32_t i = 0; i < len / 4; ++i)
    crc = _mm_crc32_u32(crc, msg[i]);
  return crc;
}

void RSP::update() {
  while (Sched::until >= 0) {
    if (cop0[4] & 0x1) return;
    CodePtr code = lookup[pc / 4];
    if (code) {
      pc = Mips::run(&cfg, code) & 0xffc;
      if ((cop0[4] & 0x42) == 0x42) R4300::set_irqs(0x1);
    } else {
      // search backup blocks for matching hash
      bool skip_compile = false;
      for (auto &backup : backups[pc]) {
        uint32_t hash = crc32(imem + pc, backup.len * 4);
        if (backup.hash == hash) {
          lookup[pc / 4] = backup.code;
          memset(code_mask + pc, 0xff, backup.len * 4);
          skip_compile = true; break;
        }
      }
      if (skip_compile) continue;

      // compile code, store length and hash in backup
      Block block;
      block.len = Mips::jit(&cfg, pc, &block.code);
      block.hash = crc32(imem + pc, block.len * 4);
      backups[pc].push_back(block);
      lookup[pc / 4] = block.code;
    }
  }
  Sched::add(TASK_RSP, 0);
}

void RSP::init(uint8_t *mem) {
  // set mem pointer, initial cop0 values
  RSP::mem = cfg.mem = mem;
  imem = mem + 0x1000;
  cop0[4] = 0x1, cop0[11] = 0x80;
  Mips::init(&cfg);
}

void RSP::test(const char *filename) {
  uint32_t inp_size = 40;
  uint32_t out_size = 1376;

  // load imem instructions
  init((uint8_t*)malloc(0x2800));
  char *name = strdup(filename);
  FILE *instrs = fopen(name, "r");
  fread(imem, 1, 0x1000, instrs);

  // open input/output files
  strcpy(name + strlen(name) - 4, ".inp");
  FILE *input = fopen(name, "r");
  strcpy(name + strlen(name) - 4, ".gol");
  FILE *output = fopen(name, "r");

  // run RSP and print results
  uint8_t *gol = mem + 0x2000;
  uint8_t *out = mem + 0x800;
  uint32_t idx = 0;
  while (fread(mem, inp_size, 1, input)) {
    for (uint32_t i = 0; i < inp_size; i += 4)
      write32(mem + i, *(uint32_t*)(mem + i));
    fread(gol, out_size, 1, output);
    Sched::until = INT64_MAX;
    cop0[4] = pc = 0, update();

    for (uint32_t i = 0; i < out_size; i += 16) {
      printf("%04x: ", i);
      for (uint32_t j = 0; j < 16; j += 4)
        printf("%08x", read32(out + i + j));
      printf("  ");
      for (uint32_t j = 0; j < 16; j += 4)
        printf("%08x", read32(gol + i + j));
      printf("\n");
    }

    bool pass = memcmp(out, gol, out_size) == 0;
    const char *msg = pass ? "passed" : "failed";
    printf("Test %d %s\n\n", idx++, msg);
    if (!pass) exit(1);
  }
  printf("All tests passed\n"), exit(0);
}
