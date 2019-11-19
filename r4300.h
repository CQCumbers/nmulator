#ifndef R4300_H
#define R4300_H

#include <SDL2/SDL.h>
#include <sys/mman.h>
#include <x86intrin.h>

namespace R4300 {
  uint8_t *pages[0x100] = {nullptr};
  constexpr uint32_t addr_mask = 0x1fffffff;
  constexpr uint32_t page_mask = 0x1fffff;

  template <typename T>
  int64_t read(uint32_t addr);

  /* === MIPS Interface registers === */

  // put into array and read offset without switch?
  constexpr uint32_t mi_version = 0x01010101;
  uint32_t mi_intr = 0x0, mi_mask = 0x0;

  uint32_t mi_read(uint8_t offset) {
    switch (offset) {
      case 0x04: return mi_version;
      case 0x08: return mi_intr;
      case 0x0c: return mi_mask;
      default: printf("[MI] read from %x\n", offset); return 0;
    }
  }

  void mi_write(uint8_t offset, uint32_t val) {
    if (offset == 0x0c) { mi_mask = _pext_u32(val, 0xaaa); return; }
    printf("[MI] write to %x\n", offset);
  }

  /* === Video Interface registers === */

  uint32_t vi_status = 0x0, vi_origin = 0x0;
  uint32_t vi_intr = 0x0, vi_line = 0x0;
  uint32_t vi_width = 0x0, vi_height = 0x0;
  uint32_t vi_line_progress = 0;
  bool vi_dirty = true;

  uint32_t si_status = 0x0;

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;
  uint8_t *pixels = nullptr;

  uint32_t vi_read(uint8_t offset) {
    switch (offset) {
      case 0x00: return vi_status;
      case 0x04: return vi_origin;
      case 0x0c: return vi_intr;
      case 0x10: return vi_line;
      default: printf("[VI] read from %x\n", offset); return 0;
    }
  }

  void vi_write(uint8_t offset, uint32_t val) {
    switch (offset) {
      case 0x00:
        if (val != vi_status) vi_dirty = true;
        vi_status = val; return;
      case 0x04: vi_origin = val & 0xffffff; return;
      case 0x08:
        if (val != vi_width) vi_dirty = true;
        vi_width = val; return;
      case 0x0c: vi_intr = val & 0x3ff; return;
      case 0x10: mi_intr &= ~0x8; return;
      case 0x18:
        if (val != vi_height) vi_dirty = true;
        vi_height = val; return;
      default: printf("[VI] write to %x\n", offset); return;
    }
  }

  void vi_update(uint32_t cycles) {
    vi_line_progress += cycles;
    if (vi_line_progress < 6150) return;

    vi_line_progress = 0;
    if (++vi_line == vi_intr) mi_intr |= 0x8;
    if (vi_line < 484) return;

    vi_line = 0;
    mi_intr |= 0x2; si_status |= 0x1000; // SI INTR

    uint32_t height = ((vi_status & 0x40) ? vi_height : vi_height / 2);
    if (vi_dirty) {
      if (texture) SDL_DestroyTexture(texture);
      auto format = (vi_status & 0x1 ? SDL_PIXELFORMAT_RGBA8888 : SDL_PIXELFORMAT_RGBA5551);
      texture = SDL_CreateTexture(renderer, format, SDL_TEXTUREACCESS_STREAMING, vi_width, height);
      vi_dirty = false;
    }
    if (!(vi_status & 0x1)) {
      uint16_t *out = reinterpret_cast<uint16_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint16_t>(vi_origin + i * 2);
      SDL_UpdateTexture(texture, nullptr, out, vi_width * 2);
    } else {
      uint32_t *out = reinterpret_cast<uint32_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint32_t>(vi_origin + i * 4);
      SDL_UpdateTexture(texture, nullptr, out, vi_width * 4);
    }
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
    for (SDL_Event e; SDL_PollEvent(&e);)
      if (e.type == SDL_QUIT) exit(0);
  }

  /* === Audio Interface registers === */

  constexpr uint32_t ai_status = 0xffffffff;
  uint32_t ai_used = 0, ai_len = 0;
  bool ai_run = false;

  uint32_t ai_read(uint8_t offset) {
    if (offset == 0x0c) return ai_status;
    printf("[AI] read from %x\n", offset); return 0;
  }

  void ai_write(uint8_t offset, uint32_t val) {
    switch (offset) {
      case 0x00: ai_run = true, ai_used = 0; return;
      case 0x04: ai_len = val & 0x3fff8; return;
      case 0x08: ai_run = val & 0x1; return;
      case 0x0c: mi_intr &= ~0x4; return;
      default: printf("[AI] write to %x\n", offset); return;
    }
  }

  void ai_update(uint32_t cycles) {
    if (ai_run && ai_used < 10000 && (ai_used += cycles) >= 10000) mi_intr |= 0x4;
  }

  /* === Peripheral Interface registers === */

  constexpr uint32_t pi_status = 0x0;
  uint32_t pi_ram = 0x0, pi_rom = 0x0;

  uint32_t pi_read(uint8_t offset) {
    if (offset == 0x10) return pi_status;
    printf("[PI] read from %x\n", offset); return 0;
  }

  void pi_write(uint8_t offset, uint32_t val) {
    switch (offset) {
      case 0x00: pi_ram = val & 0xfffff8; return;
      case 0x04: pi_rom = val & 0x1ffffff8; return;
      case 0x08: // DMA completes immediately
        memcpy(pages[0] + pi_rom, pages[0] + pi_ram, val + 1);
        mi_intr |= 0x10; return;
      case 0x0c:
        memcpy(pages[0] + pi_ram, pages[0] + pi_rom, val + 1);
        mi_intr |= 0x10; return;
      case 0x10: mi_intr &= ~0x10; return;
      default: printf("[PI] write to %x\n", offset); return;
    }
  }

  /* === Serial Interface registers === */

  uint32_t si_read(uint8_t offset) {
    if (offset == 0x18) return si_status;
    printf("[SI] read from %x\n", offset); return 0;
  }

  void si_write(uint8_t offset, uint32_t val) {
    if (offset == 0x18) { mi_intr &= ~0x2; si_status &= ~0x1000; return; }
    printf("[SI] write to %x\n", offset);
  }

  /* === Reading and Writing === */

  uint32_t mmio_read(uint32_t addr) {
    switch ((addr >> 20) & 0xff) {
      case 0x43: return mi_read(addr & 0xfc);
      case 0x44: return vi_read(addr & 0xfc);
      case 0x45: return ai_read(addr & 0xfc);
      case 0x46: return pi_read(addr & 0xfc);
      case 0x48: return si_read(addr & 0xfc);
      default: printf("[MMIO] read from %x\n", addr); return 0;
    }
  }

  void mmio_write(uint32_t addr, uint32_t val) {
    switch ((addr >> 20) & 0xff) {
      case 0x43: mi_write(addr & 0xfc, val); return;
      case 0x44: vi_write(addr & 0xfc, val); return;
      case 0x45: ai_write(addr & 0xfc, val); return;
      case 0x46: pi_write(addr & 0xfc, val); return;
      case 0x48: si_write(addr & 0xfc, val); return;
      default: printf("[MMIO] write to %x\n", addr); return;
    }
  }

  template <typename T>
  int64_t read(uint32_t addr) {
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if (!page) return mmio_read(addr);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return static_cast<T>(__builtin_bswap16(*ptr));
      case 4: return static_cast<T>(__builtin_bswap32(*ptr));
      case 8: return static_cast<T>(__builtin_bswap64(*ptr));
    }
  }

  template <typename T>
  void write(uint32_t addr, int64_t val) {
    printf("--- %x written to %x ---\n", val, addr); 
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if (!page) return mmio_write(addr, val);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = __builtin_bswap16(val); return;
      case 4: *ptr = __builtin_bswap32(val); return;
      case 8: *ptr = __builtin_bswap64(val); return;
    }
  }

  /* === Actual CPU Functions === */

  uint64_t reg_array[0x42] = {0};
  uint32_t pc = 0xa4000040;
  constexpr uint8_t hi = 0x20, lo = 0x21;
  constexpr uint8_t dev_cop0 = 0x22;

  uint32_t fetch(uint32_t addr = pc) {
    uint8_t *page = pages[(addr >> 21) & 0xff];
    uint32_t instr = *reinterpret_cast<uint32_t*>(page + (addr & page_mask));
    return __builtin_bswap32(instr);
  }

  void intrs_update(uint32_t cycles) {
    // update IP2 based on MI_INTR and MI_MASK
    if (mi_intr & mi_mask) reg_array[13 + dev_cop0] |= 0x400;
    else reg_array[13 + dev_cop0] &= ~0x400;

    // check ++COUNT against COMPARE, set IP7
    uint64_t &count = reg_array[9 + dev_cop0];
    uint64_t compare = reg_array[11 + dev_cop0];
    uint64_t new_c = (count + cycles) & 0xffffffff;
    if ((count >= compare) ^ (new_c >= compare)) reg_array[13 + dev_cop0] |= 0x8000;
    count = new_c;

    // if interrupt enabled and triggered, set pc to handler address
    uint8_t ip = (reg_array[13 + dev_cop0] >> 8) & 0xff;
    uint8_t im = (reg_array[12 + dev_cop0] >> 8) & 0xff;
    if ((reg_array[12 + dev_cop0] & 0x3) == 0x1 && (ip & im)) {
      printf("Jumping to interrupt instead of %x\n", pc);
      printf("IP: %x MI: %x MASK: %x VI_INTR %x\n", ip, mi_intr, mi_mask, vi_intr);
      printf("count: %llx compare: %llx\n", reg_array[9 + dev_cop0], reg_array[11 + dev_cop0]);
      //printf("0x80195734: %llx\n", read<uint32_t>(0x80195734));
      reg_array[14 + dev_cop0] = pc; pc = 0x80000180; reg_array[12 + dev_cop0] |= 0x2;
    }
    //printf("pc: %x s0: %llx s4: %llx\n", pc, reg_array[16], reg_array[20]);
    //printf("0x80195734: %llx\n", read<uint32_t>(0x80195734));
    //printf("$a3: %llx $t0: %llx, $v0: %llx, $t5: %llx\n", reg_array[7], reg_array[8], reg_array[2], reg_array[13]);
    for (uint8_t i = 0; i < 32; ++i) {
      if ((reg_array[i] & 0x80000000) ^ ((reg_array[i] & 0x100000000) >> 1))
        printf("Sign extension violation on $%x\n", i);
    }
  }

  void init(FILE *file) {
    // setup registers
    reg_array[20] = 0x1;
    reg_array[22] = 0x3f;
    reg_array[29] = 0xffffffffa4001ff0;

    // setup page table
    pages[0] = reinterpret_cast<uint8_t*>(mmap(
      nullptr, 0x20000000, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
    // specially handle SP/DP, DP/MI, VI/AI, PI/RI, and SI ranges
    for (uint32_t i = 0x1; i < 0x100; ++i) {
      if (i < 0x21 || i > 0x24) pages[i] = pages[0] + i * (page_mask + 1);
    }
    pages[0xfe] = nullptr; // PIF range

    // read ROM file into memory
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    fread(pages[0] + 0x04000000, 1, fsize, file);

    // Cartridge Header at 0x10000000
    memcpy(pages[0] + 0x10000000, pages[0] + 0x04000000, fsize);

    // setup VI
    pixels = reinterpret_cast<uint8_t*>(mmap(
      nullptr, 0x200000, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(640, 480, SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
  }
}

#endif
