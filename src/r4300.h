#ifndef R4300_H
#define R4300_H

#include "rsp.h"
#include "rdp.h"
#include <SDL2/SDL.h>
#include <sys/mman.h>
#include <x86intrin.h>

namespace R4300 {
  uint8_t *pages[0x100] = {nullptr};
  constexpr uint32_t addr_mask = 0x1fffffff;
  constexpr uint32_t page_mask = 0x1fffff;

  template <typename T>
  int64_t read(uint32_t addr);
  template <typename T>
  void write(uint32_t addr, int64_t val);

  /* === MIPS Interface registers === */

  // put into array and read offset without switch?
  constexpr uint32_t mi_version = 0x01010101;
  uint32_t mi_irqs = 0x0, mi_mask = 0x0;

  /* === RSP Interface registers === */

  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;

  template <bool write>
  void rsp_dma(uint32_t val) {
    uint32_t skip = val >> 20, count = (val >> 12) & 0xff, len = val & 0xfff;
    printf("RSP DMA with skip %d, count %d, len %d\n", skip, count, len);
    uint8_t *ram = pages[0] + rsp_cop0[1], *mem = RSP::dmem + rsp_cop0[0];
    for (uint8_t i = 0; i <= count; ++i, ram += skip, mem += skip)
      if (write) memcpy(ram, mem, len + 1); else memcpy(mem, ram, len + 1);
  }

  void rsp_update() {
    mi_irqs |= RSP::broke();
  }

  /* === Video Interface registers === */

  uint32_t vi_status = 0x0, vi_origin = 0x0;
  uint32_t vi_irq = 0x0, vi_line = 0x0;
  uint32_t vi_width = 640, vi_height = 480;
  uint32_t vi_line_progress = 0;
  bool vi_dirty = true;

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;
  uint8_t *pixels = nullptr;

  void write_rdp(uint32_t *rdp_out) {
    uint32_t height = vi_height >> !(vi_status & 0x40);
    for (uint32_t i = 0; rdp_out && i < vi_width * height; ++i)
      write<uint32_t>(vi_origin + (i << 2), rdp_out[i]);
  }

  void vi_update(uint32_t cycles) {
    vi_line_progress += cycles;
    if (vi_line_progress < 6150) return;
    vi_line_progress = 0;
    if (++vi_line == vi_irq) mi_irqs |= 0x8;
    if (vi_line < 584) return;

    vi_line = 0; mi_irqs |= 0x2; // SI INTR
    uint8_t format = vi_status & 0x3;
    uint32_t height = vi_height >> !(vi_status & 0x40);

    if (vi_dirty) {
      if (texture) SDL_DestroyTexture(texture);
      texture = SDL_CreateTexture(renderer,
        (format == 2 ? SDL_PIXELFORMAT_RGBA5551 : SDL_PIXELFORMAT_RGBA8888),
        SDL_TEXTUREACCESS_STREAMING, vi_width, height);
      vi_dirty = false;
    }
    //write_rdp(RDP::update(cycles));
    if (format == 2) {
      uint16_t *out = reinterpret_cast<uint16_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint16_t>(vi_origin + (i << 1));
      SDL_UpdateTexture(texture, nullptr, out, vi_width << 1);
    } else if (format == 3) {
      uint32_t *out = reinterpret_cast<uint32_t*>(pixels);
      for (uint32_t i = 0; i < vi_width * height; ++i)
        out[i] = read<uint32_t>(vi_origin + (i << 2));
      SDL_UpdateTexture(texture, nullptr, out, vi_width << 2);
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

  void ai_update(uint32_t cycles) {
    if (ai_run && ai_used < 10000 && (ai_used += cycles) >= 10000) mi_irqs |= 0x4;
  }

  /* === Peripheral Interface registers === */

  constexpr uint32_t pi_status = 0x0;
  uint32_t pi_ram = 0x0, pi_rom = 0x0;

  /* === Reading and Writing === */

  template <typename T>
  int64_t mmio_read(uint32_t addr) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000)
      return RSP::read<T>(addr);
    switch (addr & addr_mask) {
      default: printf("[MMIO] read from %x\n", addr); return 0;
      // RSP Interface
      case 0x4040000: return rsp_cop0[0];
      case 0x4040004: return rsp_cop0[1];
      case 0x4040010: return rsp_cop0[4];
      case 0x404001c: return (rsp_cop0[7] ? 0x1 : rsp_cop0[7]++);
      case 0x4080000: return RSP::pc;
      // RDP Interface
      case 0x4100000: return RDP::pc_start;
      case 0x4100004: return RDP::pc_end;
      case 0x4100008: return RDP::pc;
      case 0x410000c: return RDP::status;
      // MIPS Interface
      case 0x4300004: return mi_version;
      case 0x4300008: return mi_irqs;
      case 0x430000c: return mi_mask;
      // Video Interface
      case 0x4400000: return vi_status;
      case 0x4400004: return vi_origin;
      case 0x4400008: return vi_width;
      case 0x440000c: return vi_irq;
      case 0x4400010: return vi_line;
      case 0x4400018: return vi_height;
      // Audio Interface
      case 0x4500004: return ai_len;
      case 0x450000c: return ai_status;
      // Peripheral Interface
      case 0x4600000: return pi_ram;
      case 0x4600004: return pi_rom;
      case 0x4600010: return pi_status;
      // Serial Interface
      case 0x4800018: return (mi_irqs & 0x2) << 11;
    }
  }

  template <typename T>
  void mmio_write(uint32_t addr, uint32_t val) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000)
      return RSP::write<T>(addr, val);
    switch (addr & addr_mask) {
      default: printf("[MMIO] write to %x: %x\n", addr, val); return;
      // RSP Interface
      case 0x4040000: rsp_cop0[0] = val & 0x1fff; return;
      case 0x4040004: rsp_cop0[1] = val & 0xffffff; return;
      case 0x4040008: rsp_dma<false>(val); return;
      case 0x404000c: rsp_dma<true>(val); return;
      case 0x4040010: RSP::set_status(val); return;
      case 0x404001c: rsp_cop0[7] = 0x0; return;
      case 0x4080000: RSP::pc = val & 0xfff; return;
      // RDP Interface
      case 0x4100000: RDP::pc_start = val & addr_mask; return;
      case 0x4100004:
        RDP::pc_end = val & addr_mask;
        write_rdp(RDP::update(1)); return;
      case 0x410000c: RDP::status = _pext_u32(val, 0x2aa); return;
      // MIPS Interface
      case 0x430000c: mi_mask = _pext_u32(val, 0xaaa); return;
      // Video Interface
      case 0x4400000: 
        if (val == vi_status) return;
        vi_status = val; vi_dirty = true; return;
      case 0x4400004: vi_origin = val & 0xffffff; return;
      case 0x4400008:
        if ((val &= 0xfff) == vi_width) return;
        vi_width = val; vi_dirty = true; return;
      case 0x440000c: vi_irq = val & 0x3ff; return;
      case 0x4400010: mi_irqs &= ~0x8; return;
      case 0x4400018:
        if ((val &= 0x3ff) == vi_height) return;
        vi_height = val; vi_dirty = true; return;
      // Audio Interface
      case 0x4500000: ai_used = 0; ai_run = 1; return;
      case 0x4500004: ai_len = val & 0x3ff8; return;
      case 0x4500008: ai_run = val & 0x1; return;
      case 0x450000c: mi_irqs &= ~0x4; return;
      // Peripheral Interface
      case 0x4600000: pi_ram = val & 0xffffff; return;
      case 0x4600004: pi_rom = val & addr_mask; return;
      case 0x4600008:
        memcpy(pages[0] + pi_ram, pages[0] + pi_rom, val + 1);
        mi_irqs |= 0x10; return;
      case 0x460000c:
        memcpy(pages[0] + pi_ram, pages[0] + pi_rom, val + 1);
        mi_irqs |= 0x10; return;
      case 0x4600010: mi_irqs &= ~0x10; return;
      // Serial Interface
      case 0x4800018: mi_irqs &= ~0x2; return;
    }
  }

  template <typename T>
  int64_t read(uint32_t addr) {
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if (!page) return mmio_read<T>(addr);
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
    uint8_t *page = pages[(addr >> 21) & 0xff];
    if (!page) return mmio_write<T>(addr, val);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = __builtin_bswap16(val); return;
      case 4: *ptr = __builtin_bswap32(val); return;
      case 8: *ptr = __builtin_bswap64(val); return;
    }
  }

  uint32_t fetch(uint32_t addr) {
    return read<uint32_t>(addr);
  }

  /* === Actual CPU Functions === */

  uint64_t reg_array[0x62] = {0};
  uint32_t pc = 0xa4000040;
  constexpr uint8_t hi = 0x20, lo = 0x21;
  constexpr uint8_t dev_cop0 = 0x22, dev_cop1 = 0x42;

  void irqs_update(uint32_t cycles) {
    // update IP2 based on MI_INTR and MI_MASK
    if (mi_irqs & mi_mask) reg_array[13 + dev_cop0] |= 0x400;
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
      printf("IP: %x MI: %x MASK: %x VI_INTR %x\n", ip, mi_irqs, mi_mask, vi_irq);
      printf("COUNT: %llx COMPARE: %llx\n", reg_array[9 + dev_cop0], reg_array[11 + dev_cop0]);
      reg_array[14 + dev_cop0] = pc; pc = 0x80000180; reg_array[12 + dev_cop0] |= 0x2;
    }

    // Debug sign extension
    /*for (uint8_t i = 0; i < 32; ++i) {
      if ((reg_array[i] & 0x80000000) ^ ((reg_array[i] & 0x100000000) >> 1))
        printf("Sign extension violation on $%x\n", i);
    }*/
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
      if (i < 0x20 || i > 0x24) pages[i] = pages[0] + i * (page_mask + 1);
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
    Vulkan::init();

    // setup RSP
    RSP::dmem = pages[0] + 0x04000000;
    RSP::imem = pages[0] + 0x04001000;
    rsp_cop0[4] = 0x1;
  }
}

#endif
