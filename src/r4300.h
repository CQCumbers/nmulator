#ifndef R4300_H
#define R4300_H

#include "rsp.h"
#include "rdp.h"
#include <SDL2/SDL.h>
#include <sys/mman.h>
#include <x86intrin.h>
#include "robin_hood.h"

namespace R4300 {
  uint8_t *pages[0x100] = {nullptr};
  uint32_t tlb[0x20][4] = {{0x80000000}};
  constexpr uint32_t addr_mask = 0x1fffffff;
  constexpr uint32_t page_mask = 0x1fffff;

  bool broke = false, moved = false;
  robin_hood::unordered_map<uint32_t, bool> breaks;
  robin_hood::unordered_map<uint32_t, bool> watch_w;

  template <typename T, bool map=false>
  int64_t read(uint32_t addr);
  template <typename T, bool map=false>
  void write(uint32_t addr, int64_t val);
  void joy_update(SDL_Event &event);

  /* === MIPS Interface registers === */

  // put into array and read offset without switch?
  constexpr uint32_t mi_version = 0x01010101;
  uint32_t mi_irqs = 0x0, mi_mask = 0x0;

  /* === RSP Interface registers === */

  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;

  template <bool write>
  void rsp_dma(uint32_t val) {
    uint32_t skip = val >> 20, count = (val >> 12) & 0xff, len = val & 0xfff;
    printf("[RSP] DMA with count %x, len %x, from %llx to %llx\n", count + 1, len + 1, rsp_cop0[1], rsp_cop0[0]);
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

  void vi_update(uint32_t cycles) {
    vi_line_progress += cycles;
    if (vi_line_progress < 6510) return; // * 8) return;
    if (vi_line == vi_irq) mi_irqs |= 0x8;
    vi_line_progress = 0, vi_line += 0x2;

    if (vi_line < vi_height) return;
    bool interlaced = vi_status & 0x40;
    vi_line = interlaced & ~vi_line;
    if (vi_line == 0x1) return;
    uint8_t format = vi_status & 0x3;
    uint32_t height = vi_height >> !interlaced;

    if (vi_dirty) {
      if (texture) SDL_DestroyTexture(texture);
      texture = SDL_CreateTexture(renderer,
        (format == 2 ? SDL_PIXELFORMAT_RGBA5551 : SDL_PIXELFORMAT_RGBA8888),
        SDL_TEXTUREACCESS_STREAMING, vi_width, height);
      vi_dirty = false;
    }
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

    for (SDL_Event e; SDL_PollEvent(&e);) {
      if (e.type == SDL_QUIT) exit(0);
      joy_update(e);
    }
  }

  /* === Audio Interface registers === */

  constexpr uint32_t ntsc_clock = 48681812;
  uint32_t ai_status = 0x0, ai_ram = 0x0;
  uint32_t ai_len = 0, ai_start = 0;
  uint32_t ai_rate = 0, ai_bits = 0;
  bool ai_run = false, ai_dirty = false;
  SDL_AudioDeviceID audio_dev;

  void ai_dma(uint32_t len) {
    if (!ai_run || ai_start != 0) return;
    printf("[AI] DMA started with len %x\n", len);
    if (ai_dirty) {
      SDL_AudioFormat fmt = (ai_bits ? AUDIO_S16MSB : AUDIO_S8);
      SDL_AudioSpec spec = {
        .freq = static_cast<int>(ntsc_clock / (ai_rate + 1)) << !ai_bits,
        .format = fmt, .channels = 2, .samples = 0x100
      };
      audio_dev = SDL_OpenAudioDevice(nullptr, 0, &spec, nullptr, 0);
      SDL_PauseAudioDevice(audio_dev, 0); ai_dirty = false;
    }
    ai_start = ai_len, ai_len = len & 0x1fff8;
    if (ai_start != 0) ai_status |= 0x80000001;
    SDL_QueueAudio(audio_dev, pages[0] + ai_ram, ai_len);
  }

  void ai_update() {
    if (!ai_run || ai_len == 0) return;
    uint32_t new_len = SDL_GetQueuedAudioSize(audio_dev) >> ai_bits;
    if (new_len <= ai_start || new_len == 0)
      ai_status &= ~0x80000001, ai_start = 0, mi_irqs |= 0x4;
    ai_len = new_len - ai_start;
  }

  /* === Peripheral & Serial Interface registers === */

  constexpr uint32_t pi_status = 0x0;
  uint32_t pi_ram = 0x0, pi_rom = 0x0;
  uint32_t pi_len = 0, pi_cycles = 0;
  bool pi_write = false;
  int8_t stick_x = 0x0, stick_y = 0x0;
  uint16_t buttons = 0x0;
  uint32_t si_ram = 0x0;
  uint64_t eeprom[0x200];

  void pi_update(uint32_t cycles) {
    if (!pi_cycles) return;
    if (pi_cycles > cycles) { pi_cycles -= cycles; return; }
    printf("[PI] DMA from %x to %x, of %x bytes\n", pi_ram, pi_rom, pi_len + 1);
    if (pi_write) memcpy(pages[0] + pi_ram, pages[0] + pi_rom, pi_len + 1);
    else memcpy(pages[0] + pi_rom, pages[0] + pi_ram, pi_len + 1);
    mi_irqs |= 0x10; pi_cycles = 0; return;
  }

  void joy_status(uint32_t channel, uint32_t addr) {
    if (channel == 4) return write<uint32_t>(addr, 0x8000ff);
    if (channel != 0) return write<uint8_t>(addr - 2, 0x83);
    write<uint16_t>(addr, 0x0500); // standard controller type
    write<uint8_t>(addr + 2, 0x02); // no mempack slot
  }

  void joy_read(uint32_t channel, uint32_t addr) {
    if (channel != 0) return write<uint8_t>(addr - 2, 0x84);
    printf("[SI] buttons: %x stick: %x\n", buttons, stick_y);
    write<uint16_t>(addr, buttons);
    write<int8_t>(addr + 2, stick_x);
    write<int8_t>(addr + 3, stick_y);
  }

  void joy_update(SDL_Event &event) {
    if (event.type == SDL_KEYDOWN) {
      switch (event.key.keysym.sym) {
        case SDLK_x: buttons |= (1 << 15); break;  // A
        case SDLK_c: buttons |= (1 << 14); break;  // B
        case SDLK_z: buttons |= (1 << 13); break;  // Z
        case SDLK_RETURN: buttons |= (1 << 12); break;  // Start
        case SDLK_k: buttons |= (1 << 11); break;  // D Up
        case SDLK_j: buttons |= (1 << 10); break;  // D Down
        case SDLK_h: buttons |= (1 << 9); break;   // D Left
        case SDLK_l: buttons |= (1 << 8); break;   // D Right
        case SDLK_a: buttons |= (1 << 5); break;   // Trigger Left
        case SDLK_s: buttons |= (1 << 4); break;   // Trigger Right
        case SDLK_UP: stick_y = 80; break;         // Stick Up
        case SDLK_DOWN: stick_y = -80; break;      // Stick Down
        case SDLK_LEFT: stick_x = -80; break;      // Stick Left
        case SDLK_RIGHT: stick_x = 80; break ;     // Stick Right
      }
    } else if (event.type == SDL_KEYUP) {
      switch (event.key.keysym.sym) {
        case SDLK_x: buttons &= ~(1 << 15); break;  // A
        case SDLK_c: buttons &= ~(1 << 14); break;  // B
        case SDLK_z: buttons &= ~(1 << 13); break;  // Z
        case SDLK_RETURN: buttons &= ~(1 << 12); break;  // Start
        case SDLK_k: buttons &= ~(1 << 11); break;  // D Up
        case SDLK_j: buttons &= ~(1 << 10); break;  // D Down
        case SDLK_h: buttons &= ~(1 << 9); break;   // D Left
        case SDLK_l: buttons &= ~(1 << 8); break;   // D Right
        case SDLK_a: buttons &= ~(1 << 5); break;   // Trigger Left
        case SDLK_s: buttons &= ~(1 << 4); break;   // Trigger Right
        case SDLK_UP: stick_y = 0; break;           // Stick Up
        case SDLK_DOWN: stick_y = 0; break;         // Stick Down
        case SDLK_LEFT: stick_x = 0; break;         // Stick Left
        case SDLK_RIGHT: stick_x = 0; break;        // Stick Right
      }
    }
  }

  void eeprom_read(uint8_t len, uint32_t addr) {
    uint8_t offset = read<uint8_t>(addr);  // assumes t = 2
    for (uint8_t i = 0; i < len; i += 8)
      write<uint64_t>(addr + 1 + i, eeprom[offset + i / 8]);
  }

  void eeprom_write(uint8_t len, uint32_t addr) {
    uint8_t offset = read<uint8_t>(addr);  // assumes t = 10
    eeprom[offset] = read<uint64_t>(addr + 1);
    write<uint8_t>(addr + 9, 0);
  }

  void si_update() {
    uint32_t busy = 0x1fc007ff;
    uint32_t pc = 0x1fc007c0, channel = 0;
    write<uint8_t>(busy, 0x0);
    while (pc < busy) {
      uint8_t t = read<uint8_t>(pc++);
      if (t == 0xfe) return;
      if (t == 0) ++channel, t = 0x80;
      if (t >> 7) continue;
      uint8_t r = read<uint8_t>(pc++);
      switch (read<uint8_t>(pc++)) {
        case 0x00: joy_status(channel, pc); break;
        case 0x01: joy_read(channel, pc); break;
        case 0x02: printf("[SI] mempack read\n"); break;
        case 0x03: printf("[SI] mempack write\n"); break;
        case 0x04: eeprom_read(r, pc); pc += 1; break;
        case 0x05: eeprom_write(r, pc); pc += 9; break;
        case 0xff: joy_status(channel, pc); break;
      }
      pc += r, ++channel;
    }
  }

  /* === Reading and Writing === */

  template <typename T>
  int64_t mmio_read(uint32_t addr) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000)
      return RSP::read<T, true>(addr);
    switch (addr & addr_mask) {
      default: printf("[MMIO] read from %x\n", addr); return 0;
      // RSP Interface
      case 0x4040000: return rsp_cop0[0];
      case 0x4040004: return rsp_cop0[1];
      case 0x4040010: return rsp_cop0[4];
      case 0x404001c: return (rsp_cop0[7] ? 0x1 : rsp_cop0[7]++);
      case 0x4080000: return RSP::pc & 0xfff;
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
      case 0x4600010: return pi_cycles > 0;//pi_status;
      // RDRAM Interface
      case 0x4700000: return 0xe;      // RI_MODE
      case 0x4700004: return 0x40;     // RI_CONFIG
      case 0x470000c: return 0x14;     // RI_SELECT
      case 0x4700010: return 0x63634;  // RI_REFRESH
      // Serial Interface
      case 0x4800018: return (mi_irqs & 0x2) << 11;
    }
  }

  bool logging_on = false;

  template <typename T>
  void mmio_write(uint32_t addr, uint32_t val) {
    if ((addr & addr_mask & ~0x1fff) == 0x4000000) {
      printf("RSP MMIO Write\n");
      return RSP::write<T, true>(addr, val);
    }
    switch (addr & addr_mask) {
      default: printf("[MMIO] write to %x: %x\n", addr, val); return;
      // RSP Interface
      case 0x4040000: rsp_cop0[0] = val & 0x1fff; return;
      case 0x4040004: rsp_cop0[1] = val & 0xffffff;
        printf("Writing %llx to DMA_SRC\n", rsp_cop0[1]);
        if (rsp_cop0[1] == 0xda218 || rsp_cop0[1] == 0x390198)
          printf("Break here!\n"), logging_on = true;
        return;
      case 0x4040008: rsp_dma<false>(val); return;
      case 0x404000c: rsp_dma<true>(val); return;
      case 0x4040010: RSP::set_status(val); return;
      case 0x404001c: rsp_cop0[7] = 0x0; return;
      case 0x4080000: RSP::pc = val & 0xfff; return;
      // RDP Interface
      case 0x4100000:
        RDP::pc_start = val & addr_mask;
        RDP::pc = RDP::pc_start; return;
      case 0x4100004:
        RDP::pc_end = val & addr_mask;
        RDP::update(); return;
      case 0x410000c:
        RDP::status &= ~_pext_u32(val, 0x15);
        RDP::status |= _pext_u32(val, 0x2a); return;
      // MIPS Interface
      case 0x4300000: if (val & 0x800) mi_irqs &= ~0x20; return;
      case 0x430000c:
        mi_mask &= ~_pext_u32(val, 0x555);
        mi_mask |= _pext_u32(val, 0xaaa); return;
      // Video Interface
      case 0x4400000: 
        if (val == vi_status) return;
        vi_status = val; vi_dirty = true; return;
      case 0x4400004: vi_origin = val & 0xffffff; return;
      case 0x4400008:
        if ((val & 0xfff) == vi_width) return;
        vi_width = val & 0xfff; vi_dirty = true; return;
      case 0x440000c: vi_irq = val & 0x3ff; return;
      case 0x4400010: mi_irqs &= ~0x8; return;
      case 0x4400018:
        if ((val & 0x3ff) == vi_height) return;
        vi_height = val & 0x3ff; vi_dirty = true; return;
      // Audio Interface
      case 0x4500000: ai_ram = val & 0xfffff8; return;
      case 0x4500004: ai_dma(val); return;
      case 0x4500008: ai_run = val & 0x1; return;
      case 0x450000c: mi_irqs &= ~0x4; return;
      case 0x4500010:
        if ((val & 0xfff) == ai_rate) return;
        ai_rate = val & 0xfff; ai_dirty = true; return;
      case 0x4500014:
        if (((val >> 3) & 0x1) == ai_bits) return;
        ai_bits = (val >> 3) & 0x1; ai_dirty = true; return;
      // Peripheral Interface
      case 0x4600000: pi_ram = val & 0xffffff; return;
      case 0x4600004: pi_rom = val & addr_mask; return;
      case 0x4600008:
        if (pi_cycles) return;
        pi_len = val; pi_write = false;
        pi_cycles = pi_len / 2; return;
      case 0x460000c:
        if (pi_cycles) return;
        pi_len = val; pi_write = true;
        pi_cycles = pi_len / 2; return;
      case 0x4600010: mi_irqs &= ~0x10; return;
      // Serial Interface
      case 0x4800000: si_ram = val & 0xffffff; return;
      case 0x4800004:
        si_update();
        memcpy(pages[0] + si_ram, pages[0xfe] + 0x7c0, 0x40);
        mi_irqs |= 0x2; return;
      case 0x4800010:
        memcpy(pages[0xfe] + 0x7c0, pages[0] + si_ram, 0x40);
        mi_irqs |= 0x2; return;
      case 0x4800018: mi_irqs &= ~0x2; return;
    }
  }

  uint32_t tlb_map(uint32_t addr) {
    // assumes all global, valid, and dirty
    for (uint8_t i = 0; i < 32; ++i) {
      uint32_t mask = ~(tlb[i][0] | 0x1fff);
      if ((addr & mask) == (tlb[i][1] & mask))
        return ((tlb[i][2] << 6) & mask) | (addr & ~mask);
    }
    printf("TLB miss for addr: %x\n", addr);
    return addr;
  }

  template <typename T, bool map>
  int64_t read(uint32_t addr) {
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
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

  template <typename T, bool map>
  void write(uint32_t addr, int64_t val) {
    //if (addr == 0x8014f030 && val == 0x66) val = 0x65;
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
    broke |= watch_w[addr];
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
    return read<uint32_t, true>(addr);
  }

  /* === Actual CPU Functions === */

  uint32_t pc = 0xa4000040;
  uint64_t reg_array[0x63] = {0};
  constexpr uint8_t hi = 0x20, lo = 0x21;
  constexpr uint8_t dev_cop0 = 0x22, dev_cop1 = 0x42;

  void irqs_update(uint32_t cycles) {
    if (R4300::pages[0][0xfd880] == 0xc0)
      printf("Wrong value written to FD880\n");

    // update IP2 based on MI_INTR and MI_MASK
    if (mi_irqs & mi_mask) reg_array[13 + dev_cop0] |= 0x400;
    else reg_array[13 + dev_cop0] &= ~0x400;

    // check ++COUNT against COMPARE, set IP7
    uint64_t &count = reg_array[9 + dev_cop0];
    uint64_t compare = reg_array[11 + dev_cop0];
    if (count > compare) compare += 0x100000000;
    if (count + cycles >= compare) reg_array[13 + dev_cop0] |= 0x8000;
    count = (count + cycles) & 0xffffffff;

    // if interrupt enabled and triggered, set pc to handler address
    uint8_t ip = (reg_array[13 + dev_cop0] >> 8) & 0xff;
    uint8_t im = (reg_array[12 + dev_cop0] >> 8) & 0xff;
    if ((reg_array[12 + dev_cop0] & 0x3) == 0x1 && (ip & im)) {
      printf("Jumping to interrupt instead of %x: %x\n", pc, fetch(pc));
      printf("IP: %x MI: %x MASK: %x VI_INTR %x\n", ip, mi_irqs, mi_mask, vi_irq);
      reg_array[14 + dev_cop0] = pc; pc = 0x80000180; reg_array[12 + dev_cop0] |= 0x2;
    }
  }

  void init(FILE *file) {
    // setup registers (assumes CIC-NUS-6102)
    reg_array[1] = 0x1;
    reg_array[2] = 0xebda536;
    reg_array[3] = 0xebda536;
    reg_array[4] = 0xa536;
    reg_array[5] = 0xffffffffc0f1d859;
    reg_array[6] = 0xffffffffa4001f0c;
    reg_array[7] = 0xffffffffa4001f08;
    reg_array[8] = 0xc0;
    reg_array[10] = 0x40;
    reg_array[11] = 0xffffffffa4000040;
    reg_array[12] = 0xffffffffed10d0b3;
    reg_array[13] = 0x1402a4cc;
    reg_array[14] = 0x2de108ea;
    reg_array[15] = 0x3103e121;
    reg_array[20] = 0x1;
    reg_array[22] = 0x3f;
    reg_array[25] = 0xffffffff9debb54f;
    reg_array[29] = 0xffffffffa4001ff0;
    reg_array[31] = 0xffffffffa4001550;
    reg_array[12 + dev_cop0] = 0x34000000;

    // setup page table
    pages[0] = reinterpret_cast<uint8_t*>(mmap(
      nullptr, 0x20000000, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
    // specially handle SP/DP, DP/MI, VI/AI, PI/RI, and SI ranges
    for (uint32_t i = 0x1; i < 0x100; ++i) {
      if (i < 0x20 || i > 0x24) pages[i] = pages[0] + i * (page_mask + 1);
    }

    // read ROM file into memory
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    fread(pages[0] + 0x10000000, 1, fsize, file);
    memcpy(pages[0] + 0x04000000, pages[0] + 0x10000000, 0x40000);

    // setup VI
    pixels = reinterpret_cast<uint8_t*>(mmap(
      nullptr, 0x200000, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_CreateWindowAndRenderer(640, 480, SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    Vulkan::init();

    // setup RSP
    RSP::dmem = pages[0] + 0x04000000;
    RSP::imem = pages[0] + 0x04001000;
    rsp_cop0[4] = 0x1, rsp_cop0[11] = 0x80;

    // setup SI (assumes CIC-NUS-6102)
    write<uint32_t>(0xbfc007e4, 0x3f3f);
    write<uint32_t>(0xa0000318, 0x800000);
    write<uint32_t>(0xbfc007fc, 0x80);  // pif_status

    // CEN64 does it?
    write<uint32_t>(0xa5000508, 0x5000500);
  }
}

#endif
