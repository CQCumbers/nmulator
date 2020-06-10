#ifndef R4300_H
#define R4300_H

#include <vector>
#include "robin_hood.h"
#include "mipsjit.h"

#include <immintrin.h>
#include "nmulator.h"

#ifdef _WIN32
#  include <SDL.h>
#  undef main
#  include <memory.h>
#  include <errhandlingapi.h>
#else
#  include <SDL2/SDL.h>
#  include <sys/mman.h>
#  include <signal.h>
#endif

namespace R4300 {
  uint8_t *pages[0x20000];
  uint8_t *ram = nullptr;

  uint32_t tlb[0x20][4];
  constexpr uint32_t addr_mask = 0x1fffffff;
  constexpr uint32_t page_mask = 0xfff;

  bool tlb_miss = false;
  uint32_t pc = 0xbfc00000; //0xa4000040;
  uint64_t reg_array[0x63] = {0};
  constexpr uint8_t hi = 0x20, lo = 0x21;
  constexpr uint8_t dev_cop0 = 0x22, dev_cop1 = 0x42;

  uint64_t &count = reg_array[9 + dev_cop0];
  uint64_t &compare = reg_array[11 + dev_cop0];
  uint64_t &status = reg_array[12 + dev_cop0];
  uint64_t &cause = reg_array[13 + dev_cop0];

  template <typename T, bool map>
  int64_t read(uint32_t addr);
  template <typename T, bool map>
  void write(uint32_t addr, int64_t val);

  void joy_update(SDL_Event &event);
  void update();

  /* === MIPS Interface registers === */

  // put into array and read offset without switch?
  constexpr uint32_t mi_version = 0x01010101;
  uint32_t mi_irqs = 0x0, mi_mask = 0x0;

  void set_irqs(uint32_t mask) {
    mi_irqs |= mask;
    if (mi_irqs & mi_mask) {
      cause |= 0x400, cause &= ~0xff;
    }
  }

  void unset_irqs(uint32_t mask) {
    mi_irqs &= ~mask;
    if (!(mi_irqs & mi_mask)) cause &= ~0x400;
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

  robin_hood::unordered_map<uint32_t, Block> blocks;

  void vi_init() {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");
    const uint32_t flags = SDL_WINDOW_ALLOW_HIGHDPI;
    SDL_CreateWindowAndRenderer(640, 480, flags, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
  }

  void convert16(uint8_t *dst, const uint8_t *src, uint32_t len) {
    // convert rgba5551 to bgra8888, 16 byte aligned len
    __m128i imm0xf8 = _mm_set1_epi16(0xf8);
    __m128i shuffle = _mm_set_epi8(
      14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (uint32_t i = 0; i < len; i += 16, src += 16, dst += 32) {
      __m128i data = _mm_loadu_si128((__m128i*)src);
      __m128i pack = _mm_shuffle_epi8(data, shuffle);
      __m128i r = _mm_and_si128(_mm_srli_epi16(pack, 8), imm0xf8);
      __m128i g = _mm_and_si128(_mm_srli_epi16(pack, 3), imm0xf8);
      __m128i b = _mm_and_si128(_mm_slli_epi16(pack, 2), imm0xf8);
      __m128i gb = _mm_or_si128(_mm_slli_epi16(g, 8), b);
      _mm_storeu_si128((__m128i*)dst, _mm_unpacklo_epi16(gb, r));
      _mm_storeu_si128((__m128i*)(dst + 16), _mm_unpackhi_epi16(gb, r));
    }
  }

  void convert32(uint8_t *dst, const uint8_t *src, uint32_t len) {
    // convert rgba8888 to bgra8888, 16 byte aligned len
    __m128i shuffle = _mm_set_epi8(
      15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2);
    for (uint32_t i = 0; i < len; i += 16, src += 16, dst += 16) {
      __m128i data = _mm_loadu_si128((__m128i*)src);
      __m128i pack = _mm_shuffle_epi8(data, shuffle);
      _mm_storeu_si128((__m128i*)dst, pack);
    }
  }

  void vi_update() {
    if (vi_line == vi_irq) set_irqs(0x8);
    vi_line += 0x2, Sched::add(TASK_VI, 6510);

    if (vi_line < vi_height) return;
    bool interlaced = vi_status & 0x40;
    vi_line = interlaced & ~vi_line;
    if (vi_line == 0x1) return;
    uint8_t format = vi_status & 0x3;
    uint32_t height = vi_height >> !interlaced;

    if (vi_dirty) {
      if (texture) SDL_DestroyTexture(texture);
      texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, vi_width, height);
      vi_dirty = false;
    }

    bool fmt16 = (format == 2);
    uint8_t *pixels; int pitch, len = vi_width * height;
    SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);
    if (fmt16) convert16(pixels, ram + vi_origin, len * 2);
    else convert32(pixels, ram + vi_origin, len * 4);
    SDL_UnlockTexture(texture);

    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    for (SDL_Event e; SDL_PollEvent(&e);) {
      if (e.type == SDL_QUIT) exit(0);
      joy_update(e);
    }
  }

  /* === Audio Interface registers === */

  uint32_t ai_status, ai_rate;
  uint32_t ai_ram, ai_len;
  uint32_t ai_start, ai_end;

  bool ai_run, ai_dirty, ai_16bit;
  const uint32_t ntsc_clock = 48681812;
  const uint32_t audio_delay = 2048;
  SDL_AudioDeviceID audio_dev;

  void ai_update() {
    // calculate samples remaining in play buffer
    uint32_t prev_len = ai_len;
    ai_len = SDL_GetQueuedAudioSize(audio_dev) >> ai_16bit;
    ai_len = ai_len > audio_delay ? ai_len - audio_delay : 0;
    if (ai_len > 0) return Sched::move(TASK_AI, 1);
    if (prev_len > 0) set_irqs(0x4);
    // play samples at saved address, set param buffer empty
    if (~ai_status & 0x80000001) return;
    SDL_QueueAudio(audio_dev, ram + ai_start, ai_end);
    ai_len = ai_end >> ai_16bit, ai_status &= ~0x80000001;
    Sched::move(TASK_AI, ai_len << 12);
  }

  void ai_dma(uint32_t len) {
    if (!ai_run || len == 0) return;
    // if AI config changes, update SDL AudioSpec
    if (ai_dirty) {
      SDL_AudioFormat fmt = (ai_16bit ? AUDIO_S16MSB : AUDIO_S8);
      SDL_AudioSpec spec = {
        .freq = (int)(ntsc_clock / (ai_rate + 1)) << !ai_16bit,
        .format = fmt, .channels = 2, .samples = 256,
      };
      audio_dev = SDL_OpenAudioDevice(nullptr, 0, &spec, nullptr, 0);
      SDL_PauseAudioDevice(audio_dev, 0), ai_dirty = false;
    }
    // save DMA params, set param buffer full
    ai_start = ai_ram, ai_end = len & 0x1fff8;
    ai_status |= 0x80000001, ai_update();
  }

  /* === Peripheral & Serial Interface registers === */

  uint32_t pi_status = 0x0, pi_len = 0;
  uint32_t pi_ram = 0x0, pi_rom = 0x0;
  bool pi_write = false;
  int8_t stick_x = 0x0, stick_y = 0x0;
  uint16_t buttons = 0x0;
  uint32_t si_ram = 0x0;
  uint64_t eeprom[0x200], mempack[0x1000];

  void pi_update() {
    //printf("[PI] DMA from %x to %x, of %x bytes\n", pi_ram, pi_rom, pi_len + 1);
    if (pi_write) memcpy(ram + pi_ram, ram + pi_rom, pi_len + 1);
    else memcpy(ram + pi_rom, ram + pi_ram, pi_len + 1);
    set_irqs(0x10); pi_status &= ~0x1;
  }

  void joy_status(uint32_t channel, uint32_t addr) {
    if (channel == 4) return write<uint32_t>(addr, 0x8000ff);
    if (channel != 0) return write<uint8_t>(addr - 2, 0x83);
    write<uint16_t>(addr, 0x0500);  // standard controller type
    write<uint8_t>(addr + 2, 0x0);  // mempack slot
  }

  void joy_read(uint32_t channel, uint32_t addr) {
    if (channel != 0) return write<uint8_t>(addr - 2, 0x84);
    //printf("[SI] buttons: %x stick: %x\n", buttons, stick_y);
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
        case SDLK_o: buttons |= (1 << 3); break;   // C Up
        case SDLK_i: buttons |= (1 << 2); break;   // C Down
        case SDLK_u: buttons |= (1 << 1); break;   // C Left
        case SDLK_p: buttons |= (1 << 0); break;   // C Right
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
        case SDLK_o: buttons &= ~(1 << 3); break;   // C Up
        case SDLK_i: buttons &= ~(1 << 2); break;   // C Down
        case SDLK_u: buttons &= ~(1 << 1); break;   // C Left
        case SDLK_p: buttons &= ~(1 << 0); break;   // C Right
        case SDLK_a: buttons &= ~(1 << 5); break;   // Trigger Left
        case SDLK_s: buttons &= ~(1 << 4); break;   // Trigger Right
        case SDLK_UP: stick_y = 0; break;           // Stick Up
        case SDLK_DOWN: stick_y = 0; break;         // Stick Down
        case SDLK_LEFT: stick_x = 0; break;         // Stick Left
        case SDLK_RIGHT: stick_x = 0; break;        // Stick Right
        case SDLK_q: RDP::dump = true; break;
        case SDLK_w: broke = true; break;
      }
    }
  }

  uint8_t mempack_crc(uint8_t len, uint32_t addr) {
    uint8_t crc = 0;
    for (uint32_t i = 0; i <= len; ++i) {
      for (uint8_t mask = 0x80; mask >= 1; mask >>= 1) {
        uint8_t xor_tap = (crc & 0x80) ? 0x85 : 0x00;
        uint8_t data = read<uint8_t>(addr + i) & mask;
        crc = (crc << 1) | (i < 0x20 && data);
        crc ^= xor_tap;
      }
    }
    return crc;
  }

  void mempack_read(uint8_t len, uint32_t addr) {
    uint16_t mem = read<uint16_t>(addr) & ~0x1f;
    for (uint32_t i = 0; i < len - 1; i += 8) {
      if (mem + i >= 0x8000) break;
      write<uint64_t>(addr + 2 + i, mempack[(mem + i) / 8]);
    }
    write<uint8_t>(addr + len + 1, mempack_crc(addr + 2, len - 1));
  }

  void mempack_write(uint8_t len, uint32_t addr) {
    uint16_t mem = read<uint16_t>(addr) & ~0x1f;
    for (uint32_t i = 0; i < len - 2; i += 8) {
      if (mem + i >= 0x8000) break;
      mempack[(mem + i) / 8] = read<uint64_t>(addr + 2 + i);
    }
    write<uint8_t>(addr + len, mempack_crc(addr + 2, len - 2));
  }

  void eeprom_read(uint8_t len, uint32_t &addr) {
    uint8_t mem = read<uint8_t>(addr);
    for (uint32_t i = 0; i < len; i += 8)
      write<uint64_t>(addr + 1 + i, eeprom[mem + i / 8]);
  }

  void eeprom_write(uint8_t len, uint32_t addr) {
    uint8_t mem = read<uint8_t>(addr);
    for (uint32_t i = 0; i < len - 1; i += 8)
      eeprom[mem + i / 8] = read<uint64_t>(addr + 1 + i);
    write<uint8_t>(addr + len, 0);
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
      switch (t -= 1, read<uint8_t>(pc++)) {
        case 0x00: joy_status(channel, pc); break;
        case 0x01: joy_read(channel, pc); break;
        case 0x02: mempack_read(r, pc); break;
        case 0x03: mempack_write(t, pc); break;
        case 0x04: eeprom_read(r, pc); break;
        case 0x05: eeprom_write(t, pc); break;
        case 0xff: joy_status(channel, pc); break;
      }
      pc += t + r, ++channel;
    }
  }

  /* === Reading and Writing === */

  template <typename T>
  int64_t mmio_read(uint32_t addr) {
    switch (addr & addr_mask) {
      default: /*printf("[MMIO] read from %x\n", addr);*/ return 0;
      // RSP Interface
      case 0x4040000: return RSP::cop0[0];
      case 0x4040004: return RSP::cop0[1];
      case 0x4040010: return RSP::cop0[4];
      case 0x404001c: return (RSP::cop0[7] ? 0x1 : RSP::cop0[7]++);
      case 0x4080000: return RSP::pc & 0xffc;
      // RDP Interface
      case 0x4100000: return RSP::cop0[8];
      case 0x4100004: return RSP::cop0[9];
      case 0x4100008: return RSP::cop0[10];
      case 0x410000c: return RSP::cop0[11];
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
      case 0x4500004: return (ai_status & 0x1 ? ai_len : 0);
      case 0x450000c: return ai_status;
      // Peripheral Interface
      case 0x4600000: return pi_ram;
      case 0x4600004: return pi_rom;
      case 0x4600010: return pi_status;
      // RDRAM Interface
      case 0x4700000: return 0xe;      // RI_MODE
      case 0x4700004: return 0x40;     // RI_CONFIG
      case 0x470000c: return 0x14;     // RI_SELECT
      case 0x4700010: return 0x63634;  // RI_REFRESH
      // Serial Interface
      case 0x4800018: return (mi_irqs & 0x2) << 11;
    }
  }

  template <typename T>
  void mmio_write(uint32_t addr, uint32_t val) {
    switch (addr & addr_mask) {
      default: /*printf("[MMIO] write to %x: %x\n", addr, val);*/ return;
      // RSP Interface
      case 0x4040000: RSP::cop0[0] = val & 0x1fff; return;
      case 0x4040004: RSP::cop0[1] = val & 0xffffff; return;
      case 0x4040008: RSP::dma(val, false); return;
      case 0x404000c: RSP::dma(val, true); return;
      case 0x4040010: RSP::set_status(val); return;
      case 0x404001c: RSP::cop0[7] = 0x0; return;
      case 0x4080000: RSP::pc = val & 0xffc; return;
      // RDP Interface
      case 0x4100000:
        // set RDP_PC_START
        RSP::cop0[8] = val & addr_mask;
        RSP::cop0[10] = RSP::cop0[8]; return;
      case 0x4100004:
        // set RDP_PC_END
        Sched::add(TASK_RDP, 0);
        RSP::cop0[9] = val & addr_mask; return;
      case 0x410000c:
        // update RDP_STATUS
        RSP::cop0[11] &= ~pext(val >> 0, 0x7);
        RSP::cop0[11] |= pext(val >> 1, 0x7); return;
      // MIPS Interface
      case 0x4300000:
        if (val & 0x800) unset_irqs(0x20); return;
      case 0x430000c:
        mi_mask &= ~pext(val >> 0, 0x3f);
        mi_mask |= pext(val >> 1, 0x3f);
        if (mi_irqs & mi_mask) {
          cause |= 0x400, cause &= ~0xff;
        } else cause &= ~0x400; return;
      // Video Interface
      case 0x4400000: 
        if (val == vi_status) return;
        vi_status = val, vi_dirty = true; return;
      case 0x4400004: vi_origin = val & 0xffffff; return;
      case 0x4400008:
        if ((val & 0xfff) == vi_width) return;
        vi_width = val & 0xfff, vi_dirty = true; return;
      case 0x440000c: vi_irq = val & 0x3ff; return;
      case 0x4400010: unset_irqs(0x8); return;
      case 0x4400018:
        if ((val & 0x3ff) == vi_height) return;
        vi_height = val & 0x3ff, vi_dirty = true; return;
      // Audio Interface
      case 0x4500000: ai_ram = val & 0xfffff8; return;
      case 0x4500004: ai_dma(val); return;
      case 0x4500008: ai_run = val & 0x1; return;
      case 0x450000c: unset_irqs(0x4); return;
      case 0x4500010:
        if ((val & 0xfff) == ai_rate) return;
        ai_rate = val & 0xfff, ai_dirty = true; return;
      case 0x4500014:
        if (((val >> 3) & 0x1) == ai_16bit) return;
        ai_16bit = (val >> 3) & 0x1, ai_dirty = true; return;
      // Peripheral Interface
      case 0x4600000: pi_ram = val & 0xffffff; return;
      case 0x4600004: pi_rom = val & addr_mask; return;
      case 0x4600008:
        if (pi_status & 0x1) return;
        pi_len = val, pi_write = false;
        Sched::add(TASK_PI, 0);
        pi_status |= 0x1; return;
      case 0x460000c:
        if (pi_status & 0x1) return;
        pi_len = val, pi_write = true;
        Sched::add(TASK_PI, 0);
        pi_status |= 0x1; return;
      case 0x4600010: unset_irqs(0x10); return;
      // Serial Interface
      case 0x4800000: si_ram = val & 0xffffff; return;
      case 0x4800004: si_update();
        memcpy(ram + si_ram, ram + 0x1fc007c0, 0x40);
        set_irqs(0x2); return;
      case 0x4800010:
        memcpy(ram + 0x1fc007c0, ram + si_ram, 0x40);
        set_irqs(0x2); return;
      case 0x4800018: unset_irqs(0x2); return;
    }
  }

  uint32_t tlb_map(uint32_t addr) {
    // assumes all global, valid, and dirty
    printf("COP0 2: %llx, 3: %llx\n", reg_array[2 + dev_cop0], reg_array[3 + dev_cop0]);
    printf("Checking TLB for addr: %x\n", addr);
    for (uint32_t i = 0; i < 32; ++i) {
      uint32_t mask = ~(tlb[i][0] | 0x1fff);
      printf("TLB[%x] with mask: %x, vaddr: %x, paddr0: %x, paddr1: %x\n",
        i, mask, tlb[i][1] & mask, (tlb[i][2] << 6) & mask, (tlb[i][3] << 6) & mask);
      if ((addr & mask) == (tlb[i][1] & mask)) {
        mask = (mask >> 1) | 0x80000000;
        bool odd = (addr & -mask) != 0;
        uint32_t page = tlb[i][2 + odd] << 6;
        return (page & mask) | (addr & ~mask);
      }
    }
    tlb_miss = true;
    // load COP0 context, bad_vaddr, entry_hi
    uint64_t &ctx = reg_array[4 + dev_cop0];
    ctx &= ~0x7ffff0, ctx |= (addr >> 9) & 0x7ffff0;
    reg_array[8 + dev_cop0] = addr;
    reg_array[10 + dev_cop0] = addr & 0xffffe000;
    printf("TLB miss for addr: %x\n", addr);
    return addr;
  }

  template <typename T, bool map>
  int64_t read(uint32_t addr) {
    //printf("Reading from %x\n", addr);
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
    uint8_t *page = pages[(addr >> 12) & 0x1ffff];
    if ((addr & addr_mask) == 0x1fc007e4) {
      ram[0x1fc007ff] = 0x80;  // Hacky CIC emulation
    }
    if (!page) return mmio_read<T>(addr);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: return *ptr;
      case 2: return static_cast<T>(bswap16(*ptr));
      case 4: return static_cast<T>(bswap32(*ptr));
      case 8: return static_cast<T>(bswap64(*ptr));
    }
  }

  template <typename T, bool map>
  void write(uint32_t addr, int64_t val) {
    if (map && addr >> 30 != 0x2) addr = tlb_map(addr);
    //broke |= watch_w[addr];
    uint8_t *page = pages[(addr >> 12) & 0x1ffff];
    if (!page) return mmio_write<T>(addr, val);
    T *ptr = reinterpret_cast<T*>(page + (addr & page_mask));
    switch (sizeof(T)) {
      case 1: *ptr = val; return;
      case 2: *ptr = bswap16(val); return;
      case 4: *ptr = bswap32(val); return;
      case 8: *ptr = bswap64(val); return;
    }
  }

  uint32_t fetch(uint32_t addr) {
    return read<uint32_t, true>(addr);
  }

  /* === Actual CPU Functions === */

  //robin_hood::unordered_node_map<uint32_t, Block> blocks;
  robin_hood::unordered_map<uint32_t, std::vector<uint32_t>> prot_pages;
  const uint32_t hpage_mask = addr_mask & ~0xfff;
  bool modified = false; Block *block = &empty;

  void timer_fire() {
    cause |= 0x8000, cause &= ~0xff;
  }

  void timer_update() {
    uint32_t cycles = (compare - count) * 2;
    Sched::move(TASK_TIMER, cycles);
  }

#ifdef _WIN32
  uint8_t *alloc_pages(uint32_t size) {
    return reinterpret_cast<uint8_t*>(VirtualAlloc(
      nullptr, size, MEM_COMMIT, PAGE_READWRITE
    ));
  }

  void protect(uint32_t hpage) {
    DWORD old;
    if (prot_pages[hpage].empty()) {
      VirtualProtect(ram + hpage, 0x1000, PAGE_READONLY, &old);
    }
    prot_pages[hpage].push_back(pc & addr_mask);
  }

  void unprotect(uint32_t hpage) {
    DWORD old;
    for (uint32_t addr : prot_pages[hpage]) blocks[addr].code = nullptr;
    VirtualProtect(ram + hpage, 0x1000, PAGE_READWRITE, &old);
    prot_pages[hpage].clear(); modified = true;
  }
  
  LONG WINAPI handle_fault(_EXCEPTION_POINTERS *info) {
    DWORD sig = info->ExceptionRecord->ExceptionCode;
    if (sig != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *addr = (uint8_t*)info->ExceptionRecord->ExceptionInformation[1];
    int64_t hpage = addr - ram;
    if (!(0 <= hpage && hpage <= addr_mask)) exit(1);
    unprotect(hpage & hpage_mask); return EXCEPTION_CONTINUE_EXECUTION;
  }

  void setup_fault_handler() {
    AddVectoredExceptionHandler(true, handle_fault);
  }
#else
  uint8_t *alloc_pages(uint32_t size) {
    return reinterpret_cast<uint8_t*>(mmap(
      nullptr, size, PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, 0, 0
    ));
  }

  void protect(uint32_t hpage) {
    if (prot_pages[hpage].empty())
      mprotect(ram + hpage, 0x1000, PROT_READ);
    prot_pages[hpage].push_back(pc & addr_mask);
  }

  void unprotect(uint32_t hpage) {
    for (uint32_t addr : prot_pages[hpage]) blocks[addr].code = nullptr;
    mprotect(ram + hpage, 0x1000, PROT_READ | PROT_WRITE);
    prot_pages[hpage].clear(); modified = true;
  }

  void handle_fault(int sig, siginfo_t *info, void *raw_ctx) {
    if (sig != SIGBUS && sig != SIGSEGV) return;
    int64_t hpage = (uint8_t*)info->si_addr - ram;
    if (!(0 <= hpage && hpage <= addr_mask)) exit(1);
    unprotect(hpage & hpage_mask);
  }

  void setup_fault_handler() {
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_sigaction = handle_fault;
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    sigaction(SIGBUS, &act, nullptr);
  }
#endif

  /* === Debugger interface === */

  bool broke, step;
  robin_hood::unordered_map<uint32_t, bool> breaks;

  // read registers in gdb MIPS order
  uint64_t read_reg(uint32_t idx) {
    if (idx < 32) return reg_array[idx];
    if (idx >= 38) return reg_array[idx - 38 + dev_cop1];
    switch (idx) {
      default: printf("Invalid reg %x\n", idx), exit(1);
      case 32: return reg_array[12 + dev_cop0];
      case 33: return reg_array[33];
      case 34: return reg_array[32];
      case 35: return reg_array[8 + dev_cop0];
      case 36: return reg_array[13 + dev_cop0];
      case 37: return pc;
    }
  }

  uint64_t read_mem(uint32_t addr) {
    return read<uint64_t>(addr);
  }
 
  // create or delete breakpoint
  void set_break(uint32_t addr, bool active) {
    breaks[addr] = active;
  }

  // ignore breakpoint if just stopped
  bool get_break(uint32_t addr) {
    if (broke) return broke = false;
    return broke = step || breaks[addr];
  }

  const DbgConfig dbg_config = {
    .read_reg = read_reg,
    .read_mem = read_mem,
    .set_break = set_break
  };

  void init_debug(int port) {
    Debugger::init(port);
    step = Debugger::update(&dbg_config);
  }

  /* === Recompiler interface === */

  void update() {
    while (Sched::until >= 0) {
      if (block->code) {
        pc = block->code();
        if (broke) {
          step = Debugger::update(&dbg_config);
          blocks.clear();
          memset(block->next_pc, 0, 64);
          memset(block->next, 0, 64);
        }
        uint8_t line = (pc >> 2) & 0x7;
        if (block->next_pc[line] != pc) {
          block->next_pc[line] = pc;
          block->next[line] = &blocks[pc & addr_mask];
        }
        block = block->next[line];
      }
      
      if (!block->code) {
        CodeHolder code; 
        code.init(runtime.codeInfo());
        MipsJit<Device::r4300> jit(code);
        jit.jit_block();
        runtime.add(&block->code, &code);
      }
    }
    Sched::add(TASK_R4300, 0);
  }

  uint32_t crc32(uint8_t *bytes, uint32_t len) {
    uint32_t crc = 0, *msg = (uint32_t*)bytes;
    for (uint32_t i = 0; i < len / 4; ++i)
      crc = _mm_crc32_u32(crc, msg[i]);
    return crc;
  }

  void init(const char *filename) {
    // allocate memory, setup change detection
    ram = alloc_pages(0x20000000);
    setup_fault_handler();
    for (uint32_t i = 0x0; i < 0x20000; ++i) {
      if (i >= 0x3f00 && i < 0x4000) continue;
      if (i >= 0x4002 && i < 0x5000) continue;
      pages[i] = ram + i * (page_mask + 1);
    }

    // read ROM file into memory
    FILE *rom = fopen(filename, "r");
    if (!rom) printf("Can't find rom %s\n", filename), exit(1);
    fread(ram + 0x10000000, 1, 0x4000000, rom), fclose(rom);
    memcpy(ram + 0x4000000, ram + 0x10000000, 0x40000);

    // read PIF boot rom, setup ram based on CIC
    FILE *pifrom = fopen("pifdata.bin", "r");
    if (!pifrom) printf("Can't find pifdata.bin\n"), exit(1);
    fread(ram + 0x1fc00000, 1, 0x7c0, pifrom), fclose(pifrom);

    write<uint32_t>(0xa0000318, 0x800000);  // 8MB RDRAM
    switch (crc32(ram + 0x04000040, 0xfc0)) {
      case 0x583af077: write<uint32_t>(0xbfc007e4, 0x43f3f); break;  // 6101
      case 0x98a02fa9: write<uint32_t>(0xbfc007e4, 0x3f3f); break;   // 6102
      case 0x04e7fe6d: write<uint32_t>(0xbfc007e4, 0x783f); break;   // 6103
      case 0x035e73e4: write<uint32_t>(0xbfc007e4, 0x913f); break;   // 6105
      case 0x0f727fb1: write<uint32_t>(0xbfc007e4, 0x853f); break;   // 6106
      default: printf("No compatible CIC chip found\n"), exit(1);
    }

    // setup other components
    vi_init(), RDP::init();
    RSP::init(ram + 0x04000000);
  }

  template void write<uint32_t, false>(uint32_t addr, int64_t val);
}

#endif
