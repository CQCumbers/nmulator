/* === Bitfield Flags === */

#define M0_FILL      0x00300000
#define M0_COPY      0x00200000
#define M0_2CYCLE    0x00100000
#define M0_PERSP     0x00080000
#define M0_TLUT_EN   0x00008000
#define M0_TLUT_IA   0x00004000
#define M0_DITH      0x000000c0

#define M1_BLEND     0x00004000
#define M1_CVG2ALPHA 0x00002000
#define M1_ALPHA2CVG 0x00001000
#define M1_ZMODE     0x00000c00
#define ZINTRA       0x00000400
#define ZTRANS       0x00000800
#define ZDECAL       0x00000c00
#define M1_ON_CVG    0x00000080
#define M1_ZWRITE    0x00000020
#define M1_ZCMP      0x00000010
#define M1_AA        0x00000008
#define M1_ZSRC      0x00000004
#define M1_ALPHA     0x00000001

#define T_ZBUF       0x01
#define T_TEX        0x02
#define T_SHADE      0x04
#define T_RECT       0x08
#define T_LMAJOR     0x10

/* === Struct Definitions === */

struct TileData {
  uint cmd_idxs[64];
};

struct GlobalData {
  uint width, size;
  uint n_cmds, fmt;
};

struct RDPCommand {
  uint type, tile, pad;
  int xh, xm, xl;
  int yh, ym, yl;
  int sh, sm, sl;

  int4 shade, sde, sdx;
  int4 tex, tde, tdx;

  int sxh, sxl, syh, syl;
  uint modes[2], mux[2];
  uint tlut, tmem, texes;
  uint fill, fog, blend;
  uint env, prim, lodf;
  uint zprim, keys, keyc;
};

struct RDPTex {
  uint format, size;
  uint width, addr;
  int2 sth, stl, shift;
  uint pal, pad;
};

/* === Resource Buffers === */

[[vk::binding(0)]] StructuredBuffer<RDPCommand> cmds;
[[vk::binding(1)]] StructuredBuffer<TileData> tiles;
[[vk::binding(2)]] StructuredBuffer<RDPTex> texes;
[[vk::binding(3)]] StructuredBuffer<GlobalData> globals;

[[vk::binding(4)]] StructuredBuffer<uint16_t> tmem;
[[vk::binding(5)]] RWStructuredBuffer<uint16_t> cbuf;
[[vk::binding(6)]] RWStructuredBuffer<uint16_t> hcbuf;
[[vk::binding(7)]] RWStructuredBuffer<uint16_t> zbuf;
[[vk::binding(8)]] RWStructuredBuffer<uint16_t> hzbuf;

/* === Common utilities === */

uint16_t bswap16(uint val) {
  return uint16_t((val << 8) | (val >> 8));
}

uint pack32(int4 color) {
  uint4 c = clamp(color, 0, 0xff);
  return (c.w << 24) | (c.z << 16) | (c.y << 8) | c.x;
}

int4 unpack32(uint color) {
  int r = color >>  0, g = color >>  8;
  int b = color >> 16, a = color >> 24;
  return int4(r, g, b, a) & 0xff;
}

uint a(uint color) {
  return (color >> 24) * 0x01010101;
}

void seta(inout uint color, uint alpha) {
  color = (color & 0x00ffffff) | (alpha & 0xff000000);
}

int abs1(int val) {
  return (val >> 31) ^ val;
}

/* === Framebuffer interface === */

void load(uint2 pos, out uint color, out uint depth) {
  uint idx = pos.y * globals[0].width + pos.x;
  if (globals[0].fmt) {
    // load 24-bit color, 8-bit cvg
    color  = cbuf[idx * 2 + 0] <<  0;
    color |= cbuf[idx * 2 + 1] << 16;
  } else {
    // load 15-bit color, 3-bit cvg
    uint cval = bswap16(cbuf[idx]);
    color = uint(hcbuf[idx]) << 29;
    color |= (cval & 0x0001) << 31;
    color |= (cval & 0x003e) << 18;
    color |= (cval & 0x07c0) <<  5;
  }

  // load 14-bit depth, 4-bit dz
  uint zval = bswap16(zbuf[idx]);
  int mant = (zval >> 2) & 0x7ff;
  int expn = (zval >> 13) & 0x07;
  uint base = 0x40000 - (0x40000 >> expn);
  depth  = (mant << max(6 - expn, 0)) + base;
  depth |= (zval << 30) | (uint(hzbuf[idx]) << 28);
}

void store(uint2 pos, uint color, uint depth) {
  uint idx = pos.y * globals[0].width + pos.x;
  if (globals[0].fmt) {
    // store 24-bit color, 8-bit cvg
    cbuf[idx * 2 + 0] = uint16_t(color >>  0);
    cbuf[idx * 2 + 1] = uint16_t(color >> 16);
  } else {
    // store 15-bit color, 3-bit cvg
    uint cval = (color >> 31) & 0x1;
    cval |= (color >> 18) & 0x003e;
    cval |= (color >>  5) & 0x07c0;
    cval |= (color <<  8) & 0xf800;
    cbuf[idx] = bswap16(cval);
    hcbuf[idx] = uint16_t(color >> 29) & 3;
  }

  // store 14-bit depth, 4-bit dz
  int expn = (depth ^ 0x3ffff) | 0x400;
  expn = 31 - firstbithigh(expn << 14);
  int mant = (depth >> max(6 - expn, 0)) & 0x7ff;
  uint zval = (expn << 13) | (mant << 2);
  zbuf[idx] = bswap16(zval | (depth >> 30));
  hzbuf[idx] = uint16_t(depth >> 28) & 3;
}

/* === Format conversions === */

uint read_rgba16(uint enc) {
  // bswapped RGBA5551 to ABGR8888
  enc = ((enc << 8) | (enc >> 8)) & 0xffff;
  uint dec = -(enc & 0x1) & (0xff << 24);
  dec |= (enc & 0x3e) << 18;
  dec |= (enc & 0x7c0) << 5;
  return dec | ((enc & 0xf800) >> 8);
}

uint read_ia16(uint enc) {
  // 8 bit intensity, 8 bit alpha
  uint i = enc & 0xff, a = enc >> 8;
  return (a << 24) | (i << 16) | (i << 8) | i;
}

uint read_i4(uint enc) {
  // 4 bit intensity
  uint i = (enc << 4) | enc;
  return (i << 24) | (i << 16) | (i << 8) | i;
}

/* === Utility Functions === */

uint tmem4r(uint pos, uint off, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  uint val = tmem[pos >> 1];
  uint shift = ((off & 0x3) ^ 0x1) * 4;
  return (val >> shift) & 0xf;
}

uint tmem8r(uint pos, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  uint val = tmem[pos >> 1];
  uint shift = (pos & 0x1) * 8;
  return (val >> shift) & 0xff;
}

uint tmem16r(uint pos, uint idx) {
  pos = (pos & 0xfff) | (idx << 12);
  return tmem[pos >> 1];
}

uint texel(uint2 st, uint2 mask, RDPTex tex, RDPCommand cmd) {
  // apply mirror/mask to ST coords
  if (cmd.modes[0] & M0_TLUT_EN) tex.format = 2;
  bool2 mst = tex.shift & 0x100, en = (mask != 0);
  st ^= -(en & (st >> mask) & mst), st &= (en << mask) - 1;
  // see manual 13.8 for tmem layouts
  uint s = st.x, t = st.y, wd = tex.width;
  uint flip = (st.y & 0x1) << 2;
  bool ia = cmd.modes[0] & M0_TLUT_IA;
  // switch on format and size
  if (tex.format == 0 && tex.size == 0) {        // 4 bit RGBA
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    return read_i4(tmem4r(pos, s, cmd.tmem));
  } else if (tex.format == 0 && tex.size == 1) { // 8 bit RGBA
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint i = tmem8r(pos, cmd.tmem);
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 0 && tex.size == 2) { // 16 bit RGBA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    return read_rgba16(tmem16r(pos, cmd.tmem));
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint hi = tmem16r(pos | 0x800, cmd.tmem);
    uint lo = tmem16r(pos & 0x7ff, cmd.tmem);
    return (hi << 16) | (lo & 0xffff);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    uint idx = (tex.pal << 4) | tmem4r(pos & 0x7ff, s, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return idx;
    uint entry = tmem16r(0x800 | (idx << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint idx = tmem8r(pos & 0x7ff, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return idx;
    uint entry = tmem16r(0x800 | (idx << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 2) {                  // 32 bit CI
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint idx = tmem16r(pos & 0x7ff, cmd.tmem);
    if (~cmd.modes[0] & M0_TLUT_EN) return (idx << 16) | idx;
    uint entry = tmem16r(0x800 | ((idx >> 8) << 3), cmd.tmem);
    return ia ? read_ia16(entry) : read_rgba16(entry);
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    uint val = tmem4r(pos, s, cmd.tmem);
    uint i = val & 0xe, a = (val & 0x1) * 0xff;
    i = (i << 4) | (i << 1) | (i >> 2);
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint val = tmem8r(pos, cmd.tmem);
    uint i = (val & 0xf0) | (val >> 4);
    uint a = (val << 4) | (val & 0x0f);
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    return read_ia16(tmem16r(pos, cmd.tmem));
  } else if (tex.format == 3 && tex.size == 3) { // 32 bit IA
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint val = tmem16r(pos, cmd.tmem);
    return (val << 16) | (val & 0xffff);
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    uint pos = (tex.addr + t * wd + s / 2) ^ flip;
    return read_i4(tmem4r(pos, s, cmd.tmem));
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    uint pos = (tex.addr + t * wd + s) ^ flip;
    uint i = tmem8r(pos, cmd.tmem);
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4) {                  // 32 bit I
    uint pos = (tex.addr + t * wd + s * 2) ^ flip;
    uint val = tmem16r(pos, cmd.tmem);
    return (val << 16) | (val & 0xffff);
  }
  return 0xffff00ff;
}

/* === Pipeline Stages === */

uint read_noise(uint seed) {
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9, seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d, seed = seed ^ (seed >> 15);
  uint i = seed & 0xff;
  return 0xff000000 | (i << 16) | (i << 8) | i;
}

int4 quantize_x(int4 x) {
  bool4 sticky = (x & 0xfff) != 0;
  return (x >> 12) | sticky;
}

uint visible(uint2 pos, RDPCommand cmd, out int2 dxy, out uint cvbit) {
  // check pixel within y bounds
  int y1 = max(cmd.yh, cmd.syh);
  int y2 = min(cmd.yl, cmd.syl);
  if (!(y1 / 4 <= pos.y && pos.y <= y2 / 4)) return 0;
  // compute edges at each sub-scanline
  int4 y = pos.y * 4 + int4(0, 1, 2, 3);
  int4 x1 = cmd.xh + cmd.sh * (y - (cmd.yh & ~3));
  int4 xm = cmd.xm + cmd.sm * (y - (cmd.yh & ~3));
  int4 xl = cmd.xl + cmd.sl * (y - cmd.ym);
  // get horizontal bounds, quantize
  int4 x2 = y < (int4)cmd.ym ? xm : xl;
  bool lft = cmd.type & (T_LMAJOR | T_RECT);
  int4 xa = quantize_x(lft ? x1 : x2);
  int4 xb = quantize_x(lft ? x2 : x1);
  // compute valid sub-scanlines
  bool4 vx = (xa / 2) <= (xb / 2);
  bool4 vy = vx && (int4)y1 <= y && y < (int4)y2;
  // compute interpolation deltas
  bool offset = lft == ((cmd.sh >> 31) & 1);
  if (offset) x1.x += 3 * cmd.sh;
  dxy = pos * 256 - int2(x1.x >> 7, cmd.yh >> 2 << 8);
  if (offset) dxy.y += 3 << 6;
  // apply scissor boundaries
  int4 sxh = (uint4)cmd.sxh * 2;
  int4 sxl = (uint4)cmd.sxl * 2;
  xa = min(max(xa, sxh), sxl);
  xb = min(max(xb, sxh), sxl);
  // check x bounds at every other subpixel
  int4 x = pos.x * 8 + int4(0, 2, 0, 2);
  cvbit = vy.x && xa.x <= x.x && x.x < xb.x;
  uint cvg = dot(xa <= x + 4 && x + 4 < xb, vy);
  return cvg + dot(xa <= x && x < xb, vy);
}

int roundz(int depth) {
  int expn = (depth ^ 0x3ffff) | 0x400;
  expn = 31 - firstbithigh(expn << 14);
  int mant = (depth >> max(6 - expn, 0)) & 0x7ff;
  uint zval = (expn << 13) | (mant << 2);

  uint base = 0x40000 - (0x40000 >> expn);
  return (mant << max(6 - expn, 0)) + base;
}

int2 calc_depth(int z_in, uint depth, uint cvg, uint pixel, RDPCommand cmd) {
  if (~cmd.modes[1] & M1_ZCMP) return int2(1, depth);
  // extract old/new z and dz
  bool zsrc = cmd.modes[1] & M1_ZSRC;
  int oz = roundz(depth & 0x3ffff), odz = depth >> 28;
  int nz = zsrc ? (cmd.zprim >> 16) * 8 : z_in >> 13;
  int ndz = zsrc ? cmd.zprim / 2 : abs1(cmd.tdx.w >> 16);
  ndz = firstbithigh(ndz & 0x7fff) + 1;
  // adjust dz at greater depth
  int expn = (depth ^ 0x3ffff) | 0x400;
  expn = 31 - firstbithigh(expn << 14);
  bool dzm = (expn < 3 && odz == 0xf);
  if (expn < 3) odz = max(odz + 1, 4 - expn);
  int dz = 0x8 << max(ndz, odz);
  // compare new z with old z
  bool ovf = cvg + (pixel >> 29) > 7;
  bool zmx = oz == 0x3ffff;
  bool zle = dzm || nz - dz <= oz;
  bool zge = dzm || nz + dz >= oz;
  bool zlt = zmx || (ovf ? nz < oz : zle);
  bool zeq = ovf && nz < oz && zge;
  // depth test based on mode
  uint mode = cmd.modes[1] & M1_ZMODE;
  if (mode == ZINTRA && zeq) zlt = true;
  if (mode == ZTRANS) zlt = zmx || nz < oz;
  if (mode == ZDECAL) zlt = !zmx && zge && zle;
  return int2(zlt, (ndz << 28) | nz);
}

uint calc_shade(int2 dxy, RDPCommand cmd) {
  if (~cmd.type & T_SHADE) return 0;
  int4 rgba = cmd.shade + (dxy.y / 256 * cmd.sde) + (dxy.x / 256 * cmd.sdx);
  rgba += cmd.sde / 256 * (dxy.y & 0xff) + cmd.sdx / 256 * (dxy.x & 0xff);
  return pack32(rgba >> 16);
}

int4 calc_coord(int2 dxy, RDPCommand cmd) {
  int4 stwz = cmd.tex + (dxy.y / 256 * cmd.tde) + (dxy.x / 256 * cmd.tdx);
  stwz += cmd.tde / 256 * (dxy.y & 0xff) + cmd.tdx / 256 * (dxy.x & 0xff);
  if (cmd.modes[0] & M0_PERSP) stwz.xy *= 32768.0 / (float2)(stwz.z >> 16);
  stwz.w = clamp(stwz.w, 0, 0x3ffff << 13); return stwz;
}

uint sample_tex(int2 st_in, RDPCommand cmd, uint cycle) {
  uint cyc = cmd.modes[0] & M0_FILL;
  if (~cmd.type & T_TEX) return 0;
  if (cyc != M0_2CYCLE && cycle) return 0;
  // read tex attributes, shift coords
  RDPTex tex = texes[cmd.tmem * 8 + cmd.tile + cycle];
  int2 shl = tex.shift & 0xf, st = st_in >> 16;
  st = shl > 10 ? st << (16 - shl) : st >> shl;
  // get mask, clamp coords to tex bounds
  uint2 mask = min((tex.shift >> 4) & 0xf, 10);
  bool2 clmp = (tex.shift >> 9) || (mask == 0);
  int2 stl = tex.stl << 3, sth = tex.sth << 3;
  st = clmp ? clamp(st - stl, 0, sth - stl) : st - stl;
  // read appropriate texels
  int2 stfrac = st & 0x1f, sti = st >> 5;
  bool up = (stfrac.x + stfrac.y) & 0x20;
  int4 c1 = unpack32(texel(sti + (int2)(up), mask, tex, cmd));
  int4 c2 = unpack32(texel(sti + int2(1, 0), mask, tex, cmd));
  int4 c3 = unpack32(texel(sti + int2(0, 1), mask, tex, cmd));
  // 3-tap bilinear filter
  stfrac = up ? 0x20 - stfrac.yx : stfrac.xy;
  int4 mix = stfrac.x * (c2 - c1) + stfrac.y * (c3 - c1);
  return pack32(c1 + (mix + 0x10) / 0x20);
}

uint fill(RDPCommand cmd) {
  if (globals[0].fmt) return cmd.fill;
  else return read_rgba16(cmd.fill);
}

uint combine(uint tex0, uint tex1, uint shade, uint rand,
    inout uint cvg, uint color, RDPCommand cmd, uint cycle) {
  uint cyc = cmd.modes[0] & M0_FILL;
  if (cyc != M0_2CYCLE && !cycle) return 0;
  if (cyc == M0_COPY) return tex0;
  if (cyc == M0_FILL) return fill(cmd);
  // organize combiner inputs
  uint prim = cmd.prim, env = cmd.env, lod = cmd.lodf;
  uint mux0[16] = { color, tex0, tex1, prim, shade, env, -1, rand, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint mux1[16] = { color, tex0, tex1, prim, shade, env, cmd.keyc, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint mux2[32] = { color, tex0, tex1, prim, shade, env, cmd.keys, 0, a(tex0), a(tex1),
    a(prim), a(shade), a(env), -1, a(lod), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  uint mux3[8] = { color, tex0, tex1, prim, shade, env, -1, 0 };
  uint mux4[8] = { -1, tex0, tex1, prim, shade, env, lod, 0 };
  // get color combiner setting
  uint m0 = cmd.mux[0], m1 = cmd.mux[1];
  uint c0 = mux0[(cycle ? m0 >>  5 : m0 >> 20) & 0x0f];
  uint c1 = mux1[(cycle ? m1 >> 24 : m1 >> 28) & 0x0f];
  uint c2 = mux2[(cycle ? m0 >>  0 : m0 >> 15) & 0x1f];
  uint c3 = mux3[(cycle ? m1 >>  6 : m1 >> 15) & 0x07];
  // get alpha combiner setting
  seta(c0, mux3[(cycle ? m1 >> 21 : m0 >> 12) & 0x07]);
  seta(c1, mux3[(cycle ? m1 >>  3 : m1 >> 12) & 0x07]);
  seta(c2, mux4[(cycle ? m1 >> 18 : m0 >>  9) & 0x07]);
  seta(c3, mux3[(cycle ? m1 >>  0 : m1 >>  9) & 0x07]);
  // evaluate combiner equation
  int4 a = unpack32(c0), b = unpack32(c1);
  int4 c = unpack32(c2), d = unpack32(c3);
  return pack32(((a - b) * c + 0x80) / 256 + d);
}

uint mix_alpha_cvg(uint color, inout uint cvg, RDPCommand cmd) {
  uint cvg8 = cvg << 5, m1 = cmd.modes[1];
  uint alph = (color >> 24) + ((color >> 24) + 1) / 256;
  if (m1 & M1_ALPHA2CVG) cvg8 = (alph * cvg + 4) / 8;
  if (m1 & M1_CVG2ALPHA) seta(color, min(cvg8, 255) << 24);
  cvg = cvg8 >> 5; return color;
}

uint blend(uint pixel, uint color, uint shade, int2 zcmp,
    out uint depth, uint cvg, uint cvbit, RDPCommand cmd, uint cycle) {
  uint cyc = cmd.modes[0] & M0_FILL, m1 = cmd.modes[1];
  if (cyc != M0_2CYCLE && cycle) return pixel;
  // alpha, coverage, depth test
  if (!(color >> 24) && (m1 & M1_ALPHA)) return pixel;
  if (cyc == M0_COPY || cyc == M0_FILL) return cvg ? color : pixel;
  if (!zcmp.x || (m1 & M1_AA ? !cvg : !cvbit)) return pixel;
  // get blender input setting
  uint mux0[4] = { color, pixel, cmd.blend, cmd.fog  };
  uint mux1[4] = { a(color), a(cmd.fog), a(shade), 0 };
  uint c0 = mux0[(cycle ? m1 >> 28 : m1 >> 30) & 0x03];
  uint c1 = mux1[(cycle ? m1 >> 24 : m1 >> 26) & 0x03];
  uint c2 = mux0[(cycle ? m1 >> 20 : m1 >> 22) & 0x03];
  uint mux2[4] = { c1 ^ 0xffffffff, a(pixel), -1, 0  };
  uint c3 = mux2[(cycle ? m1 >> 16 : m1 >> 18) & 0x03];
  // evaluate blender equation
  int4 p = unpack32(c0), a = unpack32(c1) / 8 + 0;
  int4 m = unpack32(c2), b = unpack32(c3) / 8 + 1;
  uint r = pack32((a * p + b * m) / (a + b));
  if (cyc == M0_2CYCLE && !cycle) return r;
  if (cmd.modes[1] & M1_ZWRITE) depth = zcmp.y;
  // write blended coverage value
  uint ocvg = pixel >> 29, ovf = cvg + ocvg > 7;
  bool enabled = !ovf && (m1 & M1_AA);
  if (!enabled && !(m1 & M1_BLEND)) r = c0;
  uint muxc[4] = { max(cvg + ocvg, 7), (cvg + ocvg) & 7, 7, ocvg };
  seta(r, muxc[(m1 >> 8) & 0x03] << 29); return r;
}

uint dither(uint color, uint2 pos, RDPCommand cmd) {
  if ((cmd.modes[0] & M0_DITH) == M0_DITH) return color;
  uint dith[16] = { 7, 1, 6, 0, 3, 5, 2, 4, 4, 2, 5, 3, 0, 6, 1, 7 };
  uint off = dith[(pos.y & 3) * 4 + (pos.x & 3)];
  return pack32((off + unpack32(color)) & ~0x7);
}


[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  uint width = globals[0].width;
  if (GlobalID.x >= width) return;
  if (GlobalID.y >=   240) return;

  uint pixel, depth;
  load(GlobalID.xy, pixel, depth);
  uint rand = 0xff888888;

  uint tid = GroupID.y * width / 8 + GroupID.x;
  TileData tile = tiles[tid];
  for (uint i = 0; i < 64; ++i) {
    uint bitmask = tile.cmd_idxs[i];
    while (bitmask != 0) {
      uint lsb = firstbitlow(bitmask);
      RDPCommand cmd = cmds[(i << 5) | lsb];
      bitmask &= ~(0x1 << lsb);

      uint cvbit = 0, color = 0; int2 dxy;
      uint cvg = visible(GlobalID.xy, cmd, dxy, cvbit);
      if (cvg == 0) continue;

      uint shade = calc_shade(dxy, cmd);
      int4 coord = calc_coord(dxy, cmd);
      int2 zcmp  = calc_depth(coord.w, depth, cvg, pixel, cmd);
      uint tex0  = sample_tex(coord.xy, cmd, 0);
      uint tex1  = sample_tex(coord.xy, cmd, 1);

      color = combine(tex0, tex1, shade, rand, cvg, color, cmd, 0);
      color = combine(tex0, tex1, shade, rand, cvg, color, cmd, 1);
      color = mix_alpha_cvg(color, cvg, cmd);

      pixel = blend(pixel, color, shade, zcmp, depth, cvg, cvbit, cmd, 0);
      pixel = blend(pixel, color, shade, zcmp, depth, cvg, cvbit, cmd, 1);
      pixel = dither(pixel, GlobalID.xy, cmd);
    }
  }

  store(GlobalID.xy, pixel, depth);
}
