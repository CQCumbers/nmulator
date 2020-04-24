/* === Bitfield Flags === */

#define M0_COPY      0x00200000
#define M0_PERSP     0x00080000
#define M0_TLUT_IA   0x00004000

#define M1_BLEND     0x00004000
#define M1_CVG2ALPHA 0x00002000
#define M1_ALPHA2CVG 0x00001000
#define M1_ZMODE     0x00000c00
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
  uint n_cmds, pad;
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
  uint env, prim, zprim;
  uint keys[3];
};

struct RDPTex {
  uint format, size;
  uint width, addr, pal;
  uint mask[2], shift[2];
};

/* === Resource Buffers === */

[[vk::binding(0)]] StructuredBuffer<RDPCommand> cmds;
[[vk::binding(1)]] StructuredBuffer<TileData> tiles;
[[vk::binding(2)]] StructuredBuffer<RDPTex> texes;
[[vk::binding(3)]] StructuredBuffer<GlobalData> globals;

[[vk::binding(4)]] ByteAddressBuffer tmem;
[[vk::binding(5)]] RWByteAddressBuffer pixels;
[[vk::binding(6)]] RWByteAddressBuffer zbuf;

/* === Utility Functions === */

uint read_rgba16(uint input) {
  // GBARG25153 to ABGR8888
  uint output = ((input >> 3) & 0x1f) << 3;
  output |= ((input & 0x7) << 2 | ((input >> 14) & 0x3)) << 11;
  output |= ((input >> 9) & 0x1f) << 19;
  return output | (((input >> 8) & 0x1) * 0xff) << 24;
}

uint write_rgba16(uint input) {
  // ABGR8888 to GBARG25153
  uint output = ((input >> 3) & 0x1f) << 3;
  output |= ((input >> 13) & 0x7) | (((input >> 11) & 0x3) << 14);
  output |= ((input >> 19) & 0x1f) << 9;
  return output | (((input >> 31) & 0x1) << 8);
}

uint4 usadd(uint4 a, uint4 b) {
  return b == 0 ? a : a + max(b, -a);
}

int mul16(int a, int b) {
   // multiply 16.16 ints, with 16.16 result
   int a1 = a >> 16, a2 = a & 0xffff;
   int b1 = b >> 16, b2 = b & 0xffff;
   return (a1 * b1 << 16) + a1 * b2 + a2 * b1;
}

uint sel(uint idx, uint4 inputs) {
  switch (idx & 0x3) {
    case 0: return inputs.x;
    case 1: return inputs.y;
    case 2: return inputs.z;
    case 3: return inputs.w;
  }
}

uint pack32(uint4 color) {
  uint4 c = min(color >> 16, 0xff);
  return (c.w << 24) | (c.z << 16) | (c.y << 8) | c.x;
}

/* === Pipeline Stages === */

uint read_texel(RDPTex tex, RDPCommand cmd, uint s, uint t) {
  s >>= 5, t >>= 5; // convert 16.16 texture coords to texels
  uint ms = s & (1 << tex.mask[0]), mt = t & (1 << tex.mask[1]);
  //if (tex.shift[0] <= 10) { s >>= tex.shift[0]; } else { s <<= (16 - tex.shift[0]); }
  //if (tex.shift[1] <= 10) { t >>= tex.shift[1]; } else { t <<= (16 - tex.shift[1]); }
  if (tex.mask[0]) { s &= ((1 << tex.mask[0]) - 1); if (ms) s = ((1 << tex.mask[0]) - 1) - s; }
  if (tex.mask[1]) { t &= ((1 << tex.mask[1]) - 1); if (mt) t = ((1 << tex.mask[1]) - 1) - t; }
  
  tex.addr = (tex.addr & 0xfff) + (cmd.tmem << 12);
  cmd.tlut = (cmd.tlut & 0xfff) + (cmd.tmem << 12);
  if (tex.format == 0 && tex.size == 2) {        // 16 bit RGBA
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    uint texel = tmem.Load(tex_pos);
    return read_rgba16(s & 0x1 ? texel >> 16 : texel & 0xffff);
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    tex_pos = (tex.addr & ~0x7ff) | (tex_pos & 0x7ff);
    uint texel1 = tmem.Load(tex_pos + 0x800), texel2 = tmem.Load(tex_pos);
    texel1 = s & 0x1 ? texel1 & ~0xffff : texel1 << 16;
    texel2 = s & 0x1 ? texel2 >> 16 : texel2 & 0xffff;
    return texel1 | texel2;
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    uint tex_pos = tex.addr + t * tex.width + s;
    uint idx = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    uint texel = tmem.Load(cmd.tlut + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.modes[0] & M0_TLUT_IA) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint idx = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    idx = (s & 0x1 ? idx : idx >> 4) & 0xf;
    uint texel = tmem.Load(cmd.tlut + (tex.pal << 5) + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.modes[0] & M0_TLUT_IA) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    texel = s & 0x1 ? texel >> 16 : texel & 0xffff;
    uint i = texel & 0xff, a = texel >> 8;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    uint tex_pos = tex.addr + t * tex.width + s;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x3) * 8);
    uint i = texel & 0xf0, a = (texel << 4) & 0xf0;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    texel = (s & 0x1 ? texel : texel >> 4) & 0xf;
    uint i = (texel & 0xe) << 4, a = (texel & 0x1) * 0xff;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    uint tex_pos = tex.addr + t * tex.width + s;
    uint i = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    uint i = (s & 0x1 ? texel << 4 : texel) & 0xf0;
    return (i << 24) | (i << 16) | (i << 8) | i;
  }
  return 0xffff00ff;
}

uint visible(uint2 pos, RDPCommand cmd, out int2 dxy) {
  // check pixel within y bounds
  int y1 = max(cmd.yh, cmd.syh);
  int y2 = min(cmd.yl, cmd.syl);
  if (!(y1 / 4 <= pos.y && pos.y <= y2 / 4)) return 0;
  // compute edges at each sub-scanline
  int4 y = pos.y * 4 + int4(0, 1, 2, 3);
  int4 x1 = cmd.xh + cmd.sh * (y - cmd.yh) / 4;
  int4 xm = cmd.xm + cmd.sm * (y - cmd.yh) / 4;
  int4 xl = cmd.xl + cmd.sl * (y - cmd.ym) / 4;
  int4 x2 = y < int4(cmd.ym) ? xm : xl;
  // compute valid sub-scanlines
  bool lft = cmd.type & (T_LMAJOR | T_RECT);
  int4 xa = (lft ? x1 : x2), xb = (lft ? x2 : x1);
  bool4 vy = (int4)y1 <= y && y < (int4)y2 && xa < xb;
  dxy = pos * 4, dxy.x -= (x1.x >> 14), dxy.y -= cmd.yh;
  // check x bounds at every other subpixel
  xa = max(xa, (int4)cmd.sxh) >> 14;
  xb = min(xb, (int4)cmd.sxl) >> 14;
  int4 x = pos.x * 4 + int4(0, 1, 0, 1);
  return dot(vy && xa < x && x < xb, (int4)1);
}

bool2 compare_z(inout uint zmem, uint cvg, int2 dxy, RDPCommand cmd, out uint4 stwz) {
  stwz = cmd.tex + cmd.tde * dxy.y / 4;
  stwz = stwz + cmd.tdx * dxy.x / 4;
  // read old z, dz from zbuf
  bool skip = (~cmd.type & T_ZBUF) || (~cmd.modes[1] & M1_ZCMP);
  if (cvg == 0 || skip) return bool2(true);
  int oz = zmem >> 4, odz = zmem & 0xf;
  bool zmax = (oz == 0x3ffff);
  // calculate new z from slopes
  uint nz = (cmd.zprim >> 14) & 0x3fffc;
  if (~cmd.modes[1] & M1_ZSRC) nz = stwz.w >> 14;
  // get new dz from prim_z
  uint ndz = firstbithigh(max(cmd.zprim & 0xffff, 1));
  uint dz = 0x1 << max(ndz, odz);
  uint mode = cmd.modes[1] & M1_ZMODE;
  if (mode == ZTRANS || cvg > 7) dz = 0;
  // compare new z with old z
  bool zle = int((nz - dz) << 14) <= int(oz << 14);
  bool zge = int((nz + dz) << 14) >= int(oz << 14);
  if (mode != ZDECAL) zle = zmax || zle;
  else zle = zge && zle && !zmax;
  // write new z/dz if closer
  bool write = cmd.modes[1] & M1_ZWRITE;
  if (zle && write) zmem = (nz << 4) | ndz;
  return bool2(zle, zge);
}

uint sample_tex(uint3 stw, RDPCommand cmd) {
  if (~cmd.type & T_TEX) return 0;
  uint mode = cmd.modes[0]; stw >>= 16;
  RDPTex tex = texes[cmd.tmem * 8 + cmd.tile];
  if (mode & M0_PERSP) stw = 32768.0 * stw / stw.z;
  return read_texel(tex, cmd, stw.x, stw.y);
}

uint sample_shade(int2 dxy, RDPCommand cmd) {
  if (~cmd.type & T_SHADE) return 0;
  uint4 rgba = cmd.shade + cmd.sde * dxy.y / 4;
  return pack32(rgba + cmd.sdx * dxy.x / 4);
}

uint combine(uint tex, uint shade, RDPCommand cmd) {
  if (cmd.type & T_TEX) return tex;
  if (cmd.type & T_SHADE) return shade;
  return globals[0].size == 4 ? cmd.fill : read_rgba16(cmd.fill);
}

uint blend(uint pixel, uint color, uint cvg, bool far, RDPCommand cmd) {
  // multiply 3-bit coverage and 5-bit alpha
  uint mux = cmd.modes[1], alpha = color >> 27, ocvg = pixel >> 27;
  if (mux & M1_ALPHA2CVG) cvg = (alpha * cvg + 4) >> 5;
  if (mux & M1_CVG2ALPHA) alpha = cvg << 2;
  if ((mux & M1_ALPHA) && alpha < (cmd.blend >> 27)) return pixel;
  // select blender inputs
  uint p = sel(mux >> 30, uint4(color, pixel, cmd.blend, cmd.fog));
  uint a = sel(mux >> 26, uint4(alpha, cmd.fog >> 27, alpha, 0x0));
  uint m = sel(mux >> 22, uint4(color, pixel, cmd.blend, cmd.fog));
  uint b = sel(mux >> 18, uint4(a ^ 0x1f, pixel >> 27, 0x1f, 0x0)) + 1;
  // skip inputs based on flags
  bool copy = cmd.modes[0] & M0_COPY;
  bool on_cvg = mux & M1_ON_CVG, force = mux & M1_BLEND;
  bool full = cvg + ocvg > 7, aa = (mux & M1_AA) && far && !full;
  if (copy || !(force || aa)) a = 1, b = ocvg = 0;
  else if (on_cvg && !full) a = 0, b = 1;
  // blend selected inputs
  int c1 = clamp(cvg + ocvg, 0, 7), c2 = (cvg + ocvg) & 7;
  uint output = sel(mux >> 8, uint4(c1, c2, 0x7, ocvg)) << 27;
  for (uint i = 0; i < 3; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= min(oc, 0xff) << (i * 8);
  }
  return output;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  GlobalData global = globals[0];
  if (GlobalID.x >= global.width) return;
  uint tile_pos = GlobalID.y * global.width + GlobalID.x;
  uint pixel = pixels.Load(tile_pos * global.size), zmem = zbuf.Load(tile_pos * 4);
  if (global.size == 2) pixel = read_rgba16(tile_pos & 0x1 ? pixel >> 16 : pixel);

  TileData tile = tiles[GroupID.y * (global.width / 8) + GroupID.x];
  for (uint i = 0; i < 64; ++i) {
    uint bitmask = tile.cmd_idxs[i];
    while (bitmask != 0) {
      uint lsb = firstbitlow(bitmask);
      RDPCommand cmd = cmds[(i << 5) | lsb];
      bitmask &= ~(0x1 << lsb);

      int2 dxy; uint4 stwz;
      uint cvg = visible(GlobalID.xy, cmd, dxy);
      bool2 depth = compare_z(zmem, cvg, dxy, cmd, stwz);
      if (cvg == 0 || !depth.x) continue;
      uint tex = sample_tex(stwz, cmd);
      uint shade = sample_shade(dxy, cmd);
      uint color = combine(tex, shade, cmd);
      pixel = blend(pixel, color, cvg, true, cmd);
    }
  }
  
  if (global.size == 2) {
    if (tile_pos & 0x1) {
      pixels.InterlockedAnd(tile_pos * 2, 0x0000ffff);
      pixels.InterlockedOr(tile_pos * 2, write_rgba16(pixel) << 16);
    } else {
      pixels.InterlockedAnd(tile_pos * 2, 0xffff0000);
      pixels.InterlockedOr(tile_pos * 2, write_rgba16(pixel));
    }
  } else pixels.Store(tile_pos * 4, pixel);
  zbuf.Store(tile_pos * 4, zmem);
}
