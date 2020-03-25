struct PerTileData {
  uint cmd_idxs[2];
};

struct Bounds {
  int xh, xl, yh, yl;
};

struct RDPCommand {
  uint type, tile;
  int xh, xm, xl;
  int yh, ym, yl;
  int sh, sm, sl;

  int shade[4], sde[4], sdx[4];
  int tex[3], tde[3], tdx[3];
  int zpos, zde, zdx;

  uint fill, fog, blend;
  uint env, prim, zprim;
  uint cc_mux, tlut, tmem;
  uint keys[3], modes[2];
  Bounds sci;
};

struct RDPTile {
  uint format, size;
  uint width, addr, pal;
  uint mask[2], shift[2];
};

struct GlobalData {
  uint width, size;
};

#define M0_COPY     0x00200000
#define M0_PERSP    0x00080000
#define M0_TLUT_IA  0x00004000
#define M1_ZSRC     0x00000004
#define T_ZBUF      0x01
#define T_TEX       0x02
#define T_SHADE     0x04
#define T_RECT      0x08
#define T_LMAJOR    0x10

[[vk::binding(0)]] StructuredBuffer<RDPCommand> cmds;
[[vk::binding(1)]] StructuredBuffer<PerTileData> tiles;
[[vk::binding(2)]] StructuredBuffer<RDPTile> texes;
[[vk::binding(3)]] ConstantBuffer<GlobalData> global;

[[vk::binding(4)]] ByteAddressBuffer tmem;
[[vk::binding(5)]] RWByteAddressBuffer pixels;
[[vk::binding(6)]] RWByteAddressBuffer zbuf;

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

uint read_texel(RDPTile tex, RDPCommand cmd, uint s, uint t) {
  s >>= 5, t >>= 5; // convert 16.16 texture coords to texels
  uint ms = s & (1 << tex.mask[0]), mt = t & (1 << tex.mask[1]);
  /*if (tex.shift[0] <= 10) { s >>= tex.shift[0]; } else { s <<= (16 - tex.shift[0]); }
  if (tex.shift[1] <= 10) { t >>= tex.shift[1]; } else { t <<= (16 - tex.shift[1]); }*/
  if (tex.mask[0]) { s &= ((1 << tex.mask[0]) - 1); if (ms) s = ((1 << tex.mask[0]) - 1) - s; }
  if (tex.mask[1]) { t &= ((1 << tex.mask[1]) - 1); if (mt) t = ((1 << tex.mask[1]) - 1) - t; }
  
  tex.addr = (tex.addr & 0xfff) | (cmd.tmem << 12);
  if (tex.format == 0 && tex.size == 2) {        // 16 bit RGBA
    //if (s * 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s * 2;
    uint texel = tmem.Load(tex_pos);
    return read_rgba16(s & 0x1 ? texel >> 16 : texel & 0xffff);
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    if (s * 2 >= tex.width) return 0;
    uint tex_pos = min(tex.addr + t * tex.width + s * 2, 0x7ff);
    uint texel1 = tmem.Load(tex_pos + 0x800), texel2 = tmem.Load(tex_pos);
    texel1 = s & 0x1 ? texel1 & ~0xffff : texel1 << 16;
    texel2 = s & 0x1 ? texel2 >> 16 : texel2 & 0xffff;
    return texel1 | texel2;
  } else if (tex.format == 2 && tex.size == 1) { // 8 bit CI
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint idx = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    uint texel = tmem.Load(cmd.tlut + idx * 2);
    texel = idx & 0x1 ? texel >> 16 : texel & 0xffff;
    if (cmd.modes[0] & M0_TLUT_IA) {
      uint i = texel & 0xff, a = texel >> 8;
      return (a << 24) | (i << 16) | (i << 8) | i;
    } else return read_rgba16(texel);
  } else if (tex.format == 2 && tex.size == 0) { // 4 bit CI
    if (s / 2 >= tex.width) return 0;
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
    if (s * 2 >= tex.width) return 0;
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    texel = s & 0x1 ? texel >> 16 : texel & 0xffff;
    uint i = texel & 0xff, a = texel >> 8;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 1) { // 8 bit IA
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x3) * 8);
    uint i = texel & 0xf0, a = (texel << 4) & 0xf0;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 3 && tex.size == 0) { // 4 bit IA
    if (s / 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    texel = (s & 0x1 ? texel : texel >> 4) & 0xf;
    uint i = (texel & 0xe) << 4, a = (texel & 0x1) * 0xff;
    return (a << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 1) { // 8 bit I
    if (s >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s;
    uint i = (tmem.Load(tex_pos) >> ((s & 0x3) * 8)) & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | i;
  } else if (tex.format == 4 && tex.size == 0) { // 4 bit I
    if (s / 2 >= tex.width) return 0;
    uint tex_pos = tex.addr + t * tex.width + s / 2;
    uint texel = tmem.Load(tex_pos) >> ((s & 0x6) * 4);
    uint i = (s & 0x1 ? texel << 4 : texel) & 0xf0;
    return (i << 24) | (i << 16) | (i << 8) | i;
  }
  return 0xffff00ff;
}

int sadd(int a, int b) {
  int res = (a < 0) ? -2147483648 : 2147483647;
  if ((a < 0) == (b > res - a)) res = a + b;
  return res;
}

uint usadd(uint a, uint b) {
  return b == 0 ? a : a + max(b, -a);
}

int mul16(int a, int b) {
   // multiply 16.16 ints, with 16.16 result
   int a1 = a >> 16, a2 = a & 0xffff;
   int b1 = b >> 16, b2 = b & 0xffff;
   return (a1 * b1 << 16) + a1 * b2 + a2 * b1;
}

int sample_color(uint2 pos, RDPCommand cmd) {
  if (cmd.type & T_TEX) {
    RDPTile tex = texes[(cmd.tmem << 3) | (cmd.tile & 0x7)]; uint s, t, w;
    int x = pos.x << 16, dy = (pos.y << 2) - cmd.yh;
    if (cmd.modes[0] & M0_COPY) {
      s = cmd.tex[0] + ((x - cmd.xh) << 5) >> 16;
      t = cmd.tex[1] + (cmd.tde[1] * dy >> 2) >> 16;
    } else if (cmd.type & T_RECT) {
      s = cmd.tex[0] + mul16(cmd.tde[0], x - cmd.xh) >> 16;
      t = cmd.tex[1] + (cmd.tde[1] * dy >> 2) >> 16;
    } else { // triangle
      int x1 = cmd.xh + (cmd.sh * dy >> 2), dx = (x - x1) >> 16;
      s = cmd.tex[0] + (cmd.tde[0] * dy >> 2) + (cmd.tdx[0] * dx);
      t = cmd.tex[1] + (cmd.tde[1] * dy >> 2) + (cmd.tdx[1] * dx);
      w = cmd.tex[2] + (cmd.tde[2] * dy >> 2) + (cmd.tdx[2] * dx);
      s >>= 16, t >>= 16, w >>= 16;
      if (cmd.modes[0] & M0_PERSP) {
        s = (w ? int(32768.0 * float(s) / float(w)) : 0xffff);
        t = (w ? int(32768.0 * float(t) / float(w)) : 0xffff);
      }
    }
    //if (cmd.type == T_FLIP) return read_texel(tex, cmd, t, s);
    return read_texel(tex, cmd, s, t);
  } else if (cmd.type & T_SHADE) {
    // convert to subpixels, calculate major edge
    int x = pos.x << 16, y = pos.y << 2;
    int dy = y - cmd.yh; uint color = 0x0;
    int x1 = cmd.xh + (cmd.sh * dy >> 2);
    // interpolate shading along major edge and line
    for (uint i = 0; i < 4; ++i) {
      uint chan = usadd(cmd.shade[i], cmd.sde[i] * (dy + 1) >> 2);
      chan = usadd(chan, mul16(cmd.sdx[i], x - x1));
      color |= ((chan >> 16) & 0xff) << (i * 8);
    }
    return color;
  }
  return global.size == 4 ? cmd.fill : read_rgba16(cmd.fill);
}

/*uint sample_z(uint2 pos, RDPCommand cmd) {
  if (~cmd.type & T_ZBUF) return 0x7fff;
  if (cmd.modes[1] & M1_ZSRC) return cmd.zprim >> 16;
  int x = pos.x << 16, dy = (pos.y << 2) - cmd.xyh[1];
  int x1 = cmd.xyh[0] + (cmd.sh * dy >> 2);
  int z = sadd(cmd.zpos, cmd.zde * dy >> 2);
  return sadd(z, mul16(cmd.zdx, x - x1)) >> 16;
}*/

/*uint sample_color(uint2 pos, RDPCommand cmd) {
  if (~cmd.type & T_RECT) return ~0x0;
  return global.size == 4 ? cmd.fill : read_rgba16(cmd.fill);
}*/

uint sample_z(uint2 pos, RDPCommand cmd) {
  return 0x7fff;
}

/*uint visible(uint2 pos, RDPCommand cmd) {
  // check pixel within y bounds
  int y1 = max(cmd.yh, cmd.sci.yh);
  int y2 = min(cmd.yl, cmd.sci.yl);
  if (!(y1 / 4 <= pos.y && pos.y <= y2 / 4)) return 0;

  // compute rectangle x bounds
  // should be subpixel accurate too?
  if (cmd.type & T_RECT) {
    int x1 = max(cmd.xh, cmd.sci.xh) >> 16;
    int x2 = min(cmd.xl, cmd.sci.xl) >> 16;
    return x1 <= pos.x && pos.x <= x2;
  }

  // compute edges at each sub-scanline
  int4 y = pos.y * 4 + int4(0, 1, 2, 3);
  int4 x1 = cmd.xh + cmd.sh * (y - cmd.yh) / 4;
  int4 xm = cmd.xm + cmd.sm * (y - cmd.yh) / 4;
  int4 xl = cmd.xl + cmd.sl * (y - cmd.ym) / 4;
  int4 x2 = y < int4(cmd.ym) ? xm : xl;
  // compute valid sub-scanlines
  bool lft = cmd.type & T_LMAJOR;
  int4 xa = (lft ? x1 : x2), xb = (lft ? x2 : x1);
  bool4 vy = (int4(y1) <= y && y < int4(y2) && xa < xb);
  // check x bounds at every other subpixel
  xa = max(xa, int4(cmd.sci.xh)) >> 14;
  xb = min(xb, int4(cmd.sci.xl)) >> 14;
  int4 x = (pos.x << 2) + int4(0, 1, 0, 1);
  return dot(vy && xa < x && x < xb, int4(1));
}*/

uint visible(uint2 pos, RDPCommand cmd) {
  // convert to subpixels, check y bounds
  int x = (pos.x << 16) + 0x8000, y = pos.y << 2;
  int y1 = cmd.yh, y2 = cmd.yl;
  if (y + 4 <= y1 || y >= y2 + 3) return 0;
  if (cmd.type & T_RECT) return cmd.xh <= x && x <= cmd.xl;
  // calculate x bounds from slopes
  int dy = y + 1 - cmd.yh, x2 = 0;
  int x1 = cmd.xh + (cmd.sh * dy >> 2);
  if (y < cmd.ym) x2 = cmd.xm + (cmd.sm * dy >> 2);
  else x2 = cmd.xl + (cmd.sl * (y - cmd.ym) >> 2);
  x1 &= 0xffff0000, x2 &= 0xffff0000;
  bool lft = cmd.type & T_LMAJOR;
  return lft ? (x1 <= x && x <= x2 + 0xffff) : (x2 <= x && x <= x1 + 0xffff);
}

/*uint combine(RDPCommand cmd) {
  uint a, b, c, d;
  uint ma = (cmd.cc_mux >> 14) & 0x3, mb = (cmd.cc_mux >> 10) & 0x3;
  uint mc = (cmd.cc_mux >> 6) & 0x3, md = (cmd.cc_mux >> 2) & 0x3;
  // select a
  if (ma == 1) a = tex_out;
  else if (ma == 3) a = cmd.prim;
  else if (ma == 4) a = shade_out;
  else if (ma == 5) a = cmd.env;
  else if (ma == 6) a = 0xff;
  else if (ma >= 8) a = 0x0;
  // select b
  if (mb == 1) b = tex_out;
  else if (mb == 3) b = cmd.prim;
  else if (mb == 4) b = shade_out;
  else if (mb == 5) b = cmd.env;
  else if (mb == 6) b = cmd.key_center;
  else if (mb == 8) b = 0x0;
  // select c
  if (mc == 0) c = tex_out;
  else if (mc == 3) c = cmd.prim;
  else if (mc == 4) c = shade_out;
  else if (mc == 5) c = cmd.env;
  else if (mc == 6) c = cmd.key_scale;
  else if (mc == 8) c = tex_out >> 24;
  else if (mc == 10) c = cmd.prim >> 24;
  else if (mc == 11) c = shade_out >> 24;
  else if (mc == 10) c = cmd.prim >> 24;
  // select d
  if (md == 0) d = color;
  else if (md == 3) d = cmd.prim;
  else if (md == 4) d = shaded;
  else if (md == 5) d = cmd.env;
  else if (md == 6) d = 0xff;
  // combine selected sources
  uint output = 0x0;
  for (uint i = 0; i < 4; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= (oc & 0xff) << (i * 8);
  }
  return output;
}*/

uint blend(uint pixel, uint color, uint coverage, RDPCommand cmd) {
  uint a, p, b, m; uint mux = cmd.modes[1] >> 16;
  uint m1a = (mux >> 14) & 0x3, m1b = (mux >> 10) & 0x3;
  uint m2a = (mux >> 6) & 0x3, m2b = (mux >> 2) & 0x3;
  if (cmd.modes[0] & M0_COPY) m1a = 0, m2a = 1, m1b = 0, m2b = 0;
  // select p 
  if (m1a == 0) p = color;
  else if (m1a == 1) p = pixel;
  else if (m1a == 2) p = cmd.blend;
  else if (m1a == 3) p = cmd.fog;
  // select m
  if (m2a == 0) m = color;
  else if (m2a == 1) m = pixel;
  else if (m2a == 2) m = cmd.blend;
  else if (m2a == 3) m = cmd.fog;
  // select a
  if (m1b == 0) a = (color >> 24) & 0xff;
  else if (m1b == 1) a = (cmd.fog >> 24) & 0xff;
  else if (m1b == 2) a = (color >> 24) & 0xff; // should be shade, from before CC?
  else if (m1b == 3) a = 0x0;
  // select b
  if (m2b == 0) b = ~a & 0xff;
  else if (m2b == 1) b = (pixel >> 24) & 0xff;
  else if (m2b == 2) b = 0xff;
  else if (m2b == 3) b = 0x00;
  // blend selected colors
  a >>= 3, b >>= 3;
  bool color_on_cvg = (cmd.modes[1] >> 7) & 0x1;
  bool force_blend = (cmd.modes[1] >> 14) & 0x1;
  bool overflow = (color >> 24) == 0xff;
  if (color_on_cvg || (overflow && !force_blend)) b = 0;
  uint output = 0xff000000;
  for (uint i = 0; i < 3; ++i) {
    uint pc = (p >> (i * 8)) & 0xff;
    uint mc = (m >> (i * 8)) & 0xff;
    uint oc = (a * pc + b * mc) / (a + b);
    output |= min(oc, 0xff) << (i * 8);
  }
  return output; //(output * coverage + pixel * (8 - coverage)) / 8;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  if (GlobalID.x >= global.width) return;
  uint tile_pos = GlobalID.y * global.width + GlobalID.x;

  uint pixel = pixels.Load(tile_pos * global.size);
  if (global.size == 2) pixel = read_rgba16(tile_pos & 0x1 ? pixel >> 16 : pixel);
  //int zval = zbuf.Load(tile_pos * global.size);
  //zval = (tile_pos & 0x1 ? zval >> 16 : zval) & 0x7fff; zval = (zval ^ 0x7fff);
  int zval = 0x7fff;

  PerTileData tile = tiles[GroupID.y * (global.width / 8) + GroupID.x];
  for (uint i = 0; i < 2; ++i) {
    uint bitmask = WaveActiveBitOr(tile.cmd_idxs[i]);
    while (bitmask != 0) {
      uint lsb = firstbitlow(bitmask);
      RDPCommand cmd = cmds[(i << 5) | lsb];
      bitmask &= ~(0x1 << lsb);

      uint coverage = visible(GlobalID.xy, cmd);
      uint color = sample_color(GlobalID.xy, cmd);
      int z = sample_z(GlobalID.xy, cmd);
      if (coverage == 0 || z > zval) continue;
      pixel = blend(pixel, color, coverage, cmd);
      if (cmd.type & T_ZBUF) zval = z;
    }
  }
  //pixel = read_texel(texes[7], cmds[0], GlobalID.x << 5, GlobalID.y << 5);
  
  zval = (zval >> 31) | (zval ^ 0x7fff);
  if (tile_pos & 0x1) {
    if (global.size == 2) {
      pixels.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
      pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel) << 16);
    } else pixels.Store(tile_pos * global.size, pixel);
    //zbuf.InterlockedAnd(tile_pos * global.size, 0x0000ffff);
    //zbuf.InterlockedOr(tile_pos * global.size, (zval & 0xffff) << 16);
  } else {
    if (global.size == 2) {
      pixels.InterlockedAnd(tile_pos * global.size, 0xffff0000);
      pixels.InterlockedOr(tile_pos * global.size, write_rgba16(pixel));
    } else pixels.Store(tile_pos * global.size, pixel);
    //zbuf.InterlockedAnd(tile_pos * global.size, 0xffff0000);
    //zbuf.InterlockedOr(tile_pos * global.size, zval & 0xffff);
  }
}
