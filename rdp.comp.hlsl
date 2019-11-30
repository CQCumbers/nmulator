struct PerTileData {
  uint n_cmds;
  uint cmd_idxs[7];
};

struct RDPCommand {
  uint xyh[2], xym[2], xyl[2];
  int sh, sm, sl;
  uint fill, lft, type;
  uint shade[4], sde[4], sdx[4];
  uint tile, pad;
  uint tex[2], tde[2], tdx[2];
};

struct RDPTile {
  uint format, size;
  uint width, addr;
  uint mask[2], shift[2];
};

StructuredBuffer<RDPCommand> cmds : register(t0);
StructuredBuffer<PerTileData> tiles : register(t1);
StructuredBuffer<RDPTile> texes : register(t2);
static const uint width = 320;
static const uint pixel_size = 4;

ByteAddressBuffer tmem : register(t3);
RWByteAddressBuffer pixels : register(t4);

uint read_texel(RDPTile tex, uint s, uint t) {
  uint ms = s & (1 << tex.mask[0]), mt = t & (1 << tex.mask[1]);
  if (tex.mask[0]) { s &= ((1 << tex.mask[0]) - 1); if (ms) s = ((1 << tex.mask[0]) - 1) - s; }
  if (tex.mask[1]) { t &= ((1 << tex.mask[1]) - 1); if (mt) t = ((1 << tex.mask[1]) - 1) - t; }
  if (tex.shift[0] < 10) { s >>= tex.shift[0]; } else { s <<= (16 - tex.shift[0]); }
  if (tex.shift[1] < 10) { t >>= tex.shift[1]; } else { t <<= (16 - tex.shift[1]); }
  
  if (tex.format == 0 && tex.size == 2) { // 16 bit RGBA
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    if ((tex.addr + t * tex.width + s * 2) & 0x2) texel = texel >> 16;
    // convert big-endian RGBA 5551
    uint color = ((texel >> 3) & 0x1f) << 27;
    color |= (((texel & 0x7) << 2) | ((texel >> 14) & 0x3)) << 19;
    color |= ((texel >> 9) & 0x1f) << 11;
    return color | -((texel >> 8) & 0x1) & 0xff;
  } else if (tex.format == 0 && tex.size == 3) { // 32 bit RGBA
    uint texel = tmem.Load(tex.addr + t * tex.width * 2 + s * 4);
    //uint texel2 = tmem.Load(tex.addr + s * 4 + 2);// + t * tex.width + s * 2);
    return texel; //((texel2 & 0xffff) << 16) | (texel2 & 0xffff);
  } else if (tex.format == 3 && tex.size == 2) { // 16 bit IA
    uint texel = tmem.Load(tex.addr + t * tex.width + s * 2);
    uint i = (texel >> 8) & 0xff, a = texel & 0xff;
    return (i << 24) | (i << 16) | (i << 8) | a;
  }
  return ~0x0;
}

uint sample_color(uint2 pos, RDPCommand cmd) {
  // rectangle texture read
  if (cmd.type == 3 || cmd.type == 4) {
    RDPTile tex = texes[cmd.tile];
    uint s = pos.x - (cmd.xyh[0] >> 2); s = (s * (cmd.tdx[0] >> 0)) >> 10;
    uint t = pos.y - (cmd.xyh[1] >> 2); t = (t * (cmd.tde[1] >> 0)) >> 10;
    return cmd.type == 4 ? read_texel(tex, t, s) : read_texel(tex, s, t);
  }
  if (cmd.type != 2) return cmd.fill;
  // convert to subpixel, calc major edge
  uint x = pos.x << 16, y = pos.y << 2, color = 0x0;
  uint x1 = cmd.xyh[0] + cmd.sh * ((y - cmd.xyh[1]) >> 2);
  // add d(s)de and d(s)dx to inital shade, convert to RGBA8888
  for (uint i = 0; i < 4; ++i) {
    uint channel = cmd.shade[i] + cmd.sde[i];
    channel += cmd.sde[i] * ((y - cmd.xyh[1]) >> 2);
    uint c2 = cmd.sdx[i] * ((x - x1) >> 16);
    if (c2 != 0) channel += max(c2, -channel);
    color |= ((channel >> 16) & 0xff) << (24 - i * 8);
  }
  return color;
}

uint visible(uint2 pos, RDPCommand cmd) {
  // convert to subpixels, check y bounds
  uint x = pos.x << 16, y = pos.y << 2;
  if (cmd.xyh[1] > y || y >= cmd.xyl[1]) return 0;
  if (cmd.type == 1 || cmd.type == 3 || cmd.type == 4) // rectangles
    return (cmd.xyh[0] <= (pos.x << 2) && (pos.x << 2) < cmd.xyl[0]);
  // calculate x bounds from slopes
  uint x1 = cmd.xyh[0] + cmd.sh * ((y - cmd.xyh[1]) >> 2), x2;
  if (y < cmd.xym[1]) x2 = cmd.xym[0] + cmd.sm * ((y - cmd.xyh[1]) >> 2);
  else x2 = cmd.xyl[0] + cmd.sl * ((y - cmd.xym[1]) >> 2);
  return ((cmd.lft && x1 <= x && x < x2) ||
         (!cmd.lft && x2 <= x && x < x1));
}

uint shade(uint pixel, uint color, uint coverage) {
  if ((color & 0xff) == 0) return pixel; 
  return pixel * (1 - coverage) + color * coverage;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  uint pixel = pixels.Load((GlobalID.y * width + GlobalID.x) * pixel_size);
  PerTileData tile = tiles[GroupID.y * (width / 8) + GroupID.x];
  for (uint i = 0; i < tile.n_cmds; ++i) {
    RDPCommand cmd = cmds[tile.cmd_idxs[i]];
    uint coverage = visible(GlobalID.xy, cmd);
    uint color = sample_color(GlobalID.xy, cmd);
    pixel = shade(pixel, color, coverage);
  }
  pixels.Store((GlobalID.y * width + GlobalID.x) * pixel_size, pixel);
}
