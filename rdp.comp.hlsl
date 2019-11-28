struct PerTileData {
  uint n_cmds;
  uint cmd_idxs[31];
}; // Filled on CPU

struct RDPCommand {
  uint xyh[2];
  uint xym[2];
  uint xyl[2];
  int sh, sm, sl;
  uint color, lft, type;
}; // Filled on CPU

StructuredBuffer<RDPCommand> cmds : register(t0);
StructuredBuffer<PerTileData> tiles : register(t1);
RWByteAddressBuffer pixels : register(t2);
static const uint width = 320;

uint sample_color(RDPCommand cmd) {
  return cmd.color;
}

uint visible(uint2 pos, RDPCommand cmd) {
  // convert to subpixels, check y bounds
  uint x = pos.x << 16, y = pos.y << 2;
  if (cmd.xyh[1] > y || y > cmd.xyl[1]) return 0;
  if (cmd.type == 1) // rectangles
    return (cmd.xyh[0] <= (pos.x << 2) && (pos.x << 2) <= cmd.xyl[0]);
  // calculate x bounds from slopes
  uint x1 = cmd.xyh[0] + cmd.sh * ((y - cmd.xyh[1]) >> 2), x2;
  if (y < cmd.xym[1]) x2 = cmd.xym[0] + cmd.sm * ((y - cmd.xyh[1]) >> 2);
  else x2 = cmd.xyl[0] + cmd.sl * ((y - cmd.xym[1]) >> 2);
  return ((cmd.lft && x1 <= x && x <= x2) ||
         (!cmd.lft && x2 <= x && x <= x1));
}

uint shade(uint pixel, uint color, uint coverage) {
  return pixel * (1 - coverage) + color * coverage;
}

[numthreads(8, 8, 1)]
void main(uint3 GlobalID : SV_DispatchThreadID, uint3 GroupID : SV_GroupID) {
  uint pixel = pixels.Load((GlobalID.y * width + GlobalID.x) * 4);
  for (uint i = 0; i < tiles[GroupID.x].n_cmds; ++i) {
    RDPCommand cmd = cmds[tiles[GroupID.x].cmd_idxs[i]];
    uint color = sample_color(cmd);
    uint coverage = visible(GlobalID.xy, cmd);
    pixel = shade(pixel, color, coverage);
  }
  pixels.Store((GlobalID.y * width + GlobalID.x) * 4, pixel);
}
