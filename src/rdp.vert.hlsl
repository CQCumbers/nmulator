struct Output {
  float4 position : SV_POSITION;
  float4 color : COLOR;
};

static const float2 positions[] = {
  float2(0.0, -0.5),
  float2(0.5, 0.5),
  float2(-0.5, 0.5)
};

static const float3 colors[] = {
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0)
};

Output main(uint id: SV_VertexID) {
  Output output;
  output.position = float4(positions[id], 0.0, 1.0);
  output.color = float4(colors[id], 1.0);
  return output;
}
