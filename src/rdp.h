#ifndef RDP_H
#define RDP_H

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include "scheduler.h"
#include "shader.spv"

typedef struct PerTileData {
  uint32_t cmd_idxs[64];
} tile_t;

typedef struct Bounds {
  int32_t xh, xl, yh, yl;
} bounds_t;

// all coeffs are 16.16 fixed point
// except yh/yl/ym, which is 12.2
typedef struct RDPCommand {
  uint32_t type, tile;
  int32_t xh, xm, xl;
  int32_t yh, ym, yl;
  int32_t sh, sm, sl;

  int32_t shade[4], sde[4], sdx[4];
  int32_t tex[3], tde[3], tdx[3];
  int32_t zpos, zde, zdx;

  uint32_t fill, fog, blend;
  uint32_t env, prim, zprim;
  uint32_t cc_mux, tlut, tmem;
  uint32_t keys[3], modes[2];
  bounds_t sci;
} cmd_t;

typedef struct RDPTile {
  uint32_t format, size;
  uint32_t width, addr, pal;
  uint32_t mask[2], shift[2];
} tex_t;

typedef struct GlobalData {
  uint32_t width, size;
  //uint32_t zlut[0x88];
} global_t;

namespace RDP {
  void render();
}

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;
  VkCommandBuffer commands = VK_NULL_HANDLE;

  /* === Descriptor Memory Access === */

  uint8_t *mapped_mem;
  const uint32_t group_size = 8, max_cmds = 2048, max_copies = 512;
  uint32_t gwidth = 320 / group_size, gheight = 240 / group_size;
  uint32_t n_cmds = 0, n_tmems = 0;

  const VkDeviceSize cmds_size = max_cmds * sizeof(cmd_t);
  cmd_t *cmds_ptr() { return (cmd_t*)(mapped_mem); }
  VkBuffer cmds = VK_NULL_HANDLE;

  const VkDeviceSize tiles_offset = cmds_size;
  const VkDeviceSize tiles_size = gwidth * gheight * sizeof(tile_t);
  tile_t *tiles_ptr() { return (tile_t*)(mapped_mem + tiles_offset); }
  VkBuffer tiles = VK_NULL_HANDLE;

  const VkDeviceSize texes_offset = tiles_offset + tiles_size;
  const VkDeviceSize texes_size = (max_copies + 1) * sizeof(tex_t) * 8;
  tex_t *texes_ptr() { return (tex_t*)(mapped_mem + texes_offset) + n_tmems * 8; }
  VkBuffer texes = VK_NULL_HANDLE;

  const VkDeviceSize globals_offset = texes_offset + texes_size;
  const VkDeviceSize globals_size = sizeof(global_t);
  global_t *globals_ptr() { return (global_t*)(mapped_mem + globals_offset); }
  VkBuffer globals = VK_NULL_HANDLE;

  const VkDeviceSize tmem_offset = globals_offset + globals_size;
  const VkDeviceSize tmem_size = (max_copies + 1) << 12;
  uint8_t *tmem_ptr() { return mapped_mem + tmem_offset + (n_tmems << 12); }
  VkBuffer tmem = VK_NULL_HANDLE;

  const VkDeviceSize pixels_offset = tmem_offset + tmem_size;
  const VkDeviceSize pixels_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *pixels_ptr() { return mapped_mem + pixels_offset; }
  VkBuffer pixels = VK_NULL_HANDLE;

  const VkDeviceSize zbuf_offset = pixels_offset + pixels_size;
  const VkDeviceSize zbuf_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *zbuf_ptr() { return mapped_mem + zbuf_offset; }
  VkBuffer zbuf = VK_NULL_HANDLE;
  const VkDeviceSize total_size = zbuf_offset + zbuf_size;

  /* === Vulkan Initialization == */

  void init_device(const VkInstance &instance, VkPhysicalDevice *gpu) {
    // check all vulkan physical devices
    uint32_t n_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &n_gpus, 0);
    std::vector<VkPhysicalDevice> gpus(n_gpus);
    vkEnumeratePhysicalDevices(instance, &n_gpus, gpus.data());
    for (VkPhysicalDevice gpu_ : gpus) {
      // check queue families on each device
      uint32_t n_queues = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, 0);
      std::vector<VkQueueFamilyProperties> queues(n_queues);
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, &queues[0]);
      // if queue family supports compute, init virtual device
      for (uint32_t i = 0; i < n_queues; ++i) {
        if ((VK_QUEUE_COMPUTE_BIT & queues[i].queueFlags) == 0) continue;
        const float priorities = 1.0;
        const VkDeviceQueueCreateInfo queue_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = (queue_idx = i), .queueCount = 1,
          .pQueuePriorities = &priorities
        };
        const VkDeviceCreateInfo device_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info
        };
        vkCreateDevice((*gpu = gpu_), &device_info, 0, &device); return;
      }
    }
    printf("No compute queue found\n"); exit(1);
  }

  void init_pipeline(const uint32_t *code, uint32_t code_size,
      VkDescriptorSetLayout *desc_layout, VkPipelineLayout *layout, VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size, .pCode = code
    };
    VkShaderModule module_ = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &module_info, 0, &module_);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 7, .pBindings = bindings
    };
    vkCreateDescriptorSetLayout(device, &desc_layout_info, 0, desc_layout);
    // create compute pipeline with shader and descriptor set layout
    const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = desc_layout
    };
    vkCreatePipelineLayout(device, &layout_info, 0, layout);
    const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = module_, .pName = "main"
    };
    const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info, .layout = *layout
    };
    vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, pipeline);
  }

  void init_buffers(const VkPhysicalDevice &gpu, VkDeviceMemory *memory) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    // find host visible, coherent memory of at least 'size' bytes
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      const VkMemoryType memoryType = props.memoryTypes[i];
      if (!(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memoryType.propertyFlags) ||
          !(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memoryType.propertyFlags) ||
          !(total_size < props.memoryHeaps[memoryType.heapIndex].size)) continue;
      const VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .memoryTypeIndex = i, .allocationSize = total_size
      };
      vkAllocateMemory(device, &allocate_info, 0, memory);
    }
    // exit if memory not allocated
    // create cmds buffer from allocated device memory
    VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = cmds_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &cmds);
    vkBindBufferMemory(device, cmds, *memory, 0);
    // create tiles buffer
    buffer_info.size = tiles_size;
    vkCreateBuffer(device, &buffer_info, 0, &tiles);
    vkBindBufferMemory(device, tiles, *memory, tiles_offset);
    // create texes buffer
    buffer_info.size = texes_size;
    vkCreateBuffer(device, &buffer_info, 0, &texes);
    vkBindBufferMemory(device, texes, *memory, texes_offset);
    // create globals buffer
    buffer_info.size = globals_size;
    vkCreateBuffer(device, &buffer_info, 0, &globals);
    vkBindBufferMemory(device, globals, *memory, globals_offset);
    // create tmem buffer
    buffer_info.size = tmem_size;
    vkCreateBuffer(device, &buffer_info, 0, &tmem);
    vkBindBufferMemory(device, tmem, *memory, tmem_offset);
    // create pixels buffer
    buffer_info.size = pixels_size;
    vkCreateBuffer(device, &buffer_info, 0, &pixels);
    vkBindBufferMemory(device, pixels, *memory, pixels_offset);
    // create zbuf buffer
    buffer_info.size = zbuf_size;
    vkCreateBuffer(device, &buffer_info, 0, &zbuf);
    vkBindBufferMemory(device, zbuf, *memory, zbuf_offset);
  }

  void init_descriptors(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 7
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo descriptors_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &descriptors_info, descriptors);
    // bind buffers to descriptor set
    const VkDescriptorBufferInfo buffer_info[] = {
      { .buffer = cmds, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = tiles, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = texes, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = globals, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = tmem, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = pixels, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = zbuf, .offset = 0, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet write_descriptors[7];
    for (uint8_t i = 0; i < 7; ++i) {
      write_descriptors[i] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[i]
      };
    }
    vkUpdateDescriptorSets(device, 7, write_descriptors, 0, 0);
  }
  
  void record_commands(const VkPipelineLayout &layout, const VkPipeline &pipeline,
      const VkDescriptorSet &descriptors) {
    // allocate command buffer from command pool
    const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_idx
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &pool_info, 0, &pool);
    const VkCommandBufferAllocateInfo commands_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .commandBufferCount = 1,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    };
    vkAllocateCommandBuffers(device, &commands_info, &commands);
    // record commands - bind descriptor set to set layout, start pipeline
    const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    vkBeginCommandBuffer(commands, &begin_info);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
      layout, 0, 1, &descriptors, 0, 0);
    vkCmdDispatch(commands, gwidth, gheight, 1);
    vkEndCommandBuffer(commands);
  }

  void init() {
    // setup Vulkan instance, logical device
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    const VkApplicationInfo app_info = {
      .pApplicationName = "nmulator RDP",
      .apiVersion = VK_API_VERSION_1_0
    };
    const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
    };
    vkCreateInstance(&instance_info, 0, &instance);
    init_device(instance, &gpu);

    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    // use structured bindings instead of output params?
    init_pipeline(shader, sizeof(shader), &desc_layout, &layout, &pipeline);
    init_buffers(gpu, &memory);
    init_descriptors(desc_layout, &descriptors);
    record_commands(layout, pipeline, descriptors);
    
    // map memory so buffers can filled by cpu
    vkMapMemory(device, memory, 0, total_size, 0, reinterpret_cast<void**>(&mapped_mem));
    memset(tiles_ptr(), 0, tiles_size);

    uint32_t *zbuf = reinterpret_cast<uint32_t*>(zbuf_ptr());
    for (uint32_t i = 0; i < 320 * 240; ++i) zbuf[i] = 0x3ffff0;

    /*zlut_t *zlut = Vulkan::globals_ptr().zlut;
    for (uint8_t i = 0; i < 0x88; ++i) {
      uint32_t &shl = zlut[i].shift;
      uint32_t &exp = zlut[i].exponent;
      if (i < 0x40) shl = 6, exp = 0;
      else if (i < 0x60) shl = 5, exp = 1;
      else if (i < 0x70) shl = 4, exp = 2;
      else if (i < 0x78) shl = 3, exp = 3;
      else if (i < 0x7c) shl = 2, exp = 4;
      else if (i < 0x7e) shl = 1, exp = 5;
      else if (i < 0x7f) shl = 0, exp = 6;
      else if (i < 0x80) shl = 0, exp = 7;
    }*/
  }

  /* === Runtime Methods === */

  void add_tmem_copy() {
    uint8_t *last_tmem = tmem_ptr();
    tex_t *last_texes = texes_ptr();
    if (++n_tmems >= max_copies) {
      printf("[RDP] max_copies reached\n");
      RDP::render(), n_tmems = 0;
    }
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(tex_t) * 8);
  }

  void run_commands() {
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_idx, 0, &queue);
    const VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &commands
    };
    vkQueueSubmit(queue, 1, &submitInfo, 0);
    vkQueueWaitIdle(queue);
  }

  void add_rdp_cmd(cmd_t cmd) {
    uint32_t yh = (cmd.yh < 0 ? 0 : cmd.yh / (group_size << 2));
    uint32_t yl = (cmd.yl < 0 ? 0 : cmd.yl / (group_size << 2));

    /*uint32_t xh = cmd.xyh[0] + (cmd.sh * (cmd.xyl[1] - cmd.xyh[1]) >> 2);
    xh = (xh < 0 ? 0 : xh / (group_size << 16));
    uint32_t xl = (cmd.xyl[0] < 0 ? 0 : cmd.xyl[0] / (group_size << 16));
    uint32_t x1 = (cmd.lft ? xl : xh), x2 = (cmd.lft ? xh : xl);
    if (cmd.type & 0x8) {
      x1 = cmd.xyh[0] >> 2, x2 = cmd.xyl[0] >> 2;
    }*/
    uint32_t x1 = 0, x2 = gwidth + 1;

    for (uint32_t i = yh; i <= yl && i < gheight; ++i) {
      for (uint32_t j = x1; j <= x2 && j < gwidth; ++j) {
        tile_t *tile = tiles_ptr() + i * gwidth + j;
        tile->cmd_idxs[n_cmds >> 5] |= 0x1 << (n_cmds & 0x1f);
        //printf("cmd_idxs: %x\n", tile->cmd_idxs[0]);
      }
    }
    cmds_ptr()[n_cmds++] = cmd;
    if (n_cmds >= max_cmds) {
      printf("[RDP] max_cmds exceeded\n");
      RDP::render();
    }
  }

  void render(uint8_t *pixels, uint8_t *zbuf, uint32_t len) {
    if (!pixels || n_cmds == 0) return;
    memcpy(pixels_ptr(), pixels, len);
    //if (zbuf) memcpy(zbuf_ptr(), zbuf, len);
    run_commands(); n_cmds = 0; memset(tiles_ptr(), 0, tiles_size);
    memcpy(pixels, pixels_ptr(), len);
    //if (zbuf) memcpy(zbuf, zbuf_ptr(), len);
    
    // periodically clear zbuffer
    uint32_t *zbuf2 = reinterpret_cast<uint32_t*>(zbuf_ptr());
    for (uint32_t i = 0; i < 320 * 240; ++i) zbuf2[i] = 0x3ffff0;
  }
}

namespace RSP {
  extern uint64_t reg_array[0x100];
  extern const uint8_t dev_cop0;
  template <typename T, bool all>
  int64_t read(uint32_t addr);
  extern bool step;
}

namespace R4300 {
  extern bool logging_on;
  extern uint8_t *pages[0x100];
  void set_irqs(uint32_t mask);
  template <typename T, bool map>
  int64_t read(uint32_t addr);
}

namespace RDP {
  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;
  uint64_t &pc_start = rsp_cop0[8], &pc_end = rsp_cop0[9];
  uint64_t &pc = rsp_cop0[10], &status = rsp_cop0[11];

  uint32_t img_size = 0x0, img_width = 0, height = 240;
  uint8_t *img_addr = nullptr, *zbuf_addr = nullptr;
  uint32_t tex_nibs = 0x0, tex_width = 0, tex_addr = 0x0;
  //bool tile_dirty[8] = {false};

  uint32_t fill = 0x0, fog = 0x0, blend = 0x0;
  uint32_t env = 0x0, prim = 0x0, zprim = 0x0;
  uint32_t cc_mux = 0x0, tlut = 0x0;
  uint32_t keys[3] = {0}, modes[2] = {0};
  bounds_t sci;

  inline int32_t sext(uint32_t val, uint32_t bits=32) {
    if (bits >= 32) return val;
    uint32_t mask = (1 << bits) - 1, sign = 1 << (bits - 1);
    return ((val & mask) ^ sign) - sign;
  }

  inline int32_t zext(uint32_t val, uint32_t bits=32) {
    if (bits >= 32) return val;
    return val & ((1 << bits) - 1);
  }

  uint8_t opcode(uint64_t addr) {
    uint32_t out = 0;
    if (status & 0x1) out = RSP::read<uint32_t>(addr);
    else out = R4300::read<uint32_t, false>(addr);
    return (out >> 24) & 0x3f;
  }

  std::vector<uint32_t> fetch(uint64_t &addr, uint8_t len) {
    std::vector<uint32_t> out(len);
    for (uint8_t i = 0; i < len; ++i, addr += 4) {
      if (status & 0x1) out[i] = RSP::read<uint32_t>(addr);
      else out[i] = R4300::read<uint32_t, false>(addr);
    }
    return out;
  }

  void render() {
    //memset(tile_dirty, 0, 8);
    Vulkan::render(img_addr, zbuf_addr, img_width * height * img_size);
  }

  /* === Instruction Translations === */

  void set_image() {
    //render();
    std::vector<uint32_t> instr = fetch(pc, 2);
    img_size = 1 << (((instr[0] >> 19) & 0x3) - 1);
    img_width = (instr[0] & 0x3ff) + 1;
    img_addr = R4300::pages[0] + (instr[1] & 0x3ffffff);

    global_t *globals = Vulkan::globals_ptr();
    globals->width = img_width, globals->size = img_size;
    Vulkan::gwidth = img_width / Vulkan::group_size;
  }

  void set_scissor() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    sci.xh = zext(instr[0] >> 12, 12) << 14, sci.yh = zext(instr[0], 12);
    sci.xl = zext(instr[1] >> 12, 12) << 14, sci.yl = sext(instr[1], 12);
    height = (sci.yl >> 2) - (sci.yh >> 2);
    //uint32_t yh = instr[0] & 0xfff, yl = instr[1] & 0xfff;
    //height = (yl >> 2) - (yh >> 2);
    Vulkan::gheight = height / Vulkan::group_size;
  }

  void set_other_modes() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    modes[0] = instr[0], modes[1] = instr[1];
  }

  void set_fill() {
    fill = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_fog() {
    fog = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_blend() {
    blend = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_combine() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    cc_mux = ((instr[0] >> 31) & 0x7fff) << 16;
    cc_mux |= (instr[1] >> 9) & 0x1ff;
  }

  void set_env() {
    env = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_prim() {
    prim = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_zprim() {
    zprim = fetch(pc, 2)[1];
  }

  void set_zbuf() {
    zbuf_addr = R4300::pages[0] + (fetch(pc, 2)[1] & 0x3ffffff);
  }

  void set_key_r() {
    keys[0] = __builtin_bswap32(fetch(pc, 2)[1]);
  }

  void set_key_gb() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    keys[2] = (instr[1] & 0xffff) | ((instr[0] & 0xfff) << 16);
    keys[1] = (instr[1] >> 16) | ((instr[0] & 0xfff000) >> 4);
  }

  void set_texture() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    tex_nibs = 0x1 << ((instr[0] >> 19) & 0x3);
    tex_width = (instr[0] & 0x3ff) + 1;
    tex_addr = instr[1] & 0x3ffffff;
    printf("Set Texture Image: %x, width: %d\n", tex_addr, tex_width);
  }

  void set_tile() {
    Vulkan::add_tmem_copy();
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint8_t tex_idx = (instr[1] >> 24) & 0x7;
    //if (tile_dirty[tex_idx]) render();
    //else tile_dirty[tex_idx] = true;
    Vulkan::texes_ptr()[tex_idx] = {
      .format = (instr[0] >> 21) & 0x7, .size = (instr[0] >> 19) & 0x3,
      .width = ((instr[0] >> 9) & 0xff) << 3,
      .addr = (instr[0] & 0x1ff) << 3, .pal = (instr[1] >> 20) & 0xf,
      .mask = { (instr[1] >> 4) & 0xf, (instr[1] >> 14) & 0xf },
      .shift = { instr[1] & 0xf, (instr[1] >> 10) & 0xf }
    };
  }

  void load_tile() {
    Vulkan::add_tmem_copy();
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t sh = (instr[1] >> 12) & 0xfff, th = instr[1] & 0xfff;
    uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
    th >>= 2, tl >>= 2, sh >>= 2, sl >>= 2;
    tex_t tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;
    printf("[RDP] Load Tile %d, %d, %d, %d to tmem %x\n", sl, tl, sh, th, tex.addr);

    uint32_t offset = (tl * tex_width + sl) * tex_nibs / 2;
    uint32_t width = (sh - sl + 1) * tex_nibs / 2;
    uint32_t ram = tex_addr + offset;
    printf("First RAM address %x: %x\n", ram, *(uint32_t*)(R4300::pages[0] + ram));
    for (uint32_t i = 0; i <= th - tl; ++i) {
      if (tex_nibs == 8) {
        for (uint32_t i = 0; i < width; i += 4) {
          memcpy(mem + i / 2, R4300::pages[0] + ram + i, 2);
          memcpy(mem + 0x800 + i / 2, R4300::pages[0] + ram + i + 2, 2);
        }
      } else memcpy(mem, R4300::pages[0] + ram, width);
      mem += tex.width, ram += tex_width * tex_nibs / 2;
    }
  }

  void load_block() {
    Vulkan::add_tmem_copy();
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t sh = (instr[1] >> 12) & 0xfff/*, dxt = instr[1] & 0xfff*/;
    uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
    tex_t tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;

    uint32_t offset = (tl * tex_width + sl) * tex_nibs / 2;
    uint32_t width = (sh - sl + 1) * tex_nibs / 2;
    memcpy(mem, R4300::pages[0] + tex_addr + offset, width);
  }

  void load_tlut() {
    Vulkan::add_tmem_copy();
    std::vector<uint32_t> instr = fetch(pc, 2);
    uint32_t sh = (instr[1] >> 14) & 0xff;
    uint32_t sl = (instr[0] >> 14) & 0xff;
    tex_t tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    uint8_t *mem = Vulkan::tmem_ptr() + (tlut = tex.addr);

    uint32_t ram = tex_addr + sl * tex_nibs / 2;
    uint32_t width = (sh - sl + 1) * tex_nibs / 2;
    memcpy(mem, R4300::pages[0] + ram, width);
  }

  void shade_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 16);
    for (uint8_t i = 0; i < 16; i += 2)
      printf("%x %x\n", instr[i], instr[i + 1]);
    cmd.shade[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
    cmd.shade[1] = (instr[0] << 16) | (instr[4] & 0xffff);
    cmd.shade[2] = (instr[1] & 0xffff0000) | (instr[5] >> 16);
    cmd.shade[3] = (instr[1] << 16) | (instr[5] & 0xffff);
    cmd.sde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
    cmd.sde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
    cmd.sde[2] = (instr[9] & 0xffff0000) | (instr[13] >> 16);
    cmd.sde[3] = (instr[9] << 16) | (instr[13] & 0xffff);
    cmd.sdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
    cmd.sdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
    cmd.sdx[2] = (instr[3] & 0xffff0000) | (instr[7] >> 16);
    cmd.sdx[3] = (instr[3] << 16) | (instr[7] & 0xffff);
  }

  void tex_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 16);
    for (uint8_t i = 0; i < 16; i += 2)
      printf("%x %x\n", instr[i], instr[i + 1]);
    cmd.tex[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
    cmd.tex[1] = (instr[0] << 16) | (instr[4] & 0xffff);
    cmd.tex[2] = (instr[1] & 0xffff0000) | (instr[5] >> 16);
    cmd.tde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
    cmd.tde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
    cmd.tde[2] = (instr[9] & 0xffff0000) | (instr[13] >> 16);
    cmd.tdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
    cmd.tdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
    cmd.tdx[2] = (instr[3] & 0xffff0000) | (instr[7] >> 16);
    cmd.tmem = Vulkan::n_tmems;
  }

  void zbuf_triangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 4);
    for (uint8_t i = 0; i < 4; i += 2)
      printf("%x %x\n", instr[i], instr[i + 1]);
    cmd.zpos = instr[0], cmd.zde = instr[2];
    cmd.zdx = instr[1];
  }

  template <uint8_t type>
  void triangle() {
    printf("[RDP] Triangle of type %x\n", type);
    std::vector<uint32_t> instr = fetch(pc, 8);
    for (uint8_t i = 0; i < 8; i += 2)
      printf("%x %x\n", instr[i], instr[i + 1]);

    /*if (R4300::logging_on)
      printf("Ending triangle hit\n"), exit(0);
    if (instr[0] == 0xce0001d8  && instr[1] == 0x1d301d3
      && instr[2] == 0x9ec000 && instr[3] == 0x1cccc)
      R4300::logging_on = RSP::step = true;*/

    uint32_t t = type | ((instr[0] >> 19) & 0x10);
    cmd_t cmd = {
      .type = t, .tile = (instr[0] >> 16) & 0x7,
      .xh = sext(instr[4], 30), .yh = sext(instr[1], 14),
      .xm = sext(instr[6], 30), .ym = sext(instr[1] >> 16, 14),
      .xl = sext(instr[2], 30), .yl = sext(instr[0], 14),
      .sh = sext(instr[5]), .sm = sext(instr[7]), .sl = sext(instr[3]),

      .fill = fill, .fog = fog, .blend = blend,
      .env = env, .prim = prim, .zprim = zprim,
      .cc_mux = cc_mux, .tlut = tlut,
      .sci = { sci.xh, sci.xl, sci.yh, sci.yl },
      .keys = { keys[0], keys[1], keys[2] },
      .modes = { modes[0], modes[1] },
    };
    if (type & 0x4) shade_triangle(cmd);
    if (type & 0x2) tex_triangle(cmd);
    if (type & 0x1) zbuf_triangle(cmd);
    if (pc <= pc_end) Vulkan::add_rdp_cmd(cmd);
  }

  template <bool flip>
  void tex_rectangle(cmd_t &cmd) {
    std::vector<uint32_t> instr = fetch(pc, 2);
    printf("%x %x\n", instr[0], instr[1]);
    cmd.tex[0] = (flip ? instr[0] & 0xffff : instr[0] >> 16) << 6;
    cmd.tex[1] = (flip ? instr[0] >> 16 : instr[0] & 0xffff << 6);
    cmd.tde[0] = (instr[1] & 0xffff) << 11;
    cmd.tdx[0] = (instr[1] >> 16) << 11;
    cmd.tmem = Vulkan::n_tmems;
  }

  template <uint8_t type>
  void rectangle() {
    printf("[RDP] Rectangle of type %x\n", type);
    printf("TMEM 0: %x\n", *(uint32_t*)Vulkan::tmem_ptr());

    std::vector<uint32_t> instr = fetch(pc, 2);
    if (type & 0x2) printf("%x %x\n", instr[0], instr[1]);
    cmd_t cmd = {
      .type = type, .tile = (instr[1] >> 24) & 0x7,
      .xh = zext(instr[1] >> 12, 12) << 14, .yh = zext(instr[1], 12),
      .xl = zext(instr[0] >> 12, 12) << 14, .yl = zext(instr[0], 12),

      .fill = fill, .fog = fog, .blend = blend,
      .env = env, .prim = prim, .zprim = zprim,
      .cc_mux = cc_mux, .tlut = tlut,
      .sci = { sci.xh, sci.xl, sci.yh, sci.yl },
      .keys = { keys[0], keys[1], keys[2] },
      .modes = { modes[0], modes[1] },
    };
    if (type == 0xa) tex_rectangle<false>(cmd);
    if (type == 0xb) tex_rectangle<true>(cmd);
    if (pc <= pc_end) Vulkan::add_rdp_cmd(cmd);
  }

  void invalid() {
    std::vector<uint32_t> instr = fetch(pc, 2);
    printf("[RDP] Unimplemented instruction %x%x\n", instr[0], instr[1]);
    //exit(1);
  }

  uint64_t off = 0;

  void update() {
    if (pc >= pc_end) return;
    pc -= off, off = 0;
    // interpret config instructions 
    uint32_t cycles = 0;
    while (still_top(cycles)) {
      uint64_t start = pc;
      std::vector<uint32_t> instr = fetch(pc, 4);
      printf("[RDP] Command %x %x\n", instr[0], instr[1]);
      pc -= 16;
      /*if (instr[0] == 0xe43c02dc && instr[1] == 0x1442cc
          && instr[2] == 0xe7000000 && instr[3] == 0x0)
        R4300::logging_on = RSP::step = true;
      if (instr[0] == 0xce0001f0 && instr[1] == 0x1f001dc
        && instr[2] == 0x9a8000 && instr[3] == 0x2afffd)
        printf("Ending triangle hit\n"), exit(0);*/
      switch (opcode(pc)) {
        case 0x00: pc += 8; break;
        case 0x08: triangle<0x0>(); break;
        case 0x09: triangle<0x1>(); break;
        case 0x0a: triangle<0x2>(); break;
        case 0x0b: triangle<0x3>(); break;
        case 0x0c: triangle<0x4>(); break;
        case 0x0d: triangle<0x5>(); break;
        case 0x0e: triangle<0x6>(); break;
        case 0x0f: triangle<0x7>(); break;
        case 0x24: rectangle<0xa>(); break;
        case 0x25: rectangle<0xb>(); break;
        case 0x2a: set_key_gb(); break;
        case 0x2b: set_key_r(); break;
        case 0x2c: printf("[RDP] SET CONVERT\n"); pc += 8; break;
        case 0x2d: set_scissor(); break;
        case 0x2e: set_zprim(); break;
        case 0x2f: set_other_modes(); break;
        case 0x30: load_tlut(); break;
        case 0x32: printf("[RDP] SET TILE SIZE\n"); pc += 8; break;
        case 0x33: load_block(); break;
        case 0x34: load_tile(); break;
        case 0x35: set_tile(); break;
        case 0x36: rectangle<0x8>(); break;
        case 0x37: set_fill(); break;
        case 0x38: set_fog(); break;
        case 0x39: set_blend(); break;
        case 0x3a: set_prim(); break;
        case 0x3b: set_env(); break;
        case 0x3c: set_combine(); break;
        case 0x3d: set_texture(); break;
        case 0x3e: set_zbuf(); break;
        case 0x3f: set_image(); break;
        case 0x29: R4300::set_irqs(0x20); render();
        case 0x26: case 0x27: case 0x28: pc += 8; break;
        default: invalid(); break;
      }
      if (pc > pc_end) pc = pc_end, off = pc_end - start;
      if (pc == pc_end) { status |= 0x80; return; }
      cycles = 1;
    }
    sched(update, cycles);
  }
}

#endif
