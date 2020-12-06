#include <vulkan/vulkan.h>
#include <array>
#include <string.h>
#include <stdio.h>
#include "nmulator.h"
#include "comp.spv"

/* === Shader Structs === */

// all coeffs are 16.16 fixed point
// except yh/yl/ym, which is 12.2

#pragma pack(push, 1)
struct TileData {
  uint32_t cmd_idxs[64];
};

struct GlobalData {
  uint32_t width, size;
  uint32_t n_cmds, fmt;
};

struct RDPState {
  int32_t sxh, sxl, syh, syl;
  uint32_t modes[2], mux[2];
  uint32_t tlut, tmem, texes;
  uint32_t fill, fog, blend;
  uint32_t env, prim, lodf;
  uint32_t zprim, keys, keyc;
};

struct RDPCommand {
  uint32_t type, tile, pad;
  int32_t xh, xm, xl;
  int32_t yh, ym, yl;
  int32_t sh, sm, sl;

  int32_t shade[4], sde[4], sdx[4];
  int32_t tex[4], tde[4], tdx[4];
  RDPState state;
};

struct RDPTex {
  uint32_t format, size;
  uint32_t width, addr;
  int32_t sth[2], stl[2], shift[2];
  uint32_t pal, pad;
};
#pragma pack(pop)

struct RenderInfo {
  uint8_t *img, *himg;
  uint8_t *zbuf, *hzbuf;
  uint32_t img_len, zbuf_len;
  bool pending;
};

/* === Vulkan Setup === */

namespace RDP {
  void render(bool sync);
}

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;

  constexpr VkDeviceSize align(VkDeviceSize offset) {
    return (offset + 0x1f) & ~0x1f;
  }

  /* === Descriptor Memory Access === */

  void *vmems[2];
  uint8_t *mapped_mem = nullptr;
  const uint32_t group_size = 8, max_cmds = 2048, max_copies = 4095;
  uint32_t gwidth = 440 / group_size, gheight = 240 / group_size;
  uint32_t n_cmds = 0, n_tmems = 0, mem_idx = 1;

  const VkDeviceSize cmds_offset = 0;
  const VkDeviceSize cmds_size = max_cmds * sizeof(RDPCommand);
  RDPCommand *cmds_ptr() { return (RDPCommand*)(mapped_mem); }

  const VkDeviceSize tiles_offset = align(cmds_offset + cmds_size);
  const VkDeviceSize tiles_size = gwidth * gheight * sizeof(TileData);
  TileData *tiles_ptr() { return (TileData*)(mapped_mem + tiles_offset); }

  const VkDeviceSize texes_offset = align(tiles_offset + tiles_size);
  const VkDeviceSize texes_size = (max_copies + 1) * sizeof(RDPTex) * 8;
  RDPTex *texes_ptr() { return (RDPTex*)(mapped_mem + texes_offset) + n_tmems * 8; }

  const VkDeviceSize globals_offset = align(texes_offset + texes_size);
  const VkDeviceSize globals_size = sizeof(GlobalData);
  GlobalData *globals_ptr() { return (GlobalData*)(mapped_mem + globals_offset); }

  const VkDeviceSize tmem_offset = align(globals_offset + globals_size);
  const VkDeviceSize tmem_size = (max_copies + 1) << 12;
  uint8_t *tmem_ptr() { return mapped_mem + tmem_offset + (n_tmems << 12); }

  const VkDeviceSize pixels_offset = align(tmem_offset + tmem_size);
  const VkDeviceSize pixels_size = 440 * 240 * sizeof(uint32_t);
  uint8_t *pixels_ptr() { return mapped_mem + pixels_offset; }

  const VkDeviceSize hpixels_offset = align(pixels_offset + pixels_size);
  const VkDeviceSize hpixels_size = 440 * 240 * sizeof(uint32_t);
  uint8_t *hpixels_ptr() { return mapped_mem + hpixels_offset; }

  const VkDeviceSize zbuf_offset = align(hpixels_offset + hpixels_size);
  const VkDeviceSize zbuf_size = 440 * 240 * sizeof(uint16_t);
  uint8_t *zbuf_ptr() { return mapped_mem + zbuf_offset; }

  const VkDeviceSize hzbuf_offset = align(zbuf_offset + zbuf_size);
  const VkDeviceSize hzbuf_size = 440 * 240 * sizeof(uint16_t);
  uint8_t *hzbuf_ptr() { return mapped_mem + hzbuf_offset; }

  const VkDeviceSize total_size = hzbuf_offset + hzbuf_size;
  VkCommandBuffer comp_cmds[2];
  VkFence fences[2];
  VkBuffer buffers[2];
  RenderInfo renders[2];

  /* === Vulkan Initialization == */

  void init_instance(VkInstance *instance) {
    const char *layers[] = { "VK_LAYER_KHRONOS_validation" };
    const VkApplicationInfo app_info = {
      .pApplicationName = "nmulator RDP",
      .apiVersion = VK_API_VERSION_1_0
    };
    const char *exts[] = { "VK_KHR_get_physical_device_properties2" };
    const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = 1, .ppEnabledLayerNames = layers,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts
    };
    vkCreateInstance(&instance_info, 0, instance);
  }

  void init_device(const VkInstance &instance, VkPhysicalDevice *gpu) {
    // check all vulkan physical devices
    uint32_t n_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &n_gpus, 0);
    std::array<VkPhysicalDevice, 16> gpus;
    vkEnumeratePhysicalDevices(instance, &n_gpus, gpus.data());
    for (VkPhysicalDevice gpu_ : gpus) {
      // check queue families on each device
      uint32_t n_queues = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, 0);
      std::array<VkQueueFamilyProperties, 16> queues;
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, queues.data());
      // if queue family supports compute, init virtual device
      for (uint32_t i = 0; i < n_queues; ++i) {
        if (~queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) continue;
        const float priority = 1.0;
        const VkDeviceQueueCreateInfo queue_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = (queue_idx = i),
          .queueCount = 1, .pQueuePriorities = &priority
        };
        const char *exts[] = {
          "VK_KHR_storage_buffer_storage_class",
          "VK_KHR_16bit_storage", "VK_KHR_8bit_storage"
        };
        const VkDeviceCreateInfo device_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
          .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts
        };
        vkCreateDevice((*gpu = gpu_), &device_info, 0, &device);
        vkGetDeviceQueue(device, queue_idx, 0, &queue);
        return;
      }
    }
    printf("No compatible GPU found\n"), exit(1);
  }

  void init_compute(VkDescriptorSetLayout *desc_layout,
      VkPipelineLayout *layout, VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo comp_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g_main), .pCode = (uint32_t*)g_main,
    };
    VkShaderModule comp;
    vkCreateShaderModule(device, &comp_info, 0, &comp);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 9, .pBindings = bindings
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
      .module = comp, .pName = "main"
    };
    const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info, .layout = *layout
    };
    vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, pipeline);
    vkDestroyShaderModule(device, comp, nullptr);
  }

  void init_buffers(const VkPhysicalDevice &gpu, VkDeviceMemory *memory) {
    // create shared buffer to hold descriptors
    const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = total_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &buffers[0]);
    vkCreateBuffer(device, &buffer_info, 0, &buffers[1]);
    // get memory requirements for buffer
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffers[0], &requirements);
    VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size
    };
    // find host visible, coherent memory meeting requirements
    const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      const VkMemoryType memory_type = props.memoryTypes[i];
      if ((memory_type.propertyFlags & flags) != flags) continue;
      allocate_info.memoryTypeIndex = i;
      vkAllocateMemory(device, &allocate_info, 0, &memory[0]);
      vkAllocateMemory(device, &allocate_info, 0, &memory[1]);
      break;
    }
    vkBindBufferMemory(device, buffers[0], memory[0], 0);
    vkBindBufferMemory(device, buffers[1], memory[1], 0);
  }

  void init_compute_desc(const VkDescriptorSetLayout &layout, VkDescriptorSet *desc) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * 9
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &desc_info, &desc[0]);
    vkAllocateDescriptorSets(device, &desc_info, &desc[1]);
    // bind buffers to descriptor set
    for (uint32_t i = 0; i < 2; ++i) {
      const VkDescriptorBufferInfo buffer_info[] = {
        { .buffer = buffers[i], .offset = cmds_offset, .range = cmds_size },
        { .buffer = buffers[i], .offset = tiles_offset, .range = tiles_size },
        { .buffer = buffers[i], .offset = texes_offset, .range = texes_size },
        { .buffer = buffers[i], .offset = globals_offset, .range = globals_size },
        { .buffer = buffers[i], .offset = tmem_offset, .range = tmem_size },
        { .buffer = buffers[i], .offset = pixels_offset, .range = pixels_size },
        { .buffer = buffers[i], .offset = hpixels_offset, .range = hpixels_size },
        { .buffer = buffers[i], .offset = zbuf_offset, .range = zbuf_size },
        { .buffer = buffers[i], .offset = hzbuf_offset, .range = hzbuf_size }
      };
      const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = desc[i], .dstBinding = 0, .descriptorCount = 9,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = buffer_info
      };
      vkUpdateDescriptorSets(device, 1, &write, 0, 0);
    };
  }
  
  void record_compute(const VkPipelineLayout &layout,
      const VkPipeline &pipeline, const VkDescriptorSet *desc) {
    // allocate command buffer from command pool
    const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_idx
    };
    VkCommandPool pool;
    vkCreateCommandPool(device, &pool_info, 0, &pool);
    const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2
    };
    vkAllocateCommandBuffers(device, &cmd_info, comp_cmds);
    const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    // record commands - bind descriptor set and pipeline
    for (uint32_t i = 0; i < 2; ++i) {
      vkBeginCommandBuffer(comp_cmds[i], &begin_info);
      vkCmdBindPipeline(comp_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(comp_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE,
        layout, 0, 1, &desc[i], 0, 0);
      vkCmdDispatch(comp_cmds[i], gwidth, gheight, 1);
      VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
      };
      vkCmdPipelineBarrier(comp_cmds[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, 0, 0, 0);
      vkEndCommandBuffer(comp_cmds[i]);
    }
    // create fence for each command buffer
    VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    vkCreateFence(device, &fence_info, 0, &fences[0]);
    vkCreateFence(device, &fence_info, 0, &fences[1]);
  }
  
  void init() {
    // configure GPU, display, and memory
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDeviceMemory memory[2];

    init_instance(&instance);
    init_device(instance, &gpu);
    init_buffers(gpu, memory);

    // setup compute pipeline work
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkPipeline pipeline;
    VkDescriptorSet desc[2];

    init_compute(&desc_layout, &layout, &pipeline);
    init_compute_desc(desc_layout, desc);
    record_compute(layout, pipeline, desc);
    
    // map memory so buffers can filled by cpu
    vkMapMemory(device, memory[0], 0, total_size, 0, &vmems[0]);
    vkMapMemory(device, memory[1], 0, total_size, 0, &vmems[1]);
    mapped_mem = (uint8_t*)vmems[mem_idx];
    memset(tiles_ptr(), 0, tiles_size);
  }

  /* === Runtime Methods === */

  void add_tmem_copy(RDPState &state) {
    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    state.tmem = ++n_tmems;
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);
    if (n_tmems >= max_copies) RDP::render(0);
  }

  void add_rdp_cmd(RDPCommand cmd) {
    uint32_t yh = (cmd.yh < 0 ? 0 : cmd.yh / (group_size << 2));
    uint32_t yl = (cmd.yl < 0 ? 0 : cmd.yl / (group_size << 2));
    for (uint32_t i = yh; i <= yl && i < gheight; ++i) {
      for (uint32_t j = 0; j < gwidth; ++j) {
        TileData *tile = tiles_ptr() + i * gwidth + j;
        tile->cmd_idxs[n_cmds >> 5] |= 0x1 << (n_cmds & 0x1f);
      }
    }
    cmds_ptr()[n_cmds++] = cmd;
    if (n_cmds >= max_cmds) RDP::render(0);
  }

  void flush_compute(uint8_t idx) {
    const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &comp_cmds[idx],
    };
    vkQueueSubmit(queue, 1, &submit_info, fences[idx]);
    renders[idx].pending = true;
  }

  bool wait_compute(uint8_t idx) {
    if (!renders[idx].pending) return true;
    vkWaitForFences(device, 1, &fences[idx], VK_TRUE, -1);
    vkResetFences(device, 1, &fences[idx]);
    return renders[idx].pending = false;
  }

  void sync() {
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
    if (!wait_compute(mem_idx)) {
      RenderInfo *r = &renders[mem_idx];
      if (r->zbuf) {
        memcpy(r->zbuf, zbuf_ptr(), r->zbuf_len);
        memcpy(r->hzbuf, hzbuf_ptr(), r->zbuf_len);
      }
      if (r->img) {
        memcpy(r->img, pixels_ptr(), r->img_len);
        memcpy(r->himg, hpixels_ptr(), r->img_len);
      }
    }
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
  }

  void render(RenderInfo *r) {
    if (!r->img || n_cmds == 0) return;
    if (r->img_len > pixels_size) r->img_len = pixels_size;
    if (r->zbuf_len > zbuf_size) r->zbuf_len = zbuf_size;
    memcpy(&renders[mem_idx], r, sizeof(RenderInfo));

    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    sync(); // drain other buffer

    // upload current buffer to GPU
    globals_ptr()->n_cmds = n_cmds;
    if (r->zbuf) {
      memcpy(zbuf_ptr(), r->zbuf, r->zbuf_len);
      memcpy(hzbuf_ptr(), r->hzbuf, r->zbuf_len);
    }
    memcpy(pixels_ptr(), r->img, r->img_len);
    memcpy(hpixels_ptr(), r->himg, r->img_len);
    flush_compute(mem_idx);

    // change buffers and reset data
    n_cmds = 0, n_tmems = 0;
    mapped_mem = (uint8_t*)vmems[mem_idx ^= 1];
    memset(tiles_ptr(), 0, tiles_size);
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);
  }
}

/* === RDP Interface === */

static uint32_t img_size, img_width, height;
static uint32_t img_addr, zbuf_addr;
static uint32_t tex_nibs, tex_width, tex_addr;
static RDPState state;

namespace RDP {
  uint64_t &pc_start = RSP::cop0[8], &pc_end = RSP::cop0[9];
  uint64_t &pc = RSP::cop0[10], &status = RSP::cop0[11];
}

/* === Helper Functions === */

static int32_t sext(uint32_t val, uint32_t bits=32) {
  if (bits >= 32) return val;
  uint32_t mask = (1 << bits) - 1, sign = 1 << (bits - 1);
  return ((val & mask) ^ sign) - sign;
}

static int32_t zext(uint32_t val, uint32_t bits=32) {
  if (bits >= 32) return val;
  return val & ((1 << bits) - 1);
}

void RDP::render(bool sync) {
  RenderInfo render_info = {
    .img  = R4300::ram + img_addr,  .himg  = R4300::hram + img_addr,
    .zbuf = R4300::ram + zbuf_addr, .hzbuf = R4300::hram + zbuf_addr,
    .img_len = img_width * height * img_size,
    .zbuf_len = img_width * height * 2,
  };
  Vulkan::render(&render_info);
  if (sync) Vulkan::sync();
}

/* === Instruction translation === */

static void set_color_image(uint32_t *instr) {
  if ((instr[1] & R4300::mask) != img_addr) RDP::render(0);
  img_size = 1 << (((instr[0] >> 19) & 0x3) - 1);
  img_width = (instr[0] & 0x3ff) + 1;
  img_addr = instr[1] & R4300::mask;

  GlobalData *globals = Vulkan::globals_ptr();
  globals->width = img_width, globals->size = img_size;
  globals->fmt = (instr[0] >> 19) & 0x1;
  Vulkan::gwidth = img_width / Vulkan::group_size;
}

static void set_depth_image(uint32_t *instr) {
  zbuf_addr = instr[1] & R4300::mask;
}

static void set_scissor(uint32_t *instr) {
  state.sxh = zext(instr[0] >> 12, 12), state.syh = zext(instr[0], 12);
  state.sxl = zext(instr[1] >> 12, 12), state.syl = zext(instr[1], 12);
  height = (state.syl >> 2) - (state.syh >> 2);
  Vulkan::gheight = height / Vulkan::group_size + 1;
  height = Vulkan::gheight * Vulkan::group_size;
}

static void set_other_modes(uint32_t *instr) {
  memcpy(state.modes, instr, 8);
}

static void set_combine(uint32_t *instr) {
  memcpy(state.mux, instr, 8);
}

static void set_fill(uint32_t *instr) {
  state.fill = bswap32(instr[1]);
}

static void set_fog(uint32_t *instr) {
  state.fog = bswap32(instr[1]);
}

static void set_blend(uint32_t *instr) {
  state.blend = bswap32(instr[1]);
}

static void set_env(uint32_t *instr) {
  state.env = bswap32(instr[1]);
}

static void set_prim(uint32_t *instr) {
  state.lodf = (instr[0] & 0xff) << 24;
  state.prim = bswap32(instr[1]);
}

static void set_prim_depth(uint32_t *instr) {
  state.zprim = instr[1];
}

static void set_key_gb(uint32_t *instr) {
  state.keys &= 0xff, state.keyc &= 0xff;
  state.keys |= (instr[1] >>  8) & 0x00ff00;
  state.keys |= (instr[1] <<  8) & 0xff0000;
  state.keyc |= (instr[1] >> 16) & 0x00ff00;
  state.keyc |= (instr[1] <<  4) & 0xff0000;
}

static void set_key_r(uint32_t *instr) {
  state.keys &= 0xffff00, state.keys |= (instr[1] >> 0) & 0xff;
  state.keyc &= 0xffff00, state.keyc |= (instr[1] >> 8) & 0xff;
}

static void set_convert(uint32_t *instr) {
  /*uint64_t cmd = ((uint64_t)instr[0] << 32) | instr[1];
  state.convert[0] = 2 * sext(cmd >> 0, 9) + 1;
  state.convert[1] = 2 * sext(cmd >> 9, 9) + 1;
  state.convert[2] = 2 * sext(cmd >> 18, 9) + 1;
  state.convert[3] = 2 * sext(cmd >> 27, 9) + 1;
  state.convert[4] = sext(cmd >> 36, 9);
  state.convert[5] = sext(cmd >> 45, 9);*/
}

static void set_texture(uint32_t *instr) {
  tex_nibs = 0x1 << ((instr[0] >> 19) & 0x3);
  tex_width = (instr[0] & 0x3ff) + 1;
  tex_addr = instr[1] & R4300::mask;
}

static void set_tile(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  tex.format = (instr[0] >> 21) & 0x7, tex.size = (instr[0] >> 19) & 0x3;
  tex.width = ((instr[0] >> 9) & 0xff) << 3;
  tex.addr = (instr[0] & 0x1ff) << 3, tex.pal = (instr[1] >> 20) & 0xf;
  tex.shift[0] = instr[1] & 0x3ff, tex.shift[1] = (instr[1] >> 10) & 0x3ff;
}

static void set_tile_size(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  tex.sth[0] = (instr[1] >> 12) & 0xfff, tex.sth[1] = instr[1] & 0xfff;
  tex.stl[0] = (instr[0] >> 12) & 0xfff, tex.stl[1] = instr[0] & 0xfff;
}

// handle rgba32 split into hi/lo tmem
static uint32_t taddr(uint32_t addr, uint32_t tex_nibs) {
  if (tex_nibs != 8) return (addr / 2) & 0x7ff;
  uint32_t offs = (addr / 4) & 0x3ff;
  return ((addr & 0x2) << 9) | offs;
}

static void load_tile(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint8_t tex_idx = (instr[1] >> 24) & 0x7;
  RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
  tex.sth[0] = (instr[1] >> 12) & 0xfff, tex.sth[1] = instr[1] & 0xfff;
  tex.stl[0] = (instr[0] >> 12) & 0xfff, tex.stl[1] = instr[0] & 0xfff;

  // copy from texture image to tmem
  uint32_t offset = tex.stl[1] / 4 * tex_width + tex.stl[0] / 4;
  uint32_t len = tex.sth[0] / 4 - tex.stl[0] / 4 + 1, flip = 0;
  offset = offset * tex_nibs / 2, len = len * tex_nibs / 2;
  uint8_t *ram = R4300::ram + tex_addr + offset;
  uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;
  const uint32_t tn = tex_nibs;

  // swap every other 16-bit word on odd rows
  for (int32_t i = 0; i <= (tex.sth[1] - tex.stl[1]) / 4; ++i) {
    for (uint32_t j = 0; j < len; j += 2)
      ((uint16_t*)mem)[taddr(j, tn) ^ flip] = ((uint16_t*)ram)[j / 2];
    mem += tex.width, flip ^= 0x2;
    ram += tex_width * tex_nibs / 2;
  }
}

static void load_block(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint32_t sh = (instr[1] >> 12) & 0xfff, dxt = instr[1] & 0xfff;
  uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
  RDPTex tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];

  // copy from texture image to tmem
  uint32_t offset = (tl * tex_width + sl) * tex_nibs / 2;
  uint32_t len = (sh - sl + 1) * tex_nibs / 2, flip = 0;
  uint16_t *ram = (uint16_t*)(R4300::ram + tex_addr + offset);
  uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;
  const uint32_t tn = tex_nibs;

  // swap every other 16-bit word on odd rows
  for (uint32_t i = 0; i < len;) {
    for (uint32_t t = 0; t < 0x800 && i < len; i += 2) {
      ((uint16_t*)mem)[taddr(i, tn) ^ flip] = *(ram++);
      if ((i & 0x7) == 0x6) t += dxt;
    }
    mem += tex.width, flip ^= 0x2;
  }
}

static void load_tlut(uint32_t *instr) {
  Vulkan::add_tmem_copy(state);
  uint32_t sh = (instr[1] >> 14) & 0x3ff, th = instr[1] & 0xfff;
  uint32_t sl = (instr[0] >> 14) & 0x3ff, tl = (instr[0] >> 2) & 0x3ff;
  RDPTex tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];

  // copy from texture image to tmem
  uint8_t *mem = Vulkan::tmem_ptr() + (state.tlut = tex.addr);
  uint32_t ram = tex_addr + (tl * tex_width + sl) * tex_nibs / 2;
  uint32_t width = (sh - sl + 1) * tex_nibs / 2;

  // quadricate memory while copying
  for (uint32_t i = 0; i < width * 4; ++i) {
    ((uint16_t*)mem)[i] = ((uint16_t*)(R4300::ram + ram))[i / 4];
  }
}

static void shade_triangle(RDPCommand &cmd, uint32_t *instr) {
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

static void tex_triangle(RDPCommand &cmd, uint32_t *instr) {
  cmd.tex[0] = (instr[0] & 0xffff0000) | (instr[4] >> 16);
  cmd.tex[1] = (instr[0] << 16) | (instr[4] & 0xffff);
  cmd.tex[2] = (instr[1] & 0xffff0000) | (instr[5] >> 16);
  cmd.tde[0] = (instr[8] & 0xffff0000) | (instr[12] >> 16);
  cmd.tde[1] = (instr[8] << 16) | (instr[12] & 0xffff);
  cmd.tde[2] = (instr[9] & 0xffff0000) | (instr[13] >> 16);
  cmd.tdx[0] = (instr[2] & 0xffff0000) | (instr[6] >> 16);
  cmd.tdx[1] = (instr[2] << 16) | (instr[6] & 0xffff);
  cmd.tdx[2] = (instr[3] & 0xffff0000) | (instr[7] >> 16);
  if (state.modes[0] & 0x200000) cmd.tdx[0] = 0x200000;
}

static void zbuf_triangle(RDPCommand &cmd, uint32_t *instr) {
  cmd.tex[3] = instr[0], cmd.tde[3] = instr[2];
  cmd.tdx[3] = instr[1];
}

template <uint8_t type>
static void triangle(uint32_t *instr) {
  uint32_t t = type | ((instr[0] >> 19) & 0x10);
  RDPCommand cmd = {
    .type = t, .tile = (instr[0] >> 16) & 0x7,
    .xh = sext(instr[4] >>  1, 27), .xm = sext(instr[6] >> 1, 27),
    .xl = sext(instr[2] >>  1, 27), .yh = sext(instr[1] >> 0, 14),
    .ym = sext(instr[1] >> 16, 14), .yl = sext(instr[0] >> 0, 14),
    .sh = sext(instr[5] >>  3, 27), .sm = sext(instr[7] >> 3, 27),
    .sl = sext(instr[3] >>  3, 27),
  };
  cmd.state = state, instr += 8;
  if (type & 0x4) shade_triangle(cmd, instr), instr += 16;
  if (type & 0x2) tex_triangle(cmd, instr), instr += 16;
  if (type & 0x1) zbuf_triangle(cmd, instr), instr += 4;
  Vulkan::add_rdp_cmd(cmd);
}

template <bool flip>
static void tex_rectangle(RDPCommand &cmd, uint32_t *instr) {
  (flip ? cmd.tex[1] : cmd.tex[0]) = (instr[0] >> 16) << 16;
  (flip ? cmd.tex[0] : cmd.tex[1]) = (instr[0] & 0xffff) << 16;
  (flip ? cmd.tdx[0] : cmd.tde[1]) = (instr[1] & 0xffff) << 11;
  (flip ? cmd.tde[1] : cmd.tdx[0]) = (instr[1] >> 16) << 11;
  if (state.modes[0] & 0x200000) cmd.tdx[0] = 0x200000;
  cmd.tde[0] = 0, cmd.tdx[1] = 0;
}

template <uint8_t type>
static void rectangle(uint32_t *instr) {
  RDPCommand cmd = {
    .type = type, .tile = (instr[1] >> 24) & 0x7,
    .xh = zext(instr[1] >> 12, 12) << 13,
    .xl = zext(instr[0] >> 12, 12) << 13,
    .yh = zext(instr[1] >>  0, 12),
    .yl = zext(instr[0] >>  0, 12),
  };
  cmd.state = state, instr += 2;
  if (state.modes[0] & 0x200000) cmd.yl |= 3;
  if (type == 0xa) tex_rectangle<false>(cmd, instr);
  if (type == 0xb) tex_rectangle<true>(cmd, instr);
  Vulkan::add_rdp_cmd(cmd);
}

static void invalid(uint32_t *instr) {
  const char *msg = "[RDP] Invalid Command %08x %08x\n";
  printf(msg, instr[0], instr[1]);
}

void RDP::update() {
  static uint32_t instr[44], len;
  const uint32_t instr_lengths[] = {
    2, 2, 2, 2, 2, 2, 2, 2, 8, 12, 24, 28, 24, 28, 40, 44,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
    2, 2, 2, 2, 4, 4, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,
  };
  pc &= ~0x7, pc_end &= ~0x7, status |= 0x80;
  uint8_t *src = (status & 0x1 ? RSP::mem : R4300::ram);
  uint32_t mask = (status & 0x1 ? RSP::mask : R4300::mask);

  // read commands into buffer
  for (; pc < pc_end; pc += 8) {
    instr[len++] = read32(src + (pc & mask) + 0);
    instr[len++] = read32(src + (pc & mask) + 4);
    uint8_t opcode = (instr[0] >> 24) & 0x3f;
    if (len < instr_lengths[opcode]) continue;

    // call handler when complete
    switch ((len = 0), opcode) {
      case 0x08: triangle<0x0>(instr); break;
      case 0x09: triangle<0x1>(instr); break;
      case 0x0a: triangle<0x2>(instr); break;
      case 0x0b: triangle<0x3>(instr); break;
      case 0x0c: triangle<0x4>(instr); break;
      case 0x0d: triangle<0x5>(instr); break;
      case 0x0e: triangle<0x6>(instr); break;
      case 0x0f: triangle<0x7>(instr); break;
      case 0x24: rectangle<0xa>(instr); break;
      case 0x25: rectangle<0xb>(instr); break;
      case 0x2a: set_key_gb(instr); break;
      case 0x2b: set_key_r(instr); break;
      case 0x2c: set_convert(instr); break;
      case 0x2d: set_scissor(instr); break;
      case 0x2e: set_prim_depth(instr); break;
      case 0x2f: set_other_modes(instr); break;
      case 0x30: load_tlut(instr); break;
      case 0x32: set_tile_size(instr); break;
      case 0x33: load_block(instr); break;
      case 0x34: load_tile(instr); break;
      case 0x35: set_tile(instr); break;
      case 0x36: rectangle<0x8>(instr); break;
      case 0x37: set_fill(instr); break;
      case 0x38: set_fog(instr); break;
      case 0x39: set_blend(instr); break;
      case 0x3a: set_prim(instr); break;
      case 0x3b: set_env(instr); break;
      case 0x3c: set_combine(instr); break;
      case 0x3d: set_texture(instr); break;
      case 0x3e: set_depth_image(instr); break;
      case 0x3f: set_color_image(instr); break;
      case 0x29: R4300::set_irqs(0x20); render(1); break;
      case 0x00: case 0x26: case 0x27: case 0x28: break;
      default: invalid(instr); break;
    }
  }
}

void RDP::init() {
  Vulkan::init();
}
