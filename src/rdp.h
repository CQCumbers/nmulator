#ifndef RDP_H
#define RDP_H

#include <vulkan/vulkan.h>
#include <vector>
#include "rdp.spv.array"

typedef struct PerTileData {
  uint32_t n_cmds;
  uint32_t cmd_idxs[31];
} tile_t;

typedef struct RDPCommand {
  uint32_t xyh[2], xym[2], xyl[2];
  uint32_t sh, sm, sl;
  uint32_t fill, blend, fog, lft, type;
  uint32_t shade[4], sde[4], sdx[4];
  uint32_t tile, bl_mux;
  uint32_t tex[2], tde[2], tdx[2];
} cmd_t;

typedef struct RDPTile {
  uint32_t format, size;
  uint32_t width, addr;
  uint32_t mask[2], shift[2];
} tex_t;

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;
  VkCommandBuffer commands = VK_NULL_HANDLE;

  /* === Descriptor Memory Access === */

  uint8_t *mapped_mem;
  const uint32_t group_size = 8, max_cmds = 64;
  uint32_t n_cmds = 0;

  const VkDeviceSize cmds_size = max_cmds * sizeof(cmd_t);
  cmd_t *cmds_ptr() { return reinterpret_cast<cmd_t*>(mapped_mem); }
  VkBuffer cmds = VK_NULL_HANDLE;

  const VkDeviceSize tiles_offset = cmds_size;
  const VkDeviceSize tiles_size = (320 / group_size) * (240 / group_size) * sizeof(tile_t);
  tile_t *tiles_ptr() { return reinterpret_cast<tile_t*>(mapped_mem + tiles_offset); }
  VkBuffer tiles = VK_NULL_HANDLE;

  const VkDeviceSize texes_offset = tiles_offset + tiles_size;
  const VkDeviceSize texes_size = 8 * sizeof(tex_t);
  tex_t *texes_ptr() { return reinterpret_cast<tex_t*>(mapped_mem + texes_offset); }
  VkBuffer texes = VK_NULL_HANDLE;

  const VkDeviceSize tmem_offset = texes_offset + texes_size;
  const VkDeviceSize tmem_size = 0x1000;
  uint8_t *tmem_ptr() { return mapped_mem + tmem_offset; }
  VkBuffer tmem = VK_NULL_HANDLE;

  const VkDeviceSize pixels_offset = tmem_offset + tmem_size;
  const VkDeviceSize pixels_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *pixels_ptr() { return mapped_mem + pixels_offset; }
  VkBuffer pixels = VK_NULL_HANDLE;
  const VkDeviceSize total_size = pixels_offset + pixels_size;

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
    const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size, .pCode = code
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &shader_info, 0, &shader);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 5, .pBindings = bindings
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
      .module = shader, .pName = "main"
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
    // create tmem buffer
    buffer_info.size = tmem_size;
    vkCreateBuffer(device, &buffer_info, 0, &tmem);
    vkBindBufferMemory(device, tmem, *memory, tmem_offset);
    // create pixels buffer
    buffer_info.size = pixels_size;
    vkCreateBuffer(device, &buffer_info, 0, &pixels);
    vkBindBufferMemory(device, pixels, *memory, pixels_offset);
  }

  void init_descriptors(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 5
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
      { .buffer = tmem, .offset = 0, .range = VK_WHOLE_SIZE },
      { .buffer = pixels, .offset = 0, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet write_descriptors[5];
    for (uint8_t i = 0; i < 5; ++i) {
      write_descriptors[i] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .pBufferInfo = &buffer_info[i]
      };
    }
    vkUpdateDescriptorSets(device, 5, write_descriptors, 0, 0);
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
    vkCmdDispatch(commands, 320 / group_size, 240 / group_size, 1);
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
    const uint32_t *code = reinterpret_cast<const uint32_t*>(rdp_spv);
    init_pipeline(code, rdp_spv_len, &desc_layout, &layout, &pipeline);
    init_buffers(gpu, &memory);
    init_descriptors(desc_layout, &descriptors);
    record_commands(layout, pipeline, descriptors);
    
    // map memory so buffers can filled by cpu
    vkMapMemory(device, memory, 0, total_size, 0, reinterpret_cast<void**>(&mapped_mem));
    memset(tiles_ptr(), 0, tiles_size);
  }

  /* === Runtime Methods === */

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
    uint32_t yh = (cmd.xyh[1] >> 2) / group_size;
    uint32_t yl = (cmd.xyl[1] >> 2) / group_size;
    for (uint32_t i = yh; i <= yl; ++i) {
      for (uint32_t j = 0; j < 320 / group_size; ++j) {
        tile_t *tile = tiles_ptr() + i * (320 / group_size) + j;
        if (tile->n_cmds >= 31) { printf("max cmds reached\n"); break; }
        tile->cmd_idxs[tile->n_cmds++] = n_cmds;
      }
    }
    cmds_ptr()[n_cmds++] = cmd;
  }

  uint32_t *render() {
    if (n_cmds == 0) return nullptr;
    run_commands(); n_cmds = 0; memset(tiles_ptr(), 0, tiles_size);
    return reinterpret_cast<uint32_t*>(Vulkan::pixels_ptr());
  }
}

namespace RSP {
  template <typename T>
  int64_t read(uint32_t addr);
}

namespace R4300 {
  template <typename T>
  int64_t read(uint32_t addr);
  void write_rdp(uint32_t *rdp_out);
}

namespace RDP {
  uint32_t fill = 0x0, blend = 0x0, fog = 0x0;
  uint32_t pc_start = 0x0, pc_end = 0x0;
  uint32_t pc = 0x0, status = 0x0, bl_mux = 0x0;
  uint32_t tex_size = 4, tex_width = 0, tex_addr = 0x0;

  uint64_t fetch(uint32_t addr) {
    return status & 0x1 ? RSP::read<uint64_t>(addr) : R4300::read<uint64_t>(addr);
  }

  void set_other_modes() {
    bl_mux = (fetch(pc) >> 16) & 0xffff; pc += 8;
  }

  void set_fill() {
    fill = fetch(pc) & 0xffffffff; pc += 8;
  }

  void set_fog() {
    fog = fetch(pc) & 0xffffffff; pc += 8;
  }

  void set_blend() {
    blend = fetch(pc) & 0xffffffff; pc += 8;
  }

  void set_texture() {
    // maybe send to gpu here, and do load_tile on gpu with native textures?
    // handling mid-frame tmem changes, formats, etc. might be easier
    uint64_t instr = fetch(pc); pc += 8;
    tex_size = (instr >> 51) & 0x3;
    tex_width = ((instr >> 32) & 0x3f) + 1;
    tex_addr = instr & 0x3ffffff;
  }

  void set_tile() {
    // render existing tile dependent commands
    R4300::write_rdp(Vulkan::render());
    // update tile
    uint64_t instr = fetch(pc); pc += 8;
    Vulkan::texes_ptr()[(instr >> 24) & 0x7] = {
      .format = uint32_t(instr >> 53) & 0x7, .size = uint32_t(instr >> 51) & 0x3,
      .width = uint32_t((instr >> 41) & 0xff) << 3, .addr = uint32_t((instr >> 32) & 0x1ff) << 3,
      .mask = { uint32_t(instr >> 4) & 0xf, uint32_t(instr >> 14) & 0xf },
      .shift = { uint32_t(instr) & 0xf, uint32_t(instr >> 10) & 0xf }
    };
  }

  void load_tile() {
    uint64_t instr = fetch(pc); pc += 8;
    uint32_t sh = (instr >> 12) & 0xfff, th = instr & 0xfff;
    uint32_t sl = (instr >> 44) & 0xfff, tl = (instr >> 32) & 0xfff;
    uint32_t tex_mem = Vulkan::texes_ptr()[(instr >> 24) & 0x7].addr;
    uint32_t pix_size = 1 << (tex_size - 1);
    // copy tile from dram to tmem
    uint32_t offset = (tl * tex_width + sl) * pix_size, width = (((sh - sl) >> 2) + 1) * pix_size;
    uint8_t *mem = Vulkan::tmem_ptr() + tex_mem; uint32_t ram = tex_addr + offset;
    uint32_t width_pad = (width + 0x7) & ~0x7;
    for (uint32_t i = 0; i <= (th - tl) >> 2; ++i) {
      for (uint32_t j = 0; j < width; ++j) {
        //uint32_t k = (((j & ~0x3) >> 1) | (j & 0x1)) + ((j & 0x2) << 10);
        mem[j] = R4300::read<uint8_t>(ram + j);
      }
      mem += width_pad, ram += tex_width * pix_size;
    }
  }

  void fill_triangle() {
    uint64_t instr[4] = {
      fetch(pc), fetch(pc + 8), fetch(pc + 16), fetch(pc + 24)
    };
    printf("filling triangle with xh = %x, yh = %x\n", uint32_t(instr[2] >> 32), uint32_t(instr[0] & 0x3fff));
    pc += 32;
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t(instr[2] >> 32), uint32_t(instr[0] & 0x3fff) },
      .xym = { uint32_t(instr[3] >> 32), uint32_t((instr[0] >> 16) & 0x3fff) },
      .xyl = { uint32_t(instr[1] >> 32), uint32_t((instr[0] >> 32) & 0x3fff) },
      .sh = uint32_t(instr[2]), .sm = uint32_t(instr[3]), .sl = uint32_t(instr[1]),
      .lft = uint32_t((instr[0] >> 55) & 0x1), .type = 0,
      .fill = fill, .fog = fog, .blend = blend, .bl_mux = bl_mux
    });
  }

  void shade_triangle() {
    uint64_t instr[12] = {
      fetch(pc), fetch(pc + 8), fetch(pc + 16), fetch(pc + 24),
      fetch(pc + 32), fetch(pc + 40), fetch(pc + 48), fetch(pc + 56),
      fetch(pc + 64), fetch(pc + 72), fetch(pc + 80), fetch(pc + 88)
    };
    pc += 96;
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t(instr[2] >> 32), uint32_t(instr[0] & 0x3fff) },
      .xym = { uint32_t(instr[3] >> 32), uint32_t((instr[0] >> 16) & 0x3fff) },
      .xyl = { uint32_t(instr[1] >> 32), uint32_t((instr[0] >> 32) & 0x3fff) },
      .sh = uint32_t(instr[2]), .sm = uint32_t(instr[3]), .sl = uint32_t(instr[1]),
      .lft = uint32_t((instr[0] >> 55) & 0x1), .type = 2,
      .fill = fill, .fog = fog, .blend = blend, .bl_mux = bl_mux,
      .shade = {
        uint32_t(instr[4] >> 48) << 16 | uint32_t(instr[6] >> 48),
        uint32_t((instr[4] >> 32) & 0xffff) << 16 | uint32_t((instr[6] >> 32) & 0xffff),
        uint32_t((instr[4] >> 16) & 0xffff) << 16 | uint32_t((instr[6] >> 16) & 0xffff),
        uint32_t(instr[4] & 0xffff) << 16 | uint32_t(instr[6] & 0xffff),
      },
      .sde = {
        uint32_t(instr[8] >> 48) << 16 | uint32_t(instr[10] >> 48),
        uint32_t((instr[8] >> 32) & 0xffff) << 16 | uint32_t((instr[10] >> 32) & 0xffff),
        uint32_t((instr[8] >> 16) & 0xffff) << 16 | uint32_t((instr[10] >> 16) & 0xffff),
        uint32_t(instr[8] & 0xffff) << 16 | uint32_t(instr[10] & 0xffff),
      },
      .sdx = {
        uint32_t(instr[5] >> 48) << 16 | uint32_t(instr[7] >> 48),
        uint32_t((instr[5] >> 32) & 0xffff) << 16 | uint32_t((instr[7] >> 32) & 0xffff),
        uint32_t((instr[5] >> 16) & 0xffff) << 16 | uint32_t((instr[7] >> 16) & 0xffff),
        uint32_t(instr[5] & 0xffff) << 16 | uint32_t(instr[7] & 0xffff),
      }
    });
  }

  void fill_rectangle() {
    uint64_t instr = fetch(pc); pc += 8;
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t((instr >> 12) & 0xfff), uint32_t(instr & 0xfff) },
      .xyl = { uint32_t((instr >> 44) & 0xfff), uint32_t((instr >> 32) & 0x3ff) },
      .fill = fill, .fog = fog, .blend = blend, .bl_mux = bl_mux, .type = 1
    });
  }

  void tex_rectangle() {
    uint64_t instr[2] = { fetch(pc), fetch(pc + 8) }; pc += 16;
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t((instr[0] >> 12) & 0xfff), uint32_t(instr[0] & 0xfff) },
      .xyl = { uint32_t((instr[0] >> 44) & 0xfff), uint32_t((instr[0] >> 32) & 0x3ff) },
      .fill = fill, .fog = fog, .blend = blend, .bl_mux = bl_mux,
      .type = 3, .tile = uint32_t(instr[0] >> 24) & 0x7,
      .tex = { uint32_t((instr[1] >> 48) & 0xffff), uint32_t((instr[1] >> 32) & 0xffff) },
      .tde = { uint32_t(instr[1] & 0xffff), uint32_t(instr[1] & 0xffff) },
      .tdx = { uint32_t((instr[1] >> 16) & 0xffff), uint32_t((instr[1] >> 16) & 0xffff) },
    });
  }
  
  void tex_rectangle_flip() {
    uint64_t instr[2] = { fetch(pc), fetch(pc + 8) }; pc += 16;
    Vulkan::add_rdp_cmd({
      .xyh = { uint32_t((instr[0] >> 12) & 0xfff), uint32_t(instr[0] & 0xfff) },
      .xyl = { uint32_t((instr[0] >> 44) & 0xfff), uint32_t((instr[0] >> 32) & 0x3ff) },
      .fill = fill, .fog = fog, .blend = blend, .bl_mux = bl_mux,
      .type = 4, .tile = uint32_t(instr[0] >> 24) & 0x7,
      .tex = { uint32_t((instr[1] >> 32) & 0xffff), uint32_t((instr[1] >> 48) & 0xffff) },
      .tde = { uint32_t(instr[1] & 0xffff), uint32_t(instr[1] & 0xffff) },
      .tdx = { uint32_t((instr[1] >> 16) & 0xffff), uint32_t((instr[1] >> 16) & 0xffff) },
    });
  }

  uint32_t *update(uint32_t cycles) {
    // interpret config instructions 
    pc = pc_start;
    while (pc != pc_end) {
      uint64_t instr = fetch(pc);
      switch (instr >> 56) {
        case 0x08: fill_triangle(); break;
        case 0x0c: shade_triangle(); break;
        case 0x24: tex_rectangle(); break;
        case 0x25: tex_rectangle_flip(); break;
        case 0x2f: set_other_modes(); break;
        case 0x34: load_tile(); break;
        case 0x35: set_tile(); break;
        case 0x36: fill_rectangle(); break;
        case 0x37: set_fill(); break;
        case 0x38: set_fog(); break;
        case 0x39: set_blend(); break;
        case 0x3d: set_texture(); break;
        default: printf("RDP instruction %llx\n", instr); pc += 8; break;
      }
    }
    // rasterize on GPU according to config
    return Vulkan::render();
  }
}

#endif
