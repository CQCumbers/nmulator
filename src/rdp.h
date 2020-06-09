#ifndef RDP_H
#define RDP_H

#include <vulkan/vulkan.h>
#include <array>
#include <vector>
#include <string.h>
#include <stdio.h>
#include "scheduler.h"
#include "comp.spv"
#include "vert.spv"
#include "frag.spv"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <sys/mman.h>

namespace R4300 {
  extern bool logging_on;
  extern uint8_t *pages[0x100];
  void set_irqs(uint32_t mask);
  template <typename T, bool map>
  int64_t read(uint32_t addr);
}

static uint8_t *alloc_pages(uint32_t size) {
  return reinterpret_cast<uint8_t*>(mmap(
    nullptr, size, PROT_READ | PROT_WRITE,
    MAP_ANONYMOUS | MAP_SHARED, 0, 0
  ));
}

/* === Shader Structs === */

// all coeffs are 16.16 fixed point
// except yh/yl/ym, which is 12.2

#pragma pack(push, 1)
struct TileData {
  uint32_t cmd_idxs[64];
};

struct GlobalData {
  uint32_t width, size;
  uint32_t n_cmds, pad;
  uint32_t zenc[128][2];
  uint32_t zdec[8][2];
};

struct RDPState {
  int32_t sxh, sxl, syh, syl;
  uint32_t modes[2], mux[2];
  uint32_t tlut, tmem, texes;
  uint32_t fill, fog, blend;
  uint32_t env, prim, zprim;
  uint32_t keys, keyc, pad2;
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

/* === Vulkan Setup === */

namespace RDP {
  void render();
}

namespace Vulkan {
  VkDevice device = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkSemaphore available, finished;

  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_idx = 0;
  std::vector<VkCommandBuffer> graphics_cmds;
  VkCommandBuffer compute_cmds = VK_NULL_HANDLE;

  constexpr VkDeviceSize align(VkDeviceSize offset) {
    return (offset + 0x1f) & ~0x1f;
  }

  /* === Descriptor Memory Access === */

  uint8_t *mapped_mem = nullptr;
  const uint32_t group_size = 8, max_cmds = 2048, max_copies = 4096;
  uint32_t gwidth = 320 / group_size, gheight = 240 / group_size;
  uint32_t n_cmds = 0, n_tmems = 0;
  bool dump_next = false;

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
  const VkDeviceSize pixels_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *pixels_ptr() { return mapped_mem + pixels_offset; }

  const VkDeviceSize zbuf_offset = align(pixels_offset + pixels_size);
  const VkDeviceSize zbuf_size = 320 * 240 * sizeof(uint16_t);
  uint8_t *zbuf_ptr() { return mapped_mem + zbuf_offset; }

  const VkDeviceSize rdram_offset = align(zbuf_offset + zbuf_size);
  const VkDeviceSize rdram_size = 320 * 240 * sizeof(uint32_t);
  uint8_t *rdram_ptr() { return mapped_mem + rdram_offset; }

  const VkDeviceSize total_size = rdram_offset + rdram_size;
  VkBuffer buffer = VK_NULL_HANDLE;

  /* === Vulkan Initialization == */

  void init_instance(SDL_Window *window, VkInstance *instance) {
    // get required extensions from SDL
    uint32_t n_exts = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &n_exts, 0);
    std::vector<const char*> exts(n_exts);
    SDL_Vulkan_GetInstanceExtensions(window, &n_exts, exts.data());
    exts.push_back("VK_KHR_surface"), ++n_exts;
    const char *layers[] = { "VK_LAYER_KHRONOS_validation" };
    // create vulkan instance
    const VkApplicationInfo app_info = {
      .pApplicationName = "nmulator RDP",
      .apiVersion = VK_API_VERSION_1_0
    };
    const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = 1, .ppEnabledLayerNames = layers,
      .enabledExtensionCount = n_exts,
      .ppEnabledExtensionNames = exts.data()
    };
    vkCreateInstance(&instance_info, 0, instance);
  }

  void init_device(const VkInstance &instance, VkPhysicalDevice *gpu) {
    // check all vulkan physical devices
    uint32_t n_gpus = 0;
    vkEnumeratePhysicalDevices(instance, &n_gpus, 0);
    std::vector<VkPhysicalDevice> gpus(n_gpus);
    vkEnumeratePhysicalDevices(instance, &n_gpus, gpus.data());
    // TODO: handle seperate compute and graphics queue families
    for (VkPhysicalDevice gpu_ : gpus) {
      // check queue families on each device
      uint32_t n_queues = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, 0);
      std::vector<VkQueueFamilyProperties> queues(n_queues);
      vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &n_queues, queues.data());
      // if queue family supports compute, init virtual device
      for (uint32_t i = 0; i < n_queues; ++i) {
        if (~queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) continue;
        const float priority = 1.0;
        const VkDeviceQueueCreateInfo queue_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = (queue_idx = i),
          .queueCount = 1, .pQueuePriorities = &priority
        };
        const char *exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        const VkDeviceCreateInfo device_info = {
          .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
          .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts
        };
        vkCreateDevice((*gpu = gpu_), &device_info, 0, &device);
        vkGetDeviceQueue(device, queue_idx, 0, &queue);
        return;
      }
    }
    printf("No compatible GPU found\n"), exit(1);
  }

  void init_swapchain(SDL_Window *window, const VkInstance &instance,
      const VkPhysicalDevice &gpu, VkSurfaceFormatKHR *fmt, VkExtent2D *extent) {
    VkSurfaceKHR surface;
    VkSurfaceCapabilitiesKHR capable;
    SDL_Vulkan_CreateSurface(window, instance, &surface);
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &capable);
    // setup swapchain surface format
    uint32_t n_formats = 1;;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &n_formats, fmt);
    if (fmt->format == VK_FORMAT_UNDEFINED) fmt->format = VK_FORMAT_B8G8R8A8_UNORM;
    // setup present mode, window extents
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t images = 3; int32_t width, height;
    if (images - 1 < capable.minImageCount - 1) images = capable.minImageCount;
    if (images - 1 > capable.maxImageCount - 1) images = capable.maxImageCount;
    SDL_Vulkan_GetDrawableSize(window, &width, &height);
    extent->width = width, extent->height = height;
    // create complete swapchain
    const VkSwapchainCreateInfoKHR swapchain_info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface, .minImageCount = images,
      .imageFormat = fmt->format, .imageColorSpace = fmt->colorSpace,
      .imageExtent = *extent, .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = capable.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = mode, .clipped = VK_TRUE
    };
    vkCreateSwapchainKHR(device, &swapchain_info, 0, &swapchain);
  }

  void init_semaphores() {
    const VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };
    vkCreateSemaphore(device, &semaphore_info, nullptr, &available);
    vkCreateSemaphore(device, &semaphore_info, nullptr, &finished);
  }

  void init_renderpass(const VkSurfaceFormatKHR &fmt, VkRenderPass *pass) {
    const VkAttachmentDescription attach = {
      .format = fmt.format, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    const VkAttachmentReference attach_ref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &attach_ref
    };
    const VkRenderPassCreateInfo pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &attach,
      .subpassCount = 1, .pSubpasses = &subpass,
    };
    vkCreateRenderPass(device, &pass_info, 0, pass);
  }

  void init_graphics(const VkExtent2D &extent, const VkRenderPass &pass,
      VkDescriptorSetLayout *desc_layout, VkPipelineLayout *layout,
      VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo vert_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_code), .pCode = vert_code,
    };
    const VkShaderModuleCreateInfo frag_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(frag_code), .pCode = frag_code,
    };
    VkShaderModule vert, frag;
    vkCreateShaderModule(device, &vert_info, 0, &vert);
    vkCreateShaderModule(device, &frag_info, 0, &frag);
    // describe needed descriptor set layout for shader
    const VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
    };
    const VkDescriptorSetLayoutCreateInfo desc_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding
    };
    vkCreateDescriptorSetLayout(device, &desc_layout_info, 0, desc_layout);
    // create graphics pipeline with shaders
    const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = desc_layout
    };
    vkCreatePipelineLayout(device, &layout_info, 0, layout);
    const VkPipelineShaderStageCreateInfo stage_info[] = {
      {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"
      }, {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"
      }
    };
    // create empty vertex input, assembly info
    const VkPipelineVertexInputStateCreateInfo input_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const VkPipelineInputAssemblyStateCreateInfo assembly_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 
      .primitiveRestartEnable = VK_FALSE
    };
    // setup viewport and scissor
    const VkViewport viewport = {
      .width = (float)extent.width, .height = (float)extent.height,
      .minDepth = 0.0f, .maxDepth = 1.0f
    };
    const VkRect2D scissor = { .offset = {0, 0}, .extent = extent };
    const VkPipelineViewportStateCreateInfo viewport_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor
    };
    // setup fixed function operations
    const VkPipelineRasterizationStateCreateInfo rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE, .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f,
      .cullMode = VK_CULL_MODE_BACK_BIT, .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE
    };
    const VkPipelineMultisampleStateCreateInfo multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .sampleShadingEnable = VK_FALSE, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };
    const VkPipelineColorBlendAttachmentState attach = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      .blendEnable = VK_FALSE
    };
    const VkPipelineColorBlendStateCreateInfo blender = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE, .attachmentCount = 1, .pAttachments = &attach
    };
    // create graphics pipeline
    const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stage_info,
      .pVertexInputState = &input_info, .pInputAssemblyState = &assembly_info,
      .pViewportState = &viewport_info, .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling, .pColorBlendState = &blender,
      .layout = *layout, .renderPass = pass
    };
    vkCreateGraphicsPipelines(device, 0, 1, &pipeline_info, 0, pipeline);
    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
  }

  void init_framebufs(const VkSurfaceFormatKHR &fmt, const VkExtent2D &extent,
      const VkRenderPass &pass, std::vector<VkFramebuffer> &framebufs) {
    // retrieve swapchain images
    uint32_t n_images = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &n_images, nullptr);
    std::vector<VkImage> images(n_images); 
    vkGetSwapchainImagesKHR(device, swapchain, &n_images, images.data());
    framebufs.resize(n_images);
    // create view and framebuffer for each image
    for (uint32_t i = 0; i < n_images; ++i) {
      VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt.format, .subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0, .levelCount = 1,
          .baseArrayLayer = 0, .layerCount = 1
        }
      };
      VkImageView view;
      vkCreateImageView(device, &view_info, 0, &view);
      const VkFramebufferCreateInfo framebuf_info = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = pass, .attachmentCount = 1,
        .pAttachments = &view, .width = extent.width,
        .height = extent.height, .layers = 1,
      };
      vkCreateFramebuffer(device, &framebuf_info, 0, &framebufs[i]);
    }
  }

  void init_graphics_desc(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo descriptors_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &descriptors_info, descriptors);
    // bind buffer to descriptor set
    const VkDescriptorBufferInfo buffer_info = {
      .buffer = buffer, .offset = rdram_offset, .range = rdram_size
    };
    const VkWriteDescriptorSet write_descriptor = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = *descriptors, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info
    };
    vkUpdateDescriptorSets(device, 1, &write_descriptor, 0, 0);
  }

  void record_graphics(const std::vector<VkFramebuffer> &framebufs,
      const VkExtent2D &extent, const VkRenderPass &pass,
      const VkPipelineLayout &layout, const VkPipeline &pipeline,
      const VkDescriptorSet &descriptors) {
    // allocate command buffers from command pool
    const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_idx
    };
    VkCommandPool pool;
    vkCreateCommandPool(device, &pool_info, 0, &pool);
    const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = (uint32_t)framebufs.size()
    };
    graphics_cmds.resize(framebufs.size());
    vkAllocateCommandBuffers(device, &cmd_info, graphics_cmds.data());
    for (uint32_t i = 0; i < framebufs.size(); ++i) {
      // record draw commands for each framebuf
      VkCommandBuffer cmds = graphics_cmds[i];
      const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      };
      vkBeginCommandBuffer(cmds, &begin_info);
      const VkClearValue clear = { 0.0f, 0.1f, 0.2f, 1.0f };
      const VkRenderPassBeginInfo pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = pass, .framebuffer = framebufs[i],
        .clearValueCount = 1, .pClearValues = &clear,
        .renderArea.offset = { .x = 0,.y = 0 },
        .renderArea.extent = extent
      };
      vkCmdBeginRenderPass(cmds, &pass_info, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmds, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      vkCmdBindDescriptorSets(cmds, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptors, 0, 0);
      vkCmdDraw(cmds, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmds), vkEndCommandBuffer(cmds);
    }
  }

  void init_compute(VkDescriptorSetLayout *desc_layout,
      VkPipelineLayout *layout, VkPipeline *pipeline) {
    // load shader code into module
    const VkShaderModuleCreateInfo comp_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_code), .pCode = comp_code,
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
      .module = comp, .pName = "main"
    };
    const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info, .layout = *layout
    };
    vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, pipeline);
    vkDestroyShaderModule(device, comp, nullptr);
  }

  void init_buffer(const VkPhysicalDevice &gpu, VkDeviceMemory *memory) {
    // create shared buffer to hold descriptors
    const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = total_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1, .pQueueFamilyIndices = &queue_idx
    };
    vkCreateBuffer(device, &buffer_info, 0, &buffer);
    // get memory requirements for buffer
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
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
      vkAllocateMemory(device, &allocate_info, 0, memory);
    }
    vkBindBufferMemory(device, buffer, *memory, 0);
  }

  void init_compute_desc(const VkDescriptorSetLayout &layout, VkDescriptorSet *descriptors) {
    // allocate descriptor set from descriptor pool
    const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 7
    };
    const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size
    };
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &pool_info, 0, &pool);
    const VkDescriptorSetAllocateInfo descriptors_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout
    };
    vkAllocateDescriptorSets(device, &descriptors_info, descriptors);
    // bind buffers to descriptor set
    const VkDescriptorBufferInfo buffer_info[] = {
      { .buffer = buffer, .offset = cmds_offset, .range = cmds_size },
      { .buffer = buffer, .offset = tiles_offset, .range = tiles_size },
      { .buffer = buffer, .offset = texes_offset, .range = texes_size },
      { .buffer = buffer, .offset = globals_offset, .range = globals_size },
      { .buffer = buffer, .offset = tmem_offset, .range = tmem_size },
      { .buffer = buffer, .offset = pixels_offset, .range = pixels_size },
      { .buffer = buffer, .offset = zbuf_offset, .range = zbuf_size }
    };
    VkWriteDescriptorSet write_descriptors[7];
    for (uint8_t i = 0; i < 7; ++i) {
      write_descriptors[i] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptors, .dstBinding = i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info[i]
      };
    }
    vkUpdateDescriptorSets(device, 7, write_descriptors, 0, 0);
  }
  
  void record_compute(const VkPipelineLayout &layout, const VkPipeline &pipeline,
      const VkDescriptorSet &descriptors) {
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
      .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(device, &cmd_info, &compute_cmds);
    // record commands - bind descriptor set to set layout, start pipeline
    const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    vkBeginCommandBuffer(compute_cmds, &begin_info);
    vkCmdBindPipeline(compute_cmds, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(compute_cmds, VK_PIPELINE_BIND_POINT_COMPUTE,
      layout, 0, 1, &descriptors, 0, 0);
    vkCmdDispatch(compute_cmds, gwidth, gheight, 1);
    vkEndCommandBuffer(compute_cmds);
  }
  

  void init(SDL_Window *window) {
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkSurfaceFormatKHR fmt;
    VkExtent2D extent;
    VkDeviceMemory memory;
    
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorSet descriptors;
    VkPipeline pipeline;
    VkRenderPass pass;
    std::vector<VkFramebuffer> framebufs;

    // configure GPU, display, and memory
    init_instance(window, &instance);
    init_device(instance, &gpu);
    init_swapchain(window, instance, gpu, &fmt, &extent);
    init_semaphores();
    init_buffer(gpu, &memory);

    // setup graphics pipeline work
    init_renderpass(fmt, &pass);
    init_framebufs(fmt, extent, pass, framebufs);
    init_graphics(extent, pass, &desc_layout, &layout, &pipeline);
    init_compute_desc(desc_layout, &descriptors);
    record_graphics(framebufs, extent, pass, layout, pipeline, descriptors);

    // setup compute pipeline work
    init_compute(&desc_layout, &layout, &pipeline);
    init_compute_desc(desc_layout, &descriptors);
    record_compute(layout, pipeline, descriptors);
    
    // map memory so buffers can filled by cpu
    auto ptr = reinterpret_cast<void**>(&mapped_mem);
    vkMapMemory(device, memory, 0, total_size, 0, ptr);
    memset(tiles_ptr(), 0, tiles_size);

    // setup z encoding LUT
    uint32_t (*zenc)[2] = Vulkan::globals_ptr()->zenc;
    for (uint8_t i = 0; i < 128; ++i) {
      uint32_t &shr = zenc[i][0];
      uint32_t &exp = zenc[i][1];
      if (i < 0x40) shr = 10, exp = 0;
      else if (i < 0x60) shr = 9, exp = 1;
      else if (i < 0x70) shr = 8, exp = 2;
      else if (i < 0x78) shr = 7, exp = 3;
      else if (i < 0x7c) shr = 6, exp = 4;
      else if (i < 0x7e) shr = 5, exp = 5;
      else if (i < 0x7f) shr = 4, exp = 6;
      else shr = 4, exp = 7;
    }
    // setup z decoding LUT
    uint32_t (*zdec)[2] = Vulkan::globals_ptr()->zdec;
    zdec[0][0] = 6, zdec[0][1] = 0x00000;
    zdec[1][0] = 5, zdec[1][1] = 0x20000;
    zdec[2][0] = 4, zdec[2][1] = 0x30000;
    zdec[3][0] = 3, zdec[3][1] = 0x38000;
    zdec[4][0] = 2, zdec[4][1] = 0x3c000;
    zdec[5][0] = 1, zdec[5][1] = 0x3e000;
    zdec[6][0] = 0, zdec[6][1] = 0x3f000;
    zdec[7][0] = 0, zdec[7][1] = 0x3f800;
  }

  /* === Runtime Methods === */

  void run_graphics() {
    uint32_t idx = 0;
    vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, available, 0, &idx);
    VkPipelineStageFlags mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1, .pWaitSemaphores = &available, 
      .pWaitDstStageMask = &mask,
      .commandBufferCount = 1, .pCommandBuffers = &graphics_cmds[idx],
      .signalSemaphoreCount = 1, .pSignalSemaphores = &finished
    };
    vkQueueSubmit(queue, 1, &submit_info, 0);
    const VkPresentInfoKHR present_info = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1, .pWaitSemaphores = &finished,
      .swapchainCount = 1, .pSwapchains = &swapchain,
      .pImageIndices = &idx, .pResults = 0
    };
    vkQueuePresentKHR(queue, &present_info);
    vkQueueWaitIdle(queue);
  }

  void run_compute() {
    const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &compute_cmds
    };
    vkQueueSubmit(queue, 1, &submit_info, 0);
    vkQueueWaitIdle(queue);
  }

  void add_tmem_copy(RDPState &state) {
    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    if (++n_tmems >= max_copies) {
      printf("[RDP] max_copies reached\n");
      RDP::render(), n_tmems = 0;
    }
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);
    state.tmem = n_tmems;
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
    if (n_cmds >= max_cmds) {
      printf("[RDP] max_cmds exceeded\n");
      RDP::render();
    }
  }

  void dump_buffer() {
    FILE *file = fopen("dump.bin", "w");
    if (!file) printf("error: can't open file\n"), exit(1);
    fwrite(mapped_mem, 1, total_size, file);
    fclose(file), dump_next = false;
  }

  void run_buffer() {
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_SetHint(SDL_HINT_RENDER_VSYNC, "1");
    SDL_CreateWindowAndRenderer(640, 480, SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    init(window);

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA5551,
        SDL_TEXTUREACCESS_STREAMING, 320, 240);

    FILE *file = fopen("dump.bin", "r");
    if (!file) printf("error: can't open file\n"), exit(1);
    fread(mapped_mem, 1, total_size, file);
    fclose(file);

    run_compute();
    uint16_t *pixels = (uint16_t*)pixels_ptr();
    uint16_t *out = (uint16_t*)alloc_pages(320 * 240 * 2);
    for (uint32_t i = 0; i < 320 * 240; ++i)
      out[i] = bswap16(pixels[i]);
    SDL_UpdateTexture(texture, nullptr, out, 320 * 2);

    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
    while (true) {
      for (SDL_Event e; SDL_PollEvent(&e);) {
        if (e.type == SDL_QUIT) exit(0);
      }
    }
  }

  void render(uint8_t *img, uint32_t img_len, uint8_t *zbuf, uint32_t zbuf_len) {
    if (!img || n_cmds == 0) return;

    uint8_t *last_tmem = tmem_ptr();
    RDPTex *last_texes = texes_ptr();
    n_tmems = 0;
    memcpy(tmem_ptr(), last_tmem, 0x1000);
    memcpy(texes_ptr(), last_texes, sizeof(RDPTex) * 8);

    globals_ptr()->n_cmds = n_cmds;
    if (zbuf) memcpy(zbuf_ptr(), zbuf, zbuf_len);
    else memset(zbuf_ptr(), 0xff, zbuf_len);
    memcpy(pixels_ptr(), img, img_len);

    /*if (dump_next && img[1] != 0xff) {
      memset(pixels_ptr(), 0, pixels_size);
      printf("Resetting pixels\n");
      dump_buffer(), dump_next = true;
    }*/

    run_compute();
    n_cmds = 0, memset(tiles_ptr(), 0, tiles_size);
    if (zbuf) memcpy(zbuf, zbuf_ptr(), zbuf_len);
    memcpy(img, pixels_ptr(), img_len);
    
    /*if (dump_next && img[1] != 0x0 && img[1] != 0xff) {
      dump_next = false;
    }*/
  }
}

/* === RDP Interface === */

namespace RSP {
  extern uint64_t reg_array[0x100];
  extern const uint8_t dev_cop0;
  template <typename T, bool all>
  int64_t read(uint32_t addr);
  extern bool step;
}

namespace RDP {
  uint64_t *rsp_cop0 = RSP::reg_array + RSP::dev_cop0;
  uint64_t &pc_start = rsp_cop0[8], &pc_end = rsp_cop0[9];
  uint64_t &pc = rsp_cop0[10], &status = rsp_cop0[11];
  uint64_t offset;

  uint32_t img_size, img_width, height;
  uint8_t *img_addr, *zbuf_addr;
  uint32_t tex_nibs, tex_width, tex_addr;
  RDPState state;

  /* === Helper Functions === */

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

  template <uint8_t len>
  std::array<uint32_t, len> fetch(uint64_t &addr) {
    std::array<uint32_t, len> out;
    for (uint8_t i = 0; i < len; ++i, addr += 4) {
      if (status & 0x1) out[i] = RSP::read<uint32_t>(addr);
      else out[i] = R4300::read<uint32_t, false>(addr);
    }
    return out;
  }

  void render() {
    printf("RDP render to %x, %x\n", img_addr - R4300::pages[0], zbuf_addr - R4300::pages[0]);
    uint32_t img_len = img_width * height * img_size;
    uint32_t zbuf_len = img_width * height * 2;
    Vulkan::render(img_addr, img_len, zbuf_addr, zbuf_len);
  }

  /* === Instruction Translations === */

  void set_image() {
    render();
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    img_size = 1 << (((instr[0] >> 19) & 0x3) - 1);
    img_width = (instr[0] & 0x3ff) + 1;
    img_addr = R4300::pages[0] + (instr[1] & 0x3ffffff);
    printf("IMG_ADDR is now %x\n", img_addr);

    GlobalData *globals = Vulkan::globals_ptr();
    globals->width = img_width, globals->size = img_size;
    Vulkan::gwidth = img_width / Vulkan::group_size;
  }

  void set_scissor() {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    state.sxh = zext(instr[0] >> 12, 12) << 14, state.syh = zext(instr[0], 12);
    state.sxl = zext(instr[1] >> 12, 12) << 14, state.syl = sext(instr[1], 12);
    height = (state.syl >> 2) - (state.syh >> 2);
    Vulkan::gheight = height / Vulkan::group_size;
  }

  void set_other_modes() {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    //printf("Modes M0: %x\n", instr[0]);
    memcpy(state.modes, instr.data(), 8);
  }

  void set_fill() {
    state.fill = bswap32(fetch<2>(pc)[1]);
  }

  void set_fog() {
    state.fog = bswap32(fetch<2>(pc)[1]);
  }

  void set_blend() {
    state.blend = bswap32(fetch<2>(pc)[1]);
  }

  void set_combine() {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    memcpy(state.mux, instr.data(), 8);
  }

  void set_env() {
    state.env = bswap32(fetch<2>(pc)[1]);
  }

  void set_prim() {
    state.prim = bswap32(fetch<2>(pc)[1]);
  }

  void set_zprim() {
    state.zprim = fetch<2>(pc)[1];
  }

  void set_zbuf() {
    zbuf_addr = R4300::pages[0] + (fetch<2>(pc)[1] & 0x3ffffff);
  }

  void set_key_r() {
    uint32_t instr = fetch<2>(pc)[1];
    state.keys &= 0xffff00, state.keyc &= 0xffff00;
    state.keys |= (instr >> 0) & 0xff, state.keyc |= (instr >> 8) & 0xff;
  }

  void set_key_gb() {
    uint32_t instr = fetch<2>(pc)[1];
    state.keys &= 0xff, state.keyc &= 0xff;
    state.keys |= (instr >> 8) & 0xff00, state.keys |= (instr << 8) & 0xff0000;
    state.keyc |= (instr >> 16) & 0xff00, state.keyc |= (instr << 4) & 0xff0000;
  }

  void set_texture() {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    tex_nibs = 0x1 << ((instr[0] >> 19) & 0x3);
    tex_width = (instr[0] & 0x3ff) + 1;
    tex_addr = instr[1] & 0x3ffffff;
  }

  void set_tile() {
    Vulkan::add_tmem_copy(state);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    uint8_t tex_idx = (instr[1] >> 24) & 0x7;
    RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
    tex.format = (instr[0] >> 21) & 0x7, tex.size = (instr[0] >> 19) & 0x3;
    tex.width = ((instr[0] >> 9) & 0xff) << 3;
    tex.addr = (instr[0] & 0x1ff) << 3, tex.pal = (instr[1] >> 20) & 0xf;
    tex.shift[0] = instr[1] & 0x3ff, tex.shift[1] = (instr[1] >> 10) & 0x3ff;
    printf("set_tile with format = %x, size = %x, shift=%x\n", tex.format, tex.size, tex.shift[0]);
  }

  void set_tile_size() {
    Vulkan::add_tmem_copy(state);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    uint8_t tex_idx = (instr[1] >> 24) & 0x7;
    RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
    tex.sth[0] = (instr[1] >> 12) & 0xfff, tex.sth[1] = instr[1] & 0xfff;
    tex.stl[0] = (instr[0] >> 12) & 0xfff, tex.stl[1] = instr[0] & 0xfff;
  }

  uint32_t taddr(uint32_t addr, uint32_t tex_nibs) {
    // handle rgba32 split into hi/lo tmem
    if (tex_nibs != 8) return (addr / 2) & 0x7ff;
    uint32_t offs = (addr / 4) & 0x3ff;
    return ((addr & 0x2) << 9) | offs;
  }

  void load_tile() {
    // Set RDP tile size
    Vulkan::add_tmem_copy(state);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    uint8_t tex_idx = (instr[1] >> 24) & 0x7;
    RDPTex &tex = Vulkan::texes_ptr()[tex_idx];
    tex.sth[0] = (instr[1] >> 12) & 0xfff, tex.sth[1] = instr[1] & 0xfff;
    tex.stl[0] = (instr[0] >> 12) & 0xfff, tex.stl[1] = instr[0] & 0xfff;
    // Copy from texture image to tmem
    uint32_t offset = tex.stl[1] / 4 * tex_width + tex.stl[0] / 4;
    uint32_t len = tex.sth[0] / 4 - tex.stl[0] / 4 + 1, flip = 0;
    offset = offset * tex_nibs / 2, len = len * tex_nibs / 2;
    uint8_t *ram = R4300::pages[0] + tex_addr + offset;
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;
    const uint32_t tn = tex_nibs;
    // Swap every other 16-bit word on odd rows
    for (int32_t i = 0; i <= (tex.sth[1] - tex.stl[1]) / 4; ++i) {
      for (uint32_t j = 0; j < len; j += 2)
        ((uint16_t*)mem)[taddr(j, tn) ^ flip] = ((uint16_t*)ram)[j / 2];
      mem += tex.width, flip ^= 0x2;
      ram += tex_width * tex_nibs / 2;
    }
  }

  void load_block() {
    // Set copy parameters
    Vulkan::add_tmem_copy(state);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    uint32_t sh = (instr[1] >> 12) & 0xfff, dxt = instr[1] & 0xfff;
    uint32_t sl = (instr[0] >> 12) & 0xfff, tl = instr[0] & 0xfff;
    RDPTex tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    // Copy from texture image to tmem
    uint32_t offset = (tl * tex_width + sl) * tex_nibs / 2;
    uint32_t len = (sh - sl + 1) * tex_nibs / 2, flip = 0;
    uint16_t *ram = (uint16_t*)(R4300::pages[0] + tex_addr + offset);
    uint8_t *mem = Vulkan::tmem_ptr() + tex.addr;
    const uint32_t tn = tex_nibs;
    // Swap every other 16-bit word on odd rows
    for (uint32_t i = 0; i < len;) {
      for (uint32_t t = 0; t < 0x800 && i < len; i += 2) {
        ((uint16_t*)mem)[taddr(i, tn) ^ flip] = *(ram++);
        if ((i & 0x7) == 0x6) t += dxt;
      }
      mem += tex.width, flip ^= 0x2;
    }
  }

  void load_tlut() {
    // Set copy parameters
    Vulkan::add_tmem_copy(state);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    uint32_t sh = (instr[1] >> 14) & 0xff, sl = (instr[0] >> 14) & 0xff;
    RDPTex tex = Vulkan::texes_ptr()[(instr[1] >> 24) & 0x7];
    // Copy from texture image to tmem
    uint8_t *mem = Vulkan::tmem_ptr() + (state.tlut = tex.addr);
    uint32_t ram = tex_addr + sl * tex_nibs / 2;
    uint32_t width = (sh - sl + 1) * tex_nibs / 2;
    memcpy(mem, R4300::pages[0] + ram, width);
  }

  void shade_triangle(RDPCommand &cmd) {
    std::array<uint32_t, 16> instr = fetch<16>(pc);
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

  void tex_triangle(RDPCommand &cmd) {
    std::array<uint32_t, 16> instr = fetch<16>(pc);
    //for (uint8_t i = 0; i < 16; ++i) printf("%x ", instr[i]);
    //printf("\n");
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

  void zbuf_triangle(RDPCommand &cmd) {
    std::array<uint32_t, 4> instr = fetch<4>(pc);
    cmd.tex[3] = instr[0], cmd.tde[3] = instr[2];
    cmd.tdx[3] = instr[1];
  }

  template <uint8_t type>
  void triangle() {
    std::array<uint32_t, 8> instr = fetch<8>(pc);
    uint32_t t = type | ((instr[0] >> 19) & 0x10);
    RDPCommand cmd = {
      .type = t, .tile = (instr[0] >> 16) & 0x7,
      .xh = sext(instr[4], 30), .xm = sext(instr[6], 30),
      .xl = sext(instr[2], 30), .yh = sext(instr[1], 14),
      .ym = sext(instr[1] >> 16, 14), .yl = sext(instr[0], 14),
      .sh = sext(instr[5]), .sm = sext(instr[7]), .sl = sext(instr[3]),
    };
    cmd.state = state;
    if (type & 0x4) shade_triangle(cmd);
    if (type & 0x2) tex_triangle(cmd);
    if (type & 0x1) zbuf_triangle(cmd);
    if (pc <= pc_end) {
      Vulkan::add_rdp_cmd(cmd);
      printf("[RDP] Triangle of type %x\n", type);
      for (uint8_t i = 0; i < 8; ++i) printf("%x ", instr[i]);
      if (instr[0] == 0xcf800156)
        Vulkan::dump_next = true;
      printf("\n");
    }
  }

  template <bool flip>
  void tex_rectangle(RDPCommand &cmd) {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    (flip ? cmd.tex[1] : cmd.tex[0]) = (instr[0] >> 16) << 16;
    (flip ? cmd.tex[0] : cmd.tex[1]) = (instr[0] & 0xffff) << 16;
    (flip ? cmd.tdx[0] : cmd.tde[1]) = (instr[1] & 0xffff) << 11;
    (flip ? cmd.tde[1] : cmd.tdx[0]) = (instr[1] >> 16) << 11;
    //printf("[RDP] tex = %x, %x\n", cmd.tex[0], cmd.tex[1]);
    if (state.modes[0] & 0x200000) cmd.tdx[0] = 0x200000;
    cmd.tde[0] = 0, cmd.tdx[1] = 0;
  }

  template <uint8_t type>
  void rectangle() {
    //printf("[RDP] Rectangle of type %x\n", type);
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    RDPCommand cmd = {
      .type = type, .tile = (instr[1] >> 24) & 0x7,
      .xh = zext(instr[1] >> 12, 12) << 14,
      .xl = zext(instr[0] >> 12, 12) << 14,
      .yh = zext(instr[1], 12), .yl = zext(instr[0], 12),
    };
    cmd.state = state;
    if (type == 0xa) tex_rectangle<false>(cmd);
    if (type == 0xb) tex_rectangle<true>(cmd);
    if (pc <= pc_end) Vulkan::add_rdp_cmd(cmd);
  }

  void invalid() {
    std::array<uint32_t, 2> instr = fetch<2>(pc);
    printf("[RDP] Unimplemented instruction %x%x\n", instr[0], instr[1]);
    //exit(1);
  }

  void update() {
    if (pc >= pc_end) return;
    pc -= offset, offset = 0;
    // interpret config instructions 
    while (Sched::until >= 0) {
      uint64_t start = pc;
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
        case 0x32: set_tile_size(); break;
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
      if (pc > pc_end) pc = pc_end, offset = pc_end - start;
      if (pc == pc_end) { status |= 0x80; return; }
    }
    Sched::add(update, 0);
  }
}

#endif
