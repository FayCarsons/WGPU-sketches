/// <reference types="@webgpu/types" />

interface Vec2 {
  readonly x: number;
  readonly y: number;
};
type ResourceBuilder = {
  adapter: GPUAdapter | null;
  device: GPUDevice | null;
  context: GPUCanvasContext | null;
  pipeline: GPURenderPipeline | null;
  sampler: GPUSampler | null;
  uniformBuffer: GPUBuffer | null;
  videoBuffer: PingPongBuffer | null;
  differenceBuffer: PingPongBuffer | null;
}

type GPUResources = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  sampler: GPUSampler;
  uniformBuffer: GPUBuffer;
  videoBuffer: PingPongBuffer;
  differenceBuffer: PingPongBuffer;
}

function buildResources(builder: ResourceBuilder): GPUResources | Error {
  if (Object.entries(builder).some(([k, v]) => v === null)) {
    throw new Error ("Did not finish building Resources object")
  } else {
    const bindGroup = createBindGroup(builder)
    return {
      ...builder,
      bindGroup,
    } as GPUResources
  }
}

type AppState = {
  readonly canvas: HTMLCanvasElement;
  readonly video: HTMLVideoElement;
  readonly resources: GPUResources;
  readonly canvasSize: Vec2;
  readonly time: number;
  readonly threshold: number;
};

type WebGPUConfig = {
  readonly powerPreference: GPUPowerPreference;
  readonly presentationFormat: GPUTextureFormat;
};

const DEFAULT_THRESHOLD = 0.1
const DELTA = 0.001

function ix(i: boolean): number {
  return i ? 1 : 0
}

async function fetchShader(): Promise<string | Error> {
  try {
    const req = await fetch('main.wgsl');
    const res = await req.text();
    return res
  } catch (e) {
    return e as Error
  }
}

const log = console.log
const warn = console.warn

function fail(e: Error) {
  console.error("Error accessing webcam:", e);
  document.body.innerHTML = `
      <div style="color: red; font-family: sans-serif; margin: 20px;">
        Error accessing webcam: ${e.message}<br>
        Please check your camera permissions.
      </div>
    `;
}

function checkWebGPUSupport(): void {
  if (!navigator.gpu) {
    document.body.innerHTML = `
      <div style="color: red; font-family: sans-serif; margin: 20px;">
        Your browser doesn't support WebGPU! <br>
        Try Chrome 113+ or Edge 113+ with the #enable-unsafe-webgpu flag enabled.
      </div>
    `;
    throw new Error("WebGPU not supported");
  } else {
    console.log('Got WebGPU support!')
  }
};

function createCanvas(): HTMLCanvasElement {
  const canvas = document.getElementById("wgpu") as HTMLCanvasElement;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);
  return canvas;
};

function createVideo(): HTMLVideoElement {
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  return video;
};

async function initializeAdapter(config: WebGPUConfig): Promise<GPUAdapter> {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: config.powerPreference
  });

  if (!adapter) {
    throw new Error("No WebGPU adapter found");
  }

  return adapter;
};

async function initializeDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  return await adapter.requestDevice({
    requiredFeatures: ["texture-compression-bc"],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  });
};

function initializeContext(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  format: GPUTextureFormat
): GPUCanvasContext {
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  if (!context) {
    throw new Error("Failed to get WebGPU context");
  }

  context.configure({
    device,
    format,
    alphaMode: "premultiplied"
  });

  return context;
};

function createSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge"
  });
};

async function createShaderModule(device: GPUDevice): Promise<GPUShaderModule | null> {
  const code = await fetchShader();

  if (code instanceof Error) {
    fail(code)
    return null
  } else return device.createShaderModule({
    code
  });
};

function createRenderPipeline(
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  format: GPUTextureFormat,
  bindGroupLayout: GPUBindGroupLayout
): GPURenderPipeline {
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vertexMain",
      buffers: [] // Using vertex_index
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fragmentMain",
      targets: [
        { format },
        { format: "r8unorm" }
      ]
    },
    primitive: {
      topology: "triangle-list"
    }
  });
};

function createUniformBuffer(device: GPUDevice, size: Vec2): GPUBuffer {
  const buffer = device.createBuffer({
    size: 16, // width, height, threshold, time
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });

  const arrayBuffer = buffer.getMappedRange();
  new Float32Array(arrayBuffer).set([
    size.x // width
    , size.y // height
    , DEFAULT_THRESHOLD // thresh
    , 0 // time
  ]);
  buffer.unmap();

  return buffer;
};

function updateUniformBuffer(state: AppState): AppState {
  state.resources.device.queue.writeBuffer(
    state.resources.uniformBuffer,
    0,
    new Float32Array([
      state.canvasSize.x // width
      , state.canvasSize.y // height
      , state.threshold // thresh
      , state.time // time
    ])
  );

  return state
};

class PingPongBuffer {
  textures: [GPUTexture, GPUTexture];
  views: [GPUTextureView, GPUTextureView];
  private index: boolean;

  constructor(device: GPUDevice, { x, y }: Vec2, format: GPUTextureFormat = "rgba8unorm") {
    const textures: GPUTexture[] = [];
    const views: GPUTextureView[] = [];

    // Create two textures for ping-pong buffering
    for (let i = 0; i < 2; i++) {
      const texture = device.createTexture({
        size: { width: x, height: y, depthOrArrayLayers: 1 },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });

      textures.push(texture);
      views.push(texture.createView());
    }

    this.textures = textures as [GPUTexture, GPUTexture];
    this.views = views as [GPUTextureView, GPUTextureView];
    this.index = false;
  }

  front(): GPUTexture {
    return this.textures[ix(this.index)]
  }

  back(): GPUTexture {
    return this.textures[ix(!this.index)]
  }


  frontView(): GPUTextureView {
    return this.views[ix(this.index)]
  }

  backView(): GPUTextureView {
    return this.views[ix(!this.index)]
  }

  swap() {
    this.index = !this.index
  }

  layout(bindingOffset: number): Iterable<GPUBindGroupLayoutEntry> {
    return [
      {
        binding: bindingOffset,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {}
      },
      {
        binding: bindingOffset + 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {}
      }
    ]
  }
}

// Create bind group layout with just what we need:
// - sampler
// - current frame texture
// - previous frame texture
// - uniform buffer
function createBindGroupLayout(device: GPUDevice, videoBuffer: PingPongBuffer, differenceBuffer: PingPongBuffer): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      {
        binding: 0,  // Sampler
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {}
      },
      {
        binding: 1,  // Uniforms (resolution, threshold, time)
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform"
        }
      },
      ...videoBuffer.layout(2),
      {
        binding: 4, 
        visibility: GPUShaderStage.FRAGMENT,
        texture: {}
      }
    ]
  });
}

// Create a single bind group
function createBindGroup(
  {device, sampler, pipeline, uniformBuffer, videoBuffer, differenceBuffer }: ResourceBuilder,
): GPUBindGroup {
  const layout = pipeline?.getBindGroupLayout(0) as GPUBindGroupLayout

  return device?.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: sampler } as GPUBindGroupEntry,
      { binding: 1, resource: { buffer: uniformBuffer } } as GPUBindGroupEntry,
      { binding: 2, resource: videoBuffer?.frontView() } as GPUBindGroupEntry, // Current frame
      { binding: 3, resource: videoBuffer?.backView() } as GPUBindGroupEntry, // Previous frame
      { binding: 4, resource: differenceBuffer?.backView() } as GPUBindGroupEntry, // Read difference
    ]
  }) as GPUBindGroup;
}

function createVideoTexture(
  device: GPUDevice,
  video: HTMLVideoElement
): GPUTexture {
  return device.createTexture({
    size: {
      width: video.videoWidth,
      height: video.videoHeight,
      depthOrArrayLayers: 1
    },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });
};

// Update video texture with the latest frame
function updateVideoTexture(
  state: AppState
): AppState {
  state.resources.device.queue.copyExternalImageToTexture(
    { source: state.video },
    { texture: state.resources.videoBuffer.front() },
    { width: state.video.videoWidth, height: state.video.videoHeight }
  )

  state.resources.videoBuffer.swap()
  return state
}

async function setupWebcam(video: HTMLVideoElement): Promise<HTMLVideoElement | null> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        facingMode: 'user'
      }
    });

    video.srcObject = stream;

    // Wait for video to be ready
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => {
        video.play().then(() => resolve());
      };
    });

    return video;
  } catch (e) {
    fail(e as Error)
    return null
  }
};


function onResize(state: AppState): AppState {
  log('Resizing!')
  const newSize: Vec2 = {
    x: window.innerWidth,
    y: window.innerHeight
  };

  state.canvas.width = newSize.x;
  state.canvas.height = newSize.y;

  return {
    ...state,
    canvasSize: newSize
  };
};

function render(
  state: AppState,
): AppState {
  const newTime = state.time + DELTA

  // Create command encoder
  const commandEncoder = state.resources.device.createCommandEncoder();

  // copy the video to the current frame texture
  state = updateVideoTexture(state);

  // Update uniform buffer with time and other parameters
  state.resources.device.queue.writeBuffer(
    state.resources.uniformBuffer,
    0,
    new Float32Array([
      state.canvasSize.x,
      state.canvasSize.y,
      state.threshold,
      newTime,
    ])
  );

  // Render to the canvas
  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: state.resources.context.getCurrentTexture().createView(),
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store"
      },
      {
        view: state.resources.differenceBuffer.frontView(),
        loadOp: 'clear',
        storeOp: 'store'
      }
    ]
  };

  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
  passEncoder.setPipeline(state.resources.pipeline);
  passEncoder.setBindGroup(0, state.resources.bindGroup);
  passEncoder.draw(3);
  passEncoder.end();

  state.resources.device.queue.submit([commandEncoder.finish()]);
  state.resources.differenceBuffer.swap();

  // After rendering, create a new bind group that swaps the textures
  const newBindGroup = createBindGroup(
    state.resources
  );

  // Return state with updated bind group and currentFrameIndex
  return {
    ...state,
    resources: {
      ...state.resources,
      bindGroup: newBindGroup
    },
    time: newTime
  };
};

// Main initialization function
async function init(): Promise<void> {
  checkWebGPUSupport();

  const canvas = createCanvas();
  log('Got canvas')
  const video = createVideo();
  log('Created video')


  const config: WebGPUConfig = {
    powerPreference: "high-performance",
    presentationFormat: navigator.gpu.getPreferredCanvasFormat()
  };

  const adapter = await initializeAdapter(config);
  const device = await initializeDevice(adapter);
  const context = initializeContext(canvas, device, config.presentationFormat);

  const canvasSize: Vec2 = {
    x: canvas.width,
    y: canvas.height
  };

  // Create uniform buffer for canvas size
  const uniformBuffer = createUniformBuffer(device, canvasSize);

  const shaderModule = await createShaderModule(device);

  if (!shaderModule) {
    return
  }


  const sampler = createSampler(device);

  const initializedVideo = await setupWebcam(video);
  if (!initializedVideo) return

  const videoBuffer = new PingPongBuffer(device, { x: video.videoWidth, y: video.videoHeight })
  const differenceBuffer = new PingPongBuffer(device, canvasSize, 'r8unorm')

  // Create bind group layout, shader, and pipeline
  const bindGroupLayout = createBindGroupLayout(device, videoBuffer, differenceBuffer);
  const pipeline = createRenderPipeline(
    device,
    shaderModule,
    config.presentationFormat,
    bindGroupLayout
  );

  const builder: ResourceBuilder = {
      adapter,
      device,
      context,
      pipeline,
      videoBuffer, 
      differenceBuffer,
      sampler,
      uniformBuffer,
    }

  const bindGroup = createBindGroup(builder);

  let resources = buildResources(builder) as GPUResources

  let globalState: AppState = {
    canvas,
    video: initializedVideo,
    resources,
    canvasSize,
    threshold: DEFAULT_THRESHOLD,
    time: 0,
  };

  const writeState = (f: (state: AppState) => AppState) => {
    globalState = f(globalState)
  }

  // Window resize handler, now updates the global state
  window.addEventListener("resize", () => {
    writeState(onResize)
  });

  const onVideoFrame: (_: any, __: any) => void = (_, __) => {
    writeState(render);
    video.requestVideoFrameCallback(onVideoFrame)
  }

  video.requestVideoFrameCallback(onVideoFrame)
};

init();
