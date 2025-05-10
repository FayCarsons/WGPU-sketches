const NUM_SAMPLES: i32 = 32;
const MIN_DIVISIONS: f32 = 1.;
const MAX_ITERATIONS: i32 = 7;

const BORDER_COLOR: vec3f =  vec3f(0.95, 0.05, 0.05);
const SMOOTHING: f32 = 0.333;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  
  return vec4f(pos[vertexIndex], 0.0, 1.0);
}

// RENDERING
@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var currentFrame: texture_2d<f32>;
@group(0) @binding(3) var previousFrame: texture_2d<f32>;
@group(0) @binding(4) var previousDifference: texture_2d<f32>;

struct Uniforms {
  size: vec2f,
  threshold: f32,
  time: f32,
}

struct FragmentOutput {
  @location(0) fragColor: vec4f,
  @location(1) difference: vec4f,
}

fn hash22(p: vec2f) -> f32 {
  let n = sin(dot(p, vec2f(41.0, 289.0)));
  return fract(n * 43758.5453);
}

fn difference(coord: vec2f) -> vec3f {
  let curr = textureSample(currentFrame, texSampler, coord).rgb;
  let prev = textureSample(previousFrame, texSampler, coord).rgb;

  let absDiff = abs(curr - prev);
  let prevDiff = textureSample(previousDifference, texSampler, coord).rgb;
  return mix(prevDiff, absDiff, SMOOTHING);
}

// Calculate difference between frames (our version)
fn variation(center: vec2f, size: f32) -> f32 {
   var totalDifference = 0.0;
  
  // Sample several points within the quad
  for (var i = 0; i < NUM_SAMPLES; i++) {
    // Generate consistent random offset
    let r = hash22(center.xy + vec2f(f32(i), 0.0)) - 0.5;
    let sampleCoord = clamp(center + r * size, vec2f(0.001), vec2f(0.999));
    
    // Sample current and previous frame
    let curr = textureSample(currentFrame, texSampler, sampleCoord).rgb;
    let prev = textureSample(previousFrame, texSampler, sampleCoord).rgb;
    
    // Calculate absolute difference
    let diff = abs(curr - prev);
    
    // Convert to perceptual luminance difference
    let luminanceDiff = dot(diff, vec3f(0.299, 0.587, 0.114));
    
    // Add to total
    totalDifference += luminanceDiff;
  }
  
  // Average difference across samples
  let avgDifference = totalDifference / f32(NUM_SAMPLES);
  
  // Apply temporal smoothing with previous frame's value
  let prevDiff = textureSample(previousDifference, texSampler, center).r;
  let smoothedDiff = mix(avgDifference, prevDiff, SMOOTHING);
  
  return smoothedDiff;
}

@fragment
fn fragmentMain(@builtin(position) fragCoord: vec4f) -> FragmentOutput {
// Calculate texture coordinates
  var uv = fragCoord.xy / uniforms.size;
    
  let aspectRatio = uniforms.size.x / uniforms.size.y;
  var aspectCorrectedUV = uv;

  if aspectRatio > 1. {
    aspectCorrectedUV.y = (uv.y - 0.5) * aspectRatio + 0.5;
  } else {
    aspectCorrectedUV.x = (uv.x - 0.5) / aspectRatio + 0.5;
  }

  // Pre-compute all possible quad centers and sizes for all iterations
  var quadCenters: array<vec2f, MAX_ITERATIONS>;
  var quadSizes: array<f32, MAX_ITERATIONS>;
  var divisions = MIN_DIVISIONS;
  
  // Initialize first level
  quadCenters[0] = (floor(uv * divisions) + 0.5) / divisions;
  quadSizes[0] = 1.0 / divisions;
  
  // Pre-compute all possible subdivisions
  for (var i = 1; i < MAX_ITERATIONS; i++) {
    divisions *= 2.0;
    quadCenters[i] = (floor(uv * divisions) + 0.5) / divisions;
    quadSizes[i] = 1.0 / divisions;
  }
  
  // Calculate movement for all levels
  var quadMovements: array<f32, MAX_ITERATIONS>;
  for (var i = 0; i < MAX_ITERATIONS; i++) {
    quadMovements[i] = variation(quadCenters[i], quadSizes[i]);
  }
  
  // Determine the appropriate subdivision level using a uniform approach
  var finalLevel = 0;
  var finalDivisions = MIN_DIVISIONS;
  
  // Uniform control flow approach
  for (var i = 0; i < MAX_ITERATIONS - 1; i++) {
    // Continue subdividing if movement is above threshold
    let shouldContinue = quadMovements[i] > uniforms.threshold && i >= finalLevel;
    finalLevel = select(finalLevel, i + 1, shouldContinue);
    finalDivisions = select(finalDivisions, pow(2.0, f32(i + 1)) * MIN_DIVISIONS, shouldContinue);
  }
  
  // Get video color
  var color = textureSample(currentFrame, texSampler, uv).rgb;
  
  // Draw borders
  let quadLocation = fract(uv * finalDivisions);
  let lineWidth = vec2f(1.0/uniforms.size.x, 1.0/uniforms.size.y);
  let uvAbs = abs(quadLocation - 0.5);
  let border = step(0.5 - uvAbs.x, lineWidth.x * finalDivisions) + 
               step(0.5 - uvAbs.y, lineWidth.y * finalDivisions);
  
  // Apply border
  color = mix(color, 1. - color, clamp(border, 0.0, 1.0));
  
  // Return final color and movement value for next frame
  // Use the movement at the final subdivision level for the output
  return FragmentOutput(
    vec4f(color, 1.0),
    vec4f(vec3f(quadMovements[finalLevel]), 1.0)
  );
}
