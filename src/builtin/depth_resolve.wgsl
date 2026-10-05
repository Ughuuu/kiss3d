import package::common::fullscreen_triangle_xy;

// Copies a multisampled depth buffer into a single-sampled one. Each pixel keeps
// its nearest sample, so a silhouette stays on the surface in front.

// Read as unfilterable float rather than as depth: GLSL cannot `texelFetch` a
// depth texture, and GLES has no other way to read one sample.
@group(0) @binding(0)
var t_depth: texture_multisampled_2d<f32>;

// The source's sample count, set per pipeline: GLES has no `textureNumSamples`.
override samples: u32 = 4u;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_xy(vid), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let p = vec2<i32>(pos.xy);
    var depth = 1.0;
    for (var s = 0u; s < samples; s++) {
        depth = min(depth, textureLoad(t_depth, p, i32(s)).r);
    }
    return depth;
}
