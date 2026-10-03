import package::common::fullscreen_triangle_xy;

// Copies a multisampled depth buffer into a single-sampled one. Each pixel keeps
// its nearest sample, so a silhouette stays on the surface in front.

@group(0) @binding(0)
var t_depth: texture_depth_multisampled_2d;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_xy(vid), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let p = vec2<i32>(pos.xy);
    var depth = 1.0;
    for (var s = 0u; s < textureNumSamples(t_depth); s++) {
        depth = min(depth, textureLoad(t_depth, p, i32(s)));
    }
    return depth;
}
