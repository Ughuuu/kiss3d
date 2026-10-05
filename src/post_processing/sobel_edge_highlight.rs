//! A post-processing effect to highlight edges.
//!
//! The edges are found in the scene's depth, which the window hands every pass
//! of a chain holding this effect, so it works first, last or in between.

use crate::context::Context;
use crate::post_processing::post_processing_effect::{
    FormatPipelines, PostProcessingContext, PostProcessingEffect,
};
use crate::resource::RenderTarget;
use bytemuck::{Pod, Zeroable};

/// Vertex data for full-screen quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

/// Uniforms for Sobel edge highlight effect.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SobelUniforms {
    nx: f32, // 2.0 / width (pixel step in x)
    ny: f32, // 2.0 / height (pixel step in y)
    znear: f32,
    zfar: f32,
    threshold: f32,
    _padding: [f32; 3],
}

/// Post processing effect which draws detected edges on top of the original buffer.
pub struct SobelEdgeHighlight {
    pipeline: FormatPipelines,
    color_bind_group_layout: wgpu::BindGroupLayout,
    depth_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    shiftx: f32,
    shifty: f32,
    zn: f32,
    zf: f32,
    threshold: f32,
}

impl SobelEdgeHighlight {
    /// Creates a new SobelEdgeHighlight post processing effect.
    pub fn new(threshold: f32) -> SobelEdgeHighlight {
        let ctxt = Context::get();

        // Create bind group layout for color texture + sampler
        let color_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sobel_color_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // The depth texture, read texel by texel as unfilterable float: GLES has no
        // other way to read a depth texture without a comparison.
        let depth_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sobel_depth_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        // Create bind group layout for uniforms
        let uniform_bind_group_layout =
            ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sobel_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sobel_pipeline_layout"),
            bind_group_layouts: &[
                Some(&color_bind_group_layout),
                Some(&depth_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Load shader
        let shader = ctxt.create_shader_module(
            Some("sobel_shader"),
            &crate::builtin::linked("package::sobel", &[]),
        );

        // Vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let pipeline = FormatPipelines::new(move |format| {
            let ctxt = Context::get();
            ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sobel_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(vertex_buffer_layout.clone())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        });

        // Create full-screen quad vertices
        let vertices = [
            QuadVertex {
                position: [-1.0, -1.0],
            },
            QuadVertex {
                position: [1.0, -1.0],
            },
            QuadVertex {
                position: [-1.0, 1.0],
            },
            QuadVertex {
                position: [1.0, 1.0],
            },
        ];

        let vertex_buffer = ctxt.create_buffer_init(
            Some("sobel_vertex_buffer"),
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
        );

        // Create uniform buffer
        let uniform_buffer = ctxt.create_buffer_simple(
            Some("sobel_uniform_buffer"),
            std::mem::size_of::<SobelUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Create uniform bind group
        let uniform_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        SobelEdgeHighlight {
            pipeline,
            color_bind_group_layout,
            depth_bind_group_layout,
            uniform_bind_group_layout,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            shiftx: 0.0,
            shifty: 0.0,
            zn: 0.0,
            zf: 0.0,
            threshold,
        }
    }
}

impl PostProcessingEffect for SobelEdgeHighlight {
    fn update(&mut self, _: f32, w: f32, h: f32, znear: f32, zfar: f32) {
        self.shiftx = 2.0 / w;
        self.shifty = 2.0 / h;
        self.zn = znear;
        self.zf = zfar;
    }

    fn reads_depth(&self) -> bool {
        true
    }

    fn draw(&mut self, target: &RenderTarget, context: &mut PostProcessingContext) {
        let pipeline = self.pipeline.get(context.output_format);
        let ctxt = Context::get();

        // Get the source textures and sampler from the render target
        let (color_view, depth_view, sampler) = match target {
            RenderTarget::Offscreen(o) => (&o.color_view, &o.depth_view, &o.sampler),
            RenderTarget::Screen => return, // Can't post-process the screen directly
        };

        // Update uniforms
        let uniforms = SobelUniforms {
            nx: self.shiftx,
            ny: self.shifty,
            znear: self.zn,
            zfar: self.zf,
            threshold: self.threshold,
            _padding: [0.0; 3],
        };
        ctxt.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create color texture bind group for this frame
        let color_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel_color_bind_group"),
            layout: &self.color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        // Create depth texture bind group for this frame
        let depth_bind_group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel_depth_bind_group"),
            layout: &self.depth_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            }],
        });

        // Create render pass to the output view
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sobel_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: context.output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

            render_pass.set_pipeline(&pipeline);
            render_pass.set_bind_group(0, &color_bind_group, &[]);
            render_pass.set_bind_group(1, &depth_bind_group, &[]);
            render_pass.set_bind_group(2, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SobelEdgeHighlight;
    use crate::camera::OrbitCamera3d;
    use crate::color::{Color, WHITE};
    use crate::post_processing::{Grayscales, PostProcessingEffect};
    use crate::scene::SceneNode3d;
    use crate::test_gpu::{on_gpu, on_gpu_with};
    use crate::window::{CanvasSetup, NumSamples, OffscreenSurface};
    use glamx::{Quat, Vec3};

    /// Pixels of a bright cube on a bright background left dark by `chain`.
    async fn dark_pixels(
        surface: &mut OffscreenSurface,
        chain: &mut [&mut dyn PostProcessingEffect],
    ) -> usize {
        let mut scene = SceneNode3d::empty();
        scene
            .add_cube(2.0, 2.0, 2.0)
            .set_emissive(Color::new(4.0, 4.0, 4.0, 1.0))
            .set_rotation(Quat::from_rotation_z(0.4));
        let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
        surface.set_background_color(WHITE);
        surface
            .render_chain(Some(&mut scene), None, Some(&mut camera), None, None, chain)
            .await;
        surface
            .snap_image()
            .pixels()
            .filter(|p| p.0[1] < 64)
            .count()
    }

    /// The same, with `chain` run on the film before the tonemap.
    async fn dark_film_pixels(
        surface: &mut OffscreenSurface,
        chain: &mut [&mut dyn PostProcessingEffect],
    ) -> usize {
        let mut scene = SceneNode3d::empty();
        scene
            .add_cube(2.0, 2.0, 2.0)
            .set_emissive(Color::new(4.0, 4.0, 4.0, 1.0))
            .set_rotation(Quat::from_rotation_z(0.4));
        let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
        surface.set_background_color(WHITE);
        surface
            .render_chains(
                Some(&mut scene),
                None,
                Some(&mut camera),
                None,
                None,
                chain,
                &mut [],
            )
            .await;
        surface
            .snap_image()
            .pixels()
            .filter(|p| p.0[1] < 64)
            .count()
    }

    #[test]
    fn sobel_outlines_a_cube_in_the_film_chain() {
        on_gpu(64, 64, async |surface| {
            let mut sobel = SobelEdgeHighlight::new(4.0);
            let edges = dark_film_pixels(surface, &mut [&mut sobel]).await;
            assert!(edges > 40, "edge pass on the film: {}", edges);
        });
    }

    async fn sobel_outlines_the_cube_wherever_it_sits(surface: &mut OffscreenSurface) {
        let mut gray = Grayscales::new();
        let mut sobel = SobelEdgeHighlight::new(4.0);
        let plain = dark_pixels(surface, &mut [&mut gray]).await;
        let after = dark_pixels(surface, &mut [&mut gray, &mut sobel]).await;
        let before = dark_pixels(surface, &mut [&mut sobel, &mut gray]).await;
        assert!(plain < 4, "without the edge pass: {}", plain);
        assert!(after > 40, "edge pass after grayscale: {}", after);
        assert!(before > 40, "edge pass before grayscale: {}", before);
    }

    #[test]
    fn sobel_outlines_a_cube_after_another_pass() {
        on_gpu(64, 64, async |surface| {
            sobel_outlines_the_cube_wherever_it_sits(surface).await
        });
    }

    #[test]
    fn sobel_outlines_a_multisampled_cube_after_another_pass() {
        let setup = CanvasSetup {
            samples: NumSamples::Four,
            ..CanvasSetup::default()
        };
        on_gpu_with(setup, async |surface| {
            sobel_outlines_the_cube_wherever_it_sits(surface).await
        });
    }
}
