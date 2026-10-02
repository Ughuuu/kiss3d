//! The scene's depth, single-sampled, for the post-processing passes that read it.

use crate::context::Context;

/// Resolves a multisampled scene depth into a single-sampled texture an effect
/// can bind as `texture_depth_2d`.
pub(crate) struct SceneDepth {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    target: Option<wgpu::TextureView>,
}

impl SceneDepth {
    pub(crate) fn new() -> SceneDepth {
        let ctxt = Context::get();
        let layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_depth_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_depth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = ctxt.create_shader_module(
            Some("scene_depth_shader"),
            &crate::builtin::compile_shader_with_common(
                "package::depth_resolve",
                include_str!("../builtin/depth_resolve.wgsl"),
            ),
        );
        let pipeline = ctxt.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_depth_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Context::depth_format(),
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        SceneDepth {
            layout,
            pipeline,
            target: None,
        }
    }

    /// The multisampled `depth` resolved into a single-sampled texture this
    /// keeps, remade when the source changes size.
    pub(crate) fn resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) -> wgpu::TextureView {
        let size = depth.texture().size();
        let stale = self
            .target
            .as_ref()
            .is_none_or(|t| t.texture().size() != size);
        if stale {
            let texture = Context::get().create_texture(&wgpu::TextureDescriptor {
                label: Some("scene_depth_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Context::depth_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.target = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        }
        let target = self.target.clone().expect("the target was just made");

        let bind_group = Context::get().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_depth_bind_group"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene_depth_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        target
    }
}
