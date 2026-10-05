//! Clustered (forward+) lighting.
//!
//! The view frustum is divided into a 3D grid of clusters
//! ([`DEFAULT_GRID`] unless [`Window::set_cluster_grid`](crate::window::Window::set_cluster_grid)
//! says otherwise). A compute pass builds each cluster's
//! view-space AABB (only when the projection or viewport changes), and a second
//! compute pass culls the scene's "clustered" lights into the clusters they touch,
//! writing a per-cluster `(offset, count)` slice into a global light-index list.
//! The object material's fragment shader then shades each pixel using only the
//! lights of the cluster it falls in, letting hundreds of (shadowless) point and
//! spot lights scale far better than the fixed primary-light loop.
//!
//! This path requires compute shaders and fragment-stage storage buffers, so it is
//! only used when [`Context::supports_clustered_lighting`](crate::context::Context::supports_clustered_lighting)
//! is true (native + WebGPU); WebGL2 falls back to the legacy fixed-light path.
//!
//! Shadows are not yet integrated for clustered lights — see the design notes; the
//! fixed primary tier (see [`LightCollection::split_primary_clustered`]) keeps full
//! shadow support.
//!
//! [`LightCollection::split_primary_clustered`]: crate::light::LightCollection::split_primary_clustered

use crate::builtin::object_material::GpuLight;
use crate::camera::Camera3d;
use crate::context::Context;
use crate::light::LightCollection;
use bytemuck::{Pod, Zeroable};
use glamx::Mat4;

const BUILD_SRC: &str = include_str!("clustered_build.wgsl");
const CULL_SRC: &str = include_str!("clustered_cull.wgsl");

/// The cluster grid a window starts with: clusters along screen X, screen Y and
/// view depth.
pub const DEFAULT_GRID: [u32; 3] = [16, 9, 24];
/// The lights a cluster records by default; lights past it in a very dense cluster
/// are dropped.
pub const DEFAULT_MAX_LIGHTS_PER_CLUSTER: u32 = 256;
/// The most clusters along any one axis of the grid.
pub const MAX_GRID_AXIS: u32 = 128;

/// Uniforms shared by the cluster-build and light-cull compute passes.
///
/// Mirrors the WGSL `ClusterUniforms` in `clustered_build.wgsl` / `clustered_cull.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ClusterUniforms {
    /// Inverse projection matrix (clip → view), for unprojecting tile corners.
    inv_proj: [[f32; 4]; 4],
    /// View matrix (world → view), for transforming light positions during culling.
    view: [[f32; 4]; 4],
    /// (grid_x, grid_y, grid_z, num_clustered_lights).
    grid: [u32; 4],
    /// (screen_width_px, screen_height_px, tile_width_px, tile_height_px).
    screen: [f32; 4],
    /// (z_near, z_far, ln(z_far / z_near), unused).
    depth: [f32; 4],
}

/// Owns the GPU buffers and (later) compute pipelines for clustered lighting.
pub(crate) struct Clustered {
    /// Number of `GpuLight`s the `clustered_lights` buffer can currently hold.
    light_capacity: u32,
    /// All clustered (overflow) lights for the frame, std430 `array<GpuLight>`.
    clustered_lights: wgpu::Buffer,
    /// Per-cluster view-space AABBs, written by the build pass.
    cluster_aabbs: wgpu::Buffer,
    /// Per-cluster `(offset, count)` into `light_index_list`, written by cull.
    cluster_grid: wgpu::Buffer,
    /// Global light-index list, written by cull, read by the fragment shader.
    /// Each cluster owns a fixed `max_lights_per_cluster` slice at `cluster * stride`.
    light_index_list: wgpu::Buffer,
    /// Clusters along screen X, screen Y and view depth.
    grid: [u32; 3],
    /// Lights a cluster records.
    max_lights_per_cluster: u32,
    /// Compute-pass uniforms.
    uniforms: wgpu::Buffer,
    /// Cached key gating AABB rebuilds: (projection hash, width, height).
    aabb_key: Option<(u64, u32, u32)>,
    /// Last viewport the grid was sized for.
    width: u32,
    height: u32,
    /// AABB-build compute pipeline + its bind group layout.
    build_pipeline: wgpu::ComputePipeline,
    build_layout: wgpu::BindGroupLayout,
    /// Light-cull compute pipeline + its bind group layout.
    cull_pipeline: wgpu::ComputePipeline,
    cull_layout: wgpu::BindGroupLayout,
}

impl Clustered {
    /// Creates the clustered-lighting resources for the given viewport and grid.
    pub(crate) fn new(width: u32, height: u32, grid: [u32; 3], max_per_cluster: u32) -> Clustered {
        let ctxt = Context::get();

        let (grid, max_lights_per_cluster) = fit_grid(&ctxt, grid, max_per_cluster);
        let clustered_lights = alloc_lights(&ctxt, 1);
        let (cluster_aabbs, cluster_grid, light_index_list) =
            alloc_grid(&ctxt, grid, max_lights_per_cluster);
        let uniforms = ctxt.create_buffer_simple(
            Some("clustered_uniforms"),
            std::mem::size_of::<ClusterUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // AABB-build pass: uniform + read_write AABB buffer.
        let build_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clustered_build_layout"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        let build_pipeline = make_pipeline(
            &ctxt,
            "clustered_build",
            BUILD_SRC,
            "build_aabbs",
            &build_layout,
        );

        // Cull pass: uniform + aabbs(ro) + lights(ro) + grid(rw) + index(rw).
        let cull_layout = ctxt.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clustered_cull_layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let cull_pipeline = make_pipeline(
            &ctxt,
            "clustered_cull",
            CULL_SRC,
            "cull_lights",
            &cull_layout,
        );

        Clustered {
            light_capacity: 1,
            clustered_lights,
            cluster_aabbs,
            cluster_grid,
            light_index_list,
            grid,
            max_lights_per_cluster,
            uniforms,
            aabb_key: None,
            width: width.max(1),
            height: height.max(1),
            build_pipeline,
            build_layout,
            cull_pipeline,
            cull_layout,
        }
    }

    /// Updates the viewport size. Forces an AABB rebuild on the next [`run`](Self::run).
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if self.width != w || self.height != h {
            self.width = w;
            self.height = h;
            self.aabb_key = None;
        }
    }

    /// The clustered-light storage buffer (bound by the object material fragment shader).
    pub(crate) fn lights_buffer(&self) -> &wgpu::Buffer {
        &self.clustered_lights
    }

    /// The per-cluster light-grid buffer (bound by the object material fragment shader).
    pub(crate) fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.cluster_grid
    }

    /// The global light-index list buffer (bound by the object material fragment shader).
    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.light_index_list
    }

    /// Grows `clustered_lights` (grow-only) so it can hold at least `needed` lights.
    /// Returns `true` if the buffer was reallocated (its handle changed).
    fn ensure_light_capacity(&mut self, ctxt: &Context, needed: u32) -> bool {
        if needed <= self.light_capacity {
            return false;
        }
        // Round up to the next power of two for headroom, clamped to the device's
        // maximum storage-buffer binding size.
        let max_lights = (ctxt.device.limits().max_storage_buffer_binding_size / 64).max(1) as u32;
        let cap = needed
            .next_power_of_two()
            .min(max_lights)
            .max(needed.min(max_lights));
        self.clustered_lights = alloc_lights(ctxt, cap);
        self.light_capacity = cap;
        true
    }

    /// Reallocates the grid buffers when `grid` or `max_per_cluster` differ from
    /// the current ones. Returns `true` when the buffer handles changed.
    fn configure(&mut self, ctxt: &Context, grid: [u32; 3], max_per_cluster: u32) -> bool {
        let (grid, max_per_cluster) = fit_grid(ctxt, grid, max_per_cluster);
        if grid == self.grid && max_per_cluster == self.max_lights_per_cluster {
            return false;
        }
        let (aabbs, cells, index) = alloc_grid(ctxt, grid, max_per_cluster);
        self.cluster_aabbs = aabbs;
        self.cluster_grid = cells;
        self.light_index_list = index;
        self.grid = grid;
        self.max_lights_per_cluster = max_per_cluster;
        self.aabb_key = None;
        true
    }

    /// Per-frame update: uploads the clustered lights, (re)builds cluster AABBs when
    /// the projection/viewport changed, and dispatches the light-culling pass.
    ///
    /// Returns `true` if a buffer handle changed this frame (the object material
    /// must then rebuild its frame bind group).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        lights: &LightCollection,
        shadow_slots: &[u32],
        camera: &dyn Camera3d,
        width: u32,
        height: u32,
        (grid, max_per_cluster): ([u32; 3], u32),
        gpu: &mut crate::renderer::timings::GpuTimer,
    ) -> bool {
        let ctxt = Context::get();
        self.resize(width, height);
        let regrid = self.configure(&ctxt, grid, max_per_cluster);
        let [grid_x, grid_y, grid_z] = self.grid;

        let (_primary, clustered) = lights.split_primary_clustered();
        let num = clustered.len() as u32;
        let realloc = self.ensure_light_capacity(&ctxt, num.max(1)) || regrid;

        // Upload the clustered lights (overflow point/spot lights), stamping each
        // with the shadow-metadata slot the shadow mapper assigned it this frame
        // (`u32::MAX` = no shadow). The fragment shader applies `compute_shadow`
        // for any clustered light that has a slot.
        if num > 0 {
            let gpu: Vec<GpuLight> = clustered
                .iter()
                .map(|&li| {
                    let mut l = GpuLight::from_collected(&lights.lights[li]);
                    l.set_shadow_slot(shadow_slots.get(li).copied().unwrap_or(u32::MAX));
                    l
                })
                .collect();
            ctxt.write_buffer(&self.clustered_lights, 0, bytemuck::cast_slice(&gpu));
        }

        // Compute uniforms (view, inverse projection, grid, depth).
        let (view_pose, proj) = camera.view_transform_pair(0);
        let view = view_pose.to_mat4();
        let (near, far) = camera.clip_planes();
        let tile = [
            self.width as f32 / grid_x as f32,
            self.height as f32 / grid_y as f32,
        ];
        let uniforms = ClusterUniforms {
            inv_proj: proj.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            grid: [grid_x, grid_y, grid_z, num],
            screen: [self.width as f32, self.height as f32, tile[0], tile[1]],
            depth: [
                near,
                far,
                (far / near).ln(),
                self.max_lights_per_cluster as f32,
            ],
        };
        ctxt.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));

        // Rebuild cluster AABBs only when the projection or viewport changed.
        let key = (proj_hash(&proj), self.width, self.height);
        if self.aabb_key != Some(key) {
            self.aabb_key = Some(key);
            let group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clustered_build_group"),
                layout: &self.build_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.cluster_aabbs.as_entire_binding(),
                    },
                ],
            });
            let build_ts = gpu.compute_scope("clustered");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clustered_build_pass"),
                timestamp_writes: build_ts,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(grid_x.div_ceil(4), grid_y.div_ceil(4), grid_z.div_ceil(4));
        }

        // No clustered lights → leave the grid as-is; the fragment shader gates the
        // clustered loop on `num_clustered_lights == 0`, so stale grid data is unused.
        if num == 0 {
            return realloc;
        }

        // Cull lights into clusters (each cluster writes its own fixed slice).
        let group = ctxt.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clustered_cull_group"),
            layout: &self.cull_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cluster_aabbs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.clustered_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cluster_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.light_index_list.as_entire_binding(),
                },
            ],
        });
        let cull_ts = gpu.compute_scope("clustered");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clustered_cull_pass"),
            timestamp_writes: cull_ts,
        });
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((grid_x * grid_y * grid_z).div_ceil(64), 1, 1);

        realloc
    }
}

/// A cheap order-independent hash of a projection matrix, used to detect when the
/// cluster AABBs must be rebuilt.
fn proj_hash(proj: &Mat4) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in proj.to_cols_array() {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn make_pipeline(
    ctxt: &Context,
    label: &str,
    src: &str,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = ctxt.create_shader_module(Some(label), src);
    let pipeline_layout = ctxt.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    ctxt.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Clamps each grid axis to `1..=MAX_GRID_AXIS` and the per-cluster light count to
/// at least 1 and to what keeps the light-index list within the device's largest
/// storage binding.
fn fit_grid(ctxt: &Context, grid: [u32; 3], max_per_cluster: u32) -> ([u32; 3], u32) {
    let grid = grid.map(|n| n.clamp(1, MAX_GRID_AXIS));
    let clusters = (grid[0] * grid[1] * grid[2]) as u64;
    let max_binding = ctxt.device.limits().max_storage_buffer_binding_size;
    let fits = (max_binding / (clusters * 4)).clamp(1, u32::MAX as u64) as u32;
    (grid, max_per_cluster.clamp(1, fits))
}

/// Allocates the per-cluster AABB, cell and light-index buffers for `grid`.
fn alloc_grid(
    ctxt: &Context,
    grid: [u32; 3],
    max_per_cluster: u32,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let clusters = (grid[0] * grid[1] * grid[2]) as u64;
    let aabbs = ctxt.create_buffer_simple(
        Some("clustered_aabbs"),
        clusters * 32, // ClusterAABB = 2 * vec4<f32>
        wgpu::BufferUsages::STORAGE,
    );
    let cells = ctxt.create_buffer_simple(
        Some("clustered_grid"),
        clusters * 8, // vec2<u32>
        wgpu::BufferUsages::STORAGE,
    );
    let index = ctxt.create_buffer_simple(
        Some("clustered_index_list"),
        clusters * max_per_cluster as u64 * 4, // u32
        wgpu::BufferUsages::STORAGE,
    );
    (aabbs, cells, index)
}

/// Allocates a `clustered_lights` storage buffer holding `capacity` lights.
fn alloc_lights(ctxt: &Context, capacity: u32) -> wgpu::Buffer {
    ctxt.create_buffer_simple(
        Some("clustered_lights"),
        (capacity as u64) * 64, // GpuLight = 64 bytes
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

#[cfg(test)]
mod tests {
    use crate::camera::OrbitCamera3d;
    use crate::color::Color;
    use crate::context::Context;
    use crate::light::Light;
    use crate::scene::SceneNode3d;
    use crate::test_gpu::{mean_luma, on_gpu};
    use glamx::Vec3;

    #[test]
    fn a_cluster_drops_the_lights_past_its_cap() {
        on_gpu(64, 64, async |surface| {
            if !Context::get().supports_clustered_lighting() {
                return;
            }
            let mut scene = SceneNode3d::empty();
            scene
                .add_cube(10.0, 0.2, 10.0)
                .set_color(Color::new(0.8, 0.8, 0.8, 1.0));
            for i in 0..24 {
                let a = i as f32 * std::f32::consts::TAU / 24.0;
                scene
                    .add_light(
                        Light::point(3.0)
                            .with_intensity(1.0)
                            .with_casts_shadows(false),
                    )
                    .set_position(Vec3::new(a.cos() * 2.0, 0.6, a.sin() * 2.0));
            }
            let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 6.0, 6.0), Vec3::ZERO);
            let window = surface.window_mut();
            assert_eq!(window.cluster_grid(), [16, 9, 24]);
            assert_eq!(window.max_lights_per_cluster(), 256);

            surface.render_3d(&mut scene, &mut camera).await;
            let all = mean_luma(surface);

            let window = surface.window_mut();
            window.set_cluster_grid([4, 4, 0]);
            window.set_max_lights_per_cluster(1);
            assert_eq!(window.cluster_grid(), [4, 4, 1]);
            surface.render_3d(&mut scene, &mut camera).await;
            let capped = mean_luma(surface);
            assert!(all > capped + 0.01, "{} {}", all, capped);
        });
    }
}
