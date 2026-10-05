//! Built-in geometries, shaders and effects.

/// Every built-in shader, linked by `build.rs` when kiss3d compiled: the root
/// module, the `@if` flags it was linked with, and the WGSL.
static LINKED: &[(&str, &[(&str, bool)], &str)] = include!(concat!(env!("OUT_DIR"), "/linked.rs"));

/// The WGSL `build.rs` linked for `root` with `features`, in any order. A
/// combination it did not link is a missing line in `build.rs`.
pub(crate) fn linked(root: &str, features: &[(&str, bool)]) -> &'static str {
    LINKED
        .iter()
        .find(|(r, f, _)| {
            *r == root && f.len() == features.len() && f.iter().all(|x| features.contains(x))
        })
        .map(|(_, _, wgsl)| *wgsl)
        .unwrap_or_else(|| panic!("{} {:?} is not linked; add it to build.rs", root, features))
}

#[cfg(test)]
mod tests {
    use naga::back::glsl;

    /// How many times `module` samples a depth texture without a comparison. GLSL
    /// reads one only through a shadow sampler, so naga writes such a sample and
    /// the device's compiler then rejects it.
    fn plain_depth_samples(module: &naga::Module) -> usize {
        let is_depth = |expressions: &naga::Arena<naga::Expression>, image| match expressions[image]
        {
            naga::Expression::GlobalVariable(var) => matches!(
                module.types[module.global_variables[var].ty].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Depth { .. },
                    ..
                }
            ),
            _ => false,
        };
        module
            .functions
            .iter()
            .map(|(_, function)| function)
            .chain(module.entry_points.iter().map(|entry| &entry.function))
            .map(|function| {
                function
                    .expressions
                    .iter()
                    .filter(|(_, expression)| match expression {
                        naga::Expression::ImageSample {
                            image,
                            depth_ref: None,
                            ..
                        } => is_depth(&function.expressions, *image),
                        _ => false,
                    })
                    .count()
            })
            .sum()
    }

    /// Every built-in shader written the way wgpu's GLES backend writes it on an
    /// Android device without Vulkan: GLSL ES 3.10, which lacks what only desktop
    /// GL has, such as asking a texture how many mips it holds.
    #[test]
    fn every_built_in_shader_is_glsl_es() {
        let mut options = glsl::Options::default();
        options.writer_flags |= glsl::WriterFlags::FORCE_POINT_SIZE;
        // Ray queries need a ray-tracing adapter, which no GLES device is.
        let gles_shaders = super::LINKED
            .iter()
            .filter(|(_, features, _)| !features.contains(&("hardware", true)));
        for (root, features, wgsl) in gles_shaders {
            let what = format!("{} {:?}", root, features);
            let module = naga::front::wgsl::parse_str(wgsl)
                .unwrap_or_else(|e| panic!("{}: {}", what, e.emit_to_string(wgsl)));
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: {}", what, e.emit_to_string(wgsl)));
            assert_eq!(
                plain_depth_samples(&module),
                0,
                "{} samples a depth texture without a comparison, which GLSL cannot",
                what
            );
            let (module, info) = naga::back::pipeline_constants::process_overrides(
                &module,
                &info,
                None,
                &Default::default(),
            )
            .unwrap_or_else(|e| panic!("{}: {}", what, e));
            for entry in &module.entry_points {
                let pipeline = glsl::PipelineOptions {
                    shader_stage: entry.stage,
                    entry_point: entry.name.clone(),
                    multiview: None,
                };
                let mut out = String::new();
                glsl::Writer::new(
                    &mut out,
                    &module,
                    &info,
                    &options,
                    &pipeline,
                    naga::proc::BoundsCheckPolicies::default(),
                )
                .and_then(|mut writer| writer.write())
                .unwrap_or_else(|e| panic!("{} `{}` is no GLSL ES 3.10: {}", what, entry.name, e));
            }
        }
    }
}

pub use self::aov::{
    AovKind, AovRenderer, DEPTH_AOV_FORMAT, NORMALS_AOV_FORMAT, SEGMENTATION_AOV_FORMAT,
};
pub use self::normals_material::{NormalsMaterial, NORMAL_FRAGMENT_SRC, NORMAL_VERTEX_SRC};
pub use self::object_material::{ObjectMaterial, OBJECT_FRAGMENT_SRC, OBJECT_VERTEX_SRC};
pub use self::uvs_material::{UvsMaterial, UVS_FRAGMENT_SRC, UVS_VERTEX_SRC};

pub use self::lit_material2d::{LitMaterial2d, LitMaterial2dGpuData, LitParams};
pub use self::object_material2d::ObjectMaterial2d;
pub use self::shadow::{
    ShadowMapper, DEFAULT_SHADOW_DEPTH_BIAS, DEFAULT_SHADOW_RASTER_BIAS, DEFAULT_SHADOW_VIEWS,
    MAX_SHADOW_VIEWS,
};
pub use self::skinned_material2d::{Bone2d, SkinVertex2d, SkinnedMesh2d, MAX_JOINTS_2D};

mod aov;
pub(crate) mod clustered;
pub mod deform;
mod normals_material;
mod object_material;
mod shadow;
mod uvs_material;

mod lit_material2d;
mod object_material2d;
mod skinned_material2d;
