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
