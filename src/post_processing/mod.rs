//! Post-processing effects.

pub use crate::post_processing::cas::Cas;
pub use crate::post_processing::crt::Crt;
pub use crate::post_processing::fxaa::Fxaa;
pub use crate::post_processing::gi2d::{
    Gi2d, GiEmitter2d, GiOccluder2d, GiSegmentOccluder2d, MAX_EMITTERS, MAX_OCCLUDERS,
    MAX_SEGMENT_OCCLUDERS, MIN_SEGMENT_RADIUS,
};
pub use crate::post_processing::grayscales::Grayscales;
pub use crate::post_processing::hdr::{
    ColorGrading, HdrPipeline, HdrSettings, Tonemap, HDR_FORMAT, MAX_BLOOM_MIPS, OIT_ACCUM_FORMAT,
    OIT_REVEAL_FORMAT,
};
pub use crate::post_processing::loupe::{Loupe, LoupeCorner};
pub use crate::post_processing::oculus_stereo::OculusStereo;
pub use crate::post_processing::post_processing_effect::{
    FormatPipelines, PostProcessingContext, PostProcessingEffect,
};
pub(crate) use crate::post_processing::scene_depth::SceneDepth;
pub use crate::post_processing::sobel_edge_highlight::SobelEdgeHighlight;
pub use crate::post_processing::waves::Waves;

mod cas;
mod crt;
mod fxaa;
mod gi2d;
mod grayscales;
mod hdr;
mod loupe;
mod oculus_stereo;
pub mod post_processing_effect;
mod scene_depth;
mod sobel_edge_highlight;
mod waves;
