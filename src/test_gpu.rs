//! Helpers for tests that render on a real GPU through an [`OffscreenSurface`].
//! A machine without an adapter skips them, as the shader-validity test does.

use crate::context::Context;
use crate::resource::MaterialManager2d;
use crate::window::OffscreenSurface;

async fn adapter_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .is_ok()
}

/// Runs `test` on a fresh `width × height` surface and fails on any wgpu
/// validation error it raised.
pub(crate) fn on_gpu(width: u32, height: u32, test: impl AsyncFnOnce(&mut OffscreenSurface)) {
    crate::pollster::block_on(async {
        if !adapter_available().await {
            eprintln!("no GPU adapter found, skipping");
            return;
        }
        let mut surface = OffscreenSurface::new(width, height).await;
        // Dropping the last window resets this manager, and a reset that first
        // creates it needs the texture manager the drop already cleared.
        MaterialManager2d::get_global_manager(|_| ());
        let scope = Context::get()
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        test(&mut surface).await;
        let err = scope.pop().await;
        assert!(err.is_none(), "wgpu validation error: {:?}", err);
    });
}

fn luma([r, g, b]: [u8; 3]) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

/// Mean luma of the last frame, `0.0..=1.0`.
pub(crate) fn mean_luma(surface: &OffscreenSurface) -> f32 {
    let image = surface.snap_image();
    let sum: f32 = image.pixels().map(|p| luma(p.0)).sum();
    sum / (image.width() * image.height()) as f32
}
