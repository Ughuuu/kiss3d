//! The kiss3d window.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use crate::builtin::ShadowMapper;
use crate::camera::{Camera3d, FixedView2d};
use crate::color::{Color, BLACK};
use crate::context::Context;
use crate::event::{Key, Modifiers, WindowEvent};
use crate::post_processing::{HdrPipeline, HdrSettings, Tonemap};
use crate::renderer::timings::GpuTimer;
use crate::renderer::{
    PointRenderer2d, PointRenderer3d, PolylineRenderer2d, PolylineRenderer3d, RayTracer,
    RenderTimings,
};
use crate::resource::{
    FramebufferManager, MaterialManager2d, MeshManager2d, RenderTarget, Texture, TextureManager,
};
use crate::scene::SceneNode3d;
use crate::text::TextRenderer;
use crate::window::canvas::CanvasSetup;
use crate::window::{Canvas, NumSamples};
use glamx::UVec2;
use image::{GenericImage, Pixel};
use winit::dpi::LogicalSize;
use winit::window::WindowAttributes;

#[cfg(feature = "egui")]
pub(super) use super::egui_integration::EguiContext;
#[cfg(feature = "recording")]
pub(super) use super::recording::RecordingState;
use super::window_cache::WindowCache;

pub(super) static DEFAULT_WIDTH: u32 = 800u32;
pub(super) static DEFAULT_HEIGHT: u32 = 600u32;

/// Default per-layer resolution of the rasterizer shadow atlas. 2048 balances
/// shadow sharpness against the per-frame cost of clearing/rasterizing the atlas
/// (a point light alone uses six faces); raise it with
/// [`Window::set_shadow_resolution`] for crisper shadows, lower it to save memory
/// and fill (the atlas is `resolution² × views`, see [`Window::set_max_shadow_views`]).
pub(super) static DEFAULT_SHADOW_RESOLUTION: u32 = 2048u32;

/// Structure representing a window and a 3D scene.
///
/// This is the main interface with the 3d engine.
/// The single-sample film copy a screen-reading 2D material samples.
pub(super) struct ScreenCopy2d {
    pub(super) view: wgpu::TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) generation: u64,
}

pub struct Window {
    pub(super) events: Rc<Receiver<WindowEvent>>,
    pub(super) unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    pub(super) ime_events: Rc<RefCell<Vec<crate::event::ImeEvent>>>,
    /// Made on the first frame a material asks; remade when the film resizes.
    pub(super) screen_2d: Option<ScreenCopy2d>,
    pub(super) ambient_intensity: f32,
    pub(super) ambient_color: Color,
    pub(super) fog: crate::light::Fog,
    pub(super) background: Color,
    pub(super) polyline_renderer_2d: PolylineRenderer2d,
    pub(super) point_renderer_2d: PointRenderer2d,
    pub(super) point_renderer: PointRenderer3d,
    pub(super) polyline_renderer: PolylineRenderer3d,
    pub(super) text_renderer: TextRenderer,
    pub(super) framebuffer_manager: FramebufferManager,
    /// Real-time shadow mapper for the rasterization pipeline.
    pub(super) shadow_mapper: ShadowMapper,
    /// HDR film + tonemap + bloom resolve stage for the rasterizer. The scene is
    /// rendered into its `Rgba16Float` target, then tonemapped into the LDR
    /// swapchain/offscreen output. See [`HdrPipeline`].
    pub(super) hdr: HdrPipeline,
    /// Equirectangular skybox drawn as the rasterizer's scene background.
    pub(super) skybox: crate::renderer::Skybox,
    /// Screen-space ambient occlusion (created on first enable).
    pub(super) ssao: Option<crate::renderer::Ssao>,
    pub(super) ssao_enabled: bool,
    /// Clustered forward+ lighting (created on first frame when the backend
    /// supports compute + fragment storage buffers; otherwise stays `None` and the
    /// object material falls back to the fixed 8-light path).
    pub(super) clustered: Option<crate::builtin::clustered::Clustered>,
    /// Clusters along screen X, screen Y and view depth for clustered lighting.
    pub(super) cluster_grid: [u32; 3],
    /// Lights one cluster records.
    pub(super) max_lights_per_cluster: u32,
    /// Reflection probes (localized parallax-corrected environment maps). Created
    /// on first probe registration; `None` means no probes.
    pub(super) reflection_probes: Option<crate::renderer::ReflectionProbes>,
    /// Reusable GPU targets for runtime probe capture (created on first capture).
    pub(super) probe_capture: Option<crate::renderer::ProbeCapture>,
    /// Probe indices queued for a runtime scene capture next frame.
    pub(super) pending_probe_captures: Vec<usize>,
    /// Cube-face size of a probe capture and width of every probe map.
    pub(super) reflection_probe_size: u32,
    /// Screen-space reflections (created on first enable when the backend supports
    /// it; stays `None` / inactive on WebGL2).
    pub(super) ssr: Option<crate::renderer::Ssr>,
    pub(super) ssr_enabled: bool,
    /// Depth of field (created on first enable). Shares the geometry prepass with
    /// SSAO/SSR for the view-position depth it blurs by.
    pub(super) dof: Option<crate::renderer::Dof>,
    pub(super) dof_enabled: bool,
    /// Refraction background snapshot for glass (created on first use). Built from
    /// the resolved scene each frame that contains refractive surfaces.
    pub(super) transmission: Option<crate::renderer::Transmission>,
    pub(super) transmission_enabled: bool,
    /// Single-sample OIT targets for the planar-reflector capture pass, so
    /// transparent surfaces appear in mirrors. Created on first use.
    pub(super) reflector_oit: Option<crate::renderer::ReflectorOit>,
    /// The refraction snapshot of a reflector whose target is not viewport-sized.
    pub(super) reflector_transmission: Option<crate::renderer::Transmission>,
    pub(super) post_process_render_target: RenderTarget,
    /// Second LDR target, paired with `post_process_render_target` as ping-pong
    /// buffers when chaining more than one post-processing effect: each effect reads
    /// one and writes the other, and the last writes the final frame.
    pub(super) post_process_render_target_b: RenderTarget,
    /// The same pair at [`HDR_FORMAT`](crate::post_processing::HDR_FORMAT), for a
    /// chain that runs on the film before bloom and the tonemap rather than on the
    /// LDR image after them. Made the first frame such a chain is passed, so a run
    /// that never passes one allocates neither.
    pub(super) film_render_targets: Option<(RenderTarget, RenderTarget)>,
    /// The multisampled scene depth resolved for a chain whose effect reads it.
    /// Made the first frame one asks.
    pub(super) scene_depth: Option<crate::post_processing::SceneDepth>,
    /// Offscreen render target used when the window is hidden, so `snap` and
    /// recording work without a presentable surface. Created on first use.
    pub(super) offscreen_output_target: Option<RenderTarget>,
    /// Renderer for auxiliary outputs (depth, normals, segmentation). Created
    /// on first use of an AOV-producing method.
    pub(super) aov_renderer: Option<crate::builtin::AovRenderer>,
    /// Whether the window is hidden. Hidden windows render offscreen.
    pub(super) hidden: bool,
    pub(super) should_close: bool,
    /// `true` until the first surface texture has been successfully acquired.
    /// While set, frame acquisition retries (pumping window events) instead of
    /// skipping, so a freshly created window reliably renders its first frame.
    pub(super) first_frame: bool,
    pub(super) close_key: Option<Key>,
    pub(super) close_modifiers: Option<Modifiers>,
    /// Per-step timings of the most recently rendered frame, for the active
    /// renderer. `None` until the first frame. See [`Window::render_timings`].
    pub(super) last_timings: Option<RenderTimings>,
    /// Instant the previous frame started, to derive the wall-clock frame-to-frame
    /// period ([`RenderTimings::frame_wall`]). `None` until the first frame.
    pub(super) last_frame_instant: Option<web_time::Instant>,
    /// GPU timestamp-query timer (disabled if the device lacks `TIMESTAMP_QUERY`).
    pub(super) gpu_timer: GpuTimer,
    #[cfg(feature = "egui")]
    pub(super) egui_context: EguiContext,
    pub(super) canvas: Canvas,
    #[cfg(feature = "recording")]
    pub(super) recording: Option<RecordingState>,
    // NOTE: the boolean is used to avoid borrowcheker issues with
    //       the event-based switching.
    #[cfg(feature = "rt_switcher")]
    pub(super) raytracer: (Option<RayTracer>, bool),
}

impl Window {
    /// Indicates whether this window should be closed.
    #[inline]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    /// The window width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.canvas.size().0
    }

    /// The window height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.canvas.size().1
    }

    /// The size of the window.
    #[inline]
    pub fn size(&self) -> UVec2 {
        let (w, h) = self.canvas.size();
        UVec2::new(w, h)
    }

    /// The current number of MSAA samples (`1` means multisampling is disabled).
    #[inline]
    pub fn samples(&self) -> u32 {
        self.canvas.sample_count()
    }

    /// Sets the number of MSAA samples used for rendering, recreating the render
    /// targets to match. The change takes effect on the next rendered frame.
    ///
    /// This is the runtime equivalent of [`CanvasSetup::samples`]; it lets you toggle
    /// or change anti-aliasing after the window has been created.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::{Window, NumSamples};
    /// # async fn f(window: &mut Window) {
    /// window.set_samples(NumSamples::Four);
    /// # }
    /// ```
    #[inline]
    pub fn set_samples(&mut self, samples: NumSamples) {
        self.canvas.set_samples(samples);
    }

    /// Whether vsync is currently enabled (vsync is on by default).
    #[inline]
    pub fn vsync(&self) -> bool {
        self.canvas.vsync()
    }

    /// Enables or disables vsync at runtime, reconfiguring the surface present mode
    /// (`AutoVsync` ↔ `AutoNoVsync`); takes effect on the next presented frame.
    ///
    /// With vsync **off**, frames present uncapped (as fast as the GPU produces
    /// them) instead of being paced to the display refresh — useful for measuring
    /// GPU-bound throughput via the wall-clock frame time. No effect on a hidden /
    /// offscreen window (which has no presentable surface).
    #[inline]
    pub fn set_vsync(&mut self, enabled: bool) {
        self.canvas.set_vsync(enabled);
    }

    /// Gets a reference to the underlying canvas.
    ///
    /// This provides access to low-level rendering features like:
    /// - Getting the current surface texture for custom rendering
    /// - Getting the depth texture view
    /// - Presenting frames manually
    #[inline]
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }

    /// Gets a mutable reference to the underlying canvas.
    #[inline]
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    #[cfg(feature = "rt_switcher")]
    pub fn raytracer_mut(&mut self) -> Option<&mut RayTracer> {
        self.raytracer.0.as_mut()
    }

    #[cfg(feature = "rt_switcher")]
    pub fn set_raytracer(&mut self, rt: Option<RayTracer>) {
        self.raytracer.0 = rt;
    }

    /// Timings of the most recently rendered frame.
    ///
    /// Returns `None` until the first frame has been rendered. The timings are
    /// for whichever renderer ran last (rasterizer or path tracer): per-pass GPU
    /// execution times from timestamp queries (when the device supports them),
    /// plus CPU wall-clock for the submit/present calls and the whole frame. See
    /// [`RenderTimings`]. The built-in inspector displays them.
    #[inline]
    pub fn render_timings(&self) -> Option<&RenderTimings> {
        self.last_timings.as_ref()
    }

    /// Renders one frame of a 3D scene with the GPU path tracer.
    ///
    /// This is the ray-traced counterpart of [`render_3d`](Self::render_3d). It
    /// bypasses the rasterizer and instead path-traces the scene, progressively
    /// accumulating samples for a photorealistic image. Keep the same
    /// [`RayTracer`] across frames so accumulation can converge; it restarts
    /// automatically when the camera moves, the window is resized, or the scene
    /// changes.
    ///
    /// If the path tracer is disabled (see [`RayTracer::set_enabled`]), this
    /// renders the scene with the rasterizer instead, so the same render call
    /// can switch between the two renderers without restructuring the loop.
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    /// use kiss3d::renderer::RayTracer;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("Ray tracing").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///     let mut raytracer = RayTracer::new();
    ///
    ///     while window.raytrace_3d(&mut scene, &mut camera, &mut raytracer).await {}
    /// }
    /// ```
    pub async fn raytrace_3d(
        &mut self,
        scene: &mut SceneNode3d,
        camera: &mut dyn Camera3d,
        raytracer: &mut RayTracer,
    ) -> bool {
        let mut default_cam2 = FixedView2d::default();
        self.handle_events(camera, &mut default_cam2);
        self.raytrace_3d_frame(scene, camera, raytracer).await
    }

    /// Sets the window title.
    ///
    /// # Arguments
    /// * `title` - The new title for the window
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// let mut window = Window::new("Initial Title").await;
    /// window.set_title("New Title");
    /// # }
    /// ```
    pub fn set_title(&mut self, title: &str) {
        self.canvas.set_title(title)
    }

    /// Set the window icon. On wasm this does nothing.
    ///
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_icon(image::open("foo.ico").unwrap());
    /// # }
    /// ```
    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        self.canvas.set_icon(icon)
    }

    /// Sets the cursor grabbing behaviour.
    ///
    /// If cursor grabbing is enabled, the cursor is prevented from leaving the window.
    ///
    /// # Arguments
    /// * `grab` - `true` to enable cursor grabbing, `false` to disable it
    ///
    /// # Platform-specific
    /// Does nothing on web platforms.
    pub fn set_cursor_grab(&self, grab: bool) {
        self.canvas.set_cursor_grab(grab);
    }

    /// Enters or leaves borderless fullscreen on the current monitor.
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.canvas.set_fullscreen(fullscreen);
    }

    /// Enters exclusive fullscreen, the current monitor's largest video mode
    /// at its highest refresh rate, or leaves fullscreen entirely: passing
    /// `false` also leaves the borderless fullscreen of [`Self::set_fullscreen`].
    ///
    /// # Platform-specific
    /// Where the platform offers no video modes, as on the web, this is
    /// borderless fullscreen.
    pub fn set_exclusive_fullscreen(&self, exclusive: bool) {
        self.canvas.set_exclusive_fullscreen(exclusive);
    }

    /// Maximizes the window, or restores it.
    pub fn set_maximized(&self, maximized: bool) {
        self.canvas.set_maximized(maximized);
    }

    /// Whether the window is currently maximized.
    pub fn is_maximized(&self) -> bool {
        self.canvas.is_maximized()
    }

    /// Shows or hides the platform's on-screen keyboard.
    ///
    /// # Platform-specific
    /// Android and iOS summon the system keyboard; typed text then arrives as
    /// ordinary `Char`/`Key` events. Desktop and web are no-ops.
    pub fn set_keyboard_visible(&self, visible: bool) {
        self.canvas.set_keyboard_visible(visible);
    }

    /// Whether the window is currently fullscreen.
    pub fn is_fullscreen(&self) -> bool {
        self.canvas.is_fullscreen()
    }

    /// Files dropped onto the window since the last call, in drop order.
    ///
    /// # Platform-specific
    /// Always empty on the web: browsers deliver file drops to the page, not
    /// the canvas.
    pub fn dropped_files(&self) -> Vec<std::path::PathBuf> {
        self.canvas.take_dropped_files()
    }

    /// This frame's composed text, preedits and commits in order; empty until
    /// [`Self::set_ime_allowed`]. egui's fields hear the same events on their own.
    pub fn ime_events(&self) -> Vec<crate::event::ImeEvent> {
        self.ime_events.borrow().clone()
    }

    /// Let the platform compose text through its input method.
    pub fn set_ime_allowed(&self, allowed: bool) {
        self.canvas.set_ime_allowed(allowed);
    }

    /// `[left, top, right, bottom]` insets in pixels; zero everywhere but iOS.
    pub fn safe_area(&self) -> [f32; 4] {
        self.canvas.safe_area()
    }

    /// How many pixels of the window the on-screen keyboard covers, from the
    /// bottom; zero with it down, and zero everywhere but Android and iOS.
    pub fn keyboard_height(&self) -> f32 {
        self.canvas.keyboard_height()
    }

    /// How much larger than standard the reader asked their text to be: iOS
    /// Dynamic Type today, and 1.0 where the platform has not been asked.
    pub fn text_scale(&self) -> f32 {
        self.canvas.text_scale()
    }

    /// Sets the cursor position in window coordinates.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in pixels from the left edge of the window
    /// * `y` - The y-coordinate in pixels from the top edge of the window
    #[inline]
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    /// Controls the cursor visibility.
    ///
    /// # Arguments
    /// * `hide` - `true` to hide the cursor, `false` to show it
    #[inline]
    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
    }

    /// Closes the window.
    ///
    /// After calling this method, [`render()`](Self::render) will return `false` on the next frame,
    /// allowing the render loop to exit gracefully.
    #[inline]
    pub fn close(&mut self) {
        self.should_close = true;
    }

    /// Hides the window without closing it.
    ///
    /// Use [`show()`](Self::show) to make it visible again.
    /// The window continues to exist and can be shown again later.
    #[inline]
    pub fn hide(&mut self) {
        self.hidden = true;
        self.canvas.hide()
    }

    /// Makes the window visible.
    ///
    /// Use [`hide()`](Self::hide) to hide it again.
    #[inline]
    pub fn show(&mut self) {
        self.hidden = false;
        self.canvas.show()
    }

    /// Sets the background color for the window.
    ///
    /// # Arguments
    /// * `r` - Red component (0.0 to 1.0)
    /// * `g` - Green component (0.0 to 1.0)
    /// * `b` - Blue component (0.0 to 1.0)
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// use kiss3d::color::DARK_BLUE;
    /// let mut window = Window::new("Example").await;
    /// window.set_background_color(DARK_BLUE);
    /// # }
    /// ```
    #[inline]
    pub fn set_background_color(&mut self, color: Color) {
        self.background = color;
    }

    /// Loads a texture from a file and returns a reference to it.
    ///
    /// The texture is managed by the global texture manager and will be reused
    /// if loaded again with the same name.
    ///
    /// # Arguments
    /// * `path` - Path to the texture file
    /// * `name` - A unique name to identify this texture
    ///
    /// # Returns
    /// A reference-counted texture that can be applied to scene objects
    pub fn add_texture(&mut self, path: &Path, name: &str) -> Arc<Texture> {
        TextureManager::get_global_manager(|tm| tm.add(path, name))
    }

    /// Returns the DPI scale factor of the screen.
    ///
    /// This is the ratio between physical pixels and logical pixels.
    /// On high-DPI displays (like Retina displays), this will be greater than 1.0.
    ///
    /// # Returns
    /// The scale factor (e.g., 1.0 for standard displays, 2.0 for Retina displays)
    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    /// Sets the ambient light intensity for the scene.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::light::Light;
    /// # use glamx::Vec3;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // Set global ambient lighting intensity
    /// window.set_ambient(0.3);
    /// # }
    /// ```
    ///
    /// Note: Individual lights should be added to the scene tree using
    /// `SceneNode3d::add_point_light()`, `add_directional_light()`, or `add_spot_light()`.
    pub fn set_ambient(&mut self, ambient: f32) {
        self.ambient_intensity = ambient;
    }

    /// Returns the current ambient lighting intensity.
    pub fn ambient(&self) -> f32 {
        self.ambient_intensity
    }

    /// Sets the global ambient light color.
    ///
    /// The ambient term added to every surface is `ambient_color * ambient *
    /// albedo * ao`, so the color tints the fill light while
    /// [`set_ambient`](Self::set_ambient) controls its brightness. Defaults to
    /// white.
    pub fn set_ambient_color(&mut self, color: Color) {
        self.ambient_color = color;
    }

    /// Returns the current ambient light color.
    pub fn ambient_color(&self) -> Color {
        self.ambient_color
    }

    /// Sets the distance fog applied to the rasterized scene.
    ///
    /// Pass a [`Fog`](crate::light::Fog) describing the falloff curve and color,
    /// or [`Fog::default()`](crate::light::Fog::default) (mode [`FogMode::Off`](crate::light::FogMode::Off))
    /// to disable fog. Fog blends shaded fragments toward the fog color by their
    /// view-space distance from the camera.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_fog(Fog::exponential(Color::new(0.6, 0.7, 0.8, 1.0), 0.02));
    /// # }
    /// ```
    pub fn set_fog(&mut self, fog: crate::light::Fog) {
        self.fog = fog;
    }

    /// Returns the current distance fog settings.
    pub fn fog(&self) -> crate::light::Fog {
        self.fog
    }

    /// Mutable access to the distance fog settings.
    pub fn fog_mut(&mut self) -> &mut crate::light::Fog {
        &mut self.fog
    }

    /// Sets the rasterizer skybox from an equirectangular image file.
    ///
    /// Accepts HDR (`.hdr`), EXR, or any format the `image` crate decodes. The
    /// image is drawn as the scene background and uses the same direction→UV
    /// mapping as the path tracer's HDRI, so the two backends show the same sky.
    /// Returns `false` if the file cannot be decoded.
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # use std::path::Path;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// window.set_skybox_from_file(Path::new("assets/sky.hdr"));
    /// # }
    /// ```
    pub fn set_skybox_from_file(&mut self, path: &Path) -> bool {
        self.skybox.set_from_file(path)
    }

    /// Sets the rasterizer skybox from an encoded equirectangular image held in
    /// memory (e.g. `include_bytes!`-embedded for wasm). Returns `false` if the
    /// bytes can't be decoded.
    pub fn set_skybox_from_memory(&mut self, bytes: &[u8]) -> bool {
        match image::load_from_memory(bytes) {
            Ok(img) => {
                self.skybox.set_image(&img);
                true
            }
            Err(_) => false,
        }
    }

    /// Sets the rasterizer skybox from an already-decoded equirectangular image.
    pub fn set_skybox_image(&mut self, image: &image::DynamicImage) {
        self.skybox.set_image(image);
    }

    /// Sets the skybox Y-axis rotation (radians) and luminance multiplier.
    pub fn set_skybox_orientation(&mut self, rotation_radians: f32, intensity: f32) {
        self.skybox.set_orientation(rotation_radians, intensity);
    }

    /// Lights the rasterized scene with `image` (an equirectangular map) instead of
    /// the drawn skybox, which stays the background; it turns with the skybox's
    /// rotation and lights even with no skybox drawn. `None`, the default, lights
    /// with the drawn skybox. The path tracer keeps the skybox for both.
    pub fn set_sky_lighting_image(&mut self, image: Option<&image::DynamicImage>) {
        self.skybox.set_lighting_image(image);
    }

    /// Whether a lighting image apart from the drawn skybox is set.
    pub fn has_sky_lighting_image(&self) -> bool {
        self.skybox.has_lighting_image()
    }

    /// Sets the image-based-lighting multiplier apart from the drawn skybox's
    /// (clamped to `>= 0`). `None`, the default, follows the intensity given to
    /// [`set_skybox_orientation`](Self::set_skybox_orientation).
    pub fn set_sky_lighting_intensity(&mut self, intensity: Option<f32>) {
        self.skybox.set_lighting_intensity(intensity);
    }

    /// Returns the image-based-lighting multiplier in use: the one set apart, else
    /// the skybox's.
    pub fn sky_lighting_intensity(&self) -> f32 {
        self.skybox.lighting_intensity()
    }

    /// Removes the skybox, so subsequent frames render the plain background color.
    pub fn clear_skybox(&mut self) {
        self.skybox.clear();
    }

    /// Whether a skybox environment is currently set.
    pub fn has_skybox(&self) -> bool {
        self.skybox.is_set()
    }

    /// Enables or disables screen-space ambient occlusion (SSAO).
    ///
    /// When enabled, a depth/view-position prepass plus a hemisphere-sampling
    /// pass darken the ambient lighting in creases and contact areas. Adds a
    /// geometry prepass per frame. Disabled by default.
    pub fn set_ssao_enabled(&mut self, enabled: bool) {
        self.ssao_enabled = enabled;
    }

    /// Whether SSAO is enabled.
    pub fn ssao_enabled(&self) -> bool {
        self.ssao_enabled
    }

    /// Mutable access to the SSAO settings (radius, bias, intensity, power),
    /// creating the SSAO state if needed.
    pub fn ssao_settings_mut(&mut self) -> &mut crate::renderer::SsaoSettings {
        let (w, h) = self.canvas.size();
        self.ssao
            .get_or_insert_with(|| crate::renderer::Ssao::new(w, h))
            .settings_mut()
    }

    /// Registers a reflection probe and returns its index (up to
    /// [`MAX_PROBES`](crate::renderer::MAX_PROBES)), or `None` if full.
    ///
    /// A probe is a localized, parallax-corrected environment map: reflective
    /// surfaces inside its influence box sample it (with box-projected parallax)
    /// instead of the global skybox. Fill its content with
    /// [`set_reflection_probe_image`](Self::set_reflection_probe_image) (a baked
    /// HDR) or [`capture_reflection_probe`](Self::capture_reflection_probe) (a live
    /// scene capture).
    pub fn add_reflection_probe(
        &mut self,
        probe: crate::renderer::ReflectionProbe,
    ) -> Option<usize> {
        let size = self.reflection_probe_size;
        self.reflection_probes
            .get_or_insert_with(|| crate::renderer::ReflectionProbes::with_size(size))
            .add(probe)
    }

    /// Sets the reflection-probe resolution, shared by every probe: a runtime
    /// capture renders cube faces `size` pixels square, and each probe map is `size`
    /// wide and half as tall. Clamped to `2..=4096`. A change reallocates the probe
    /// maps black: set the baked images again and re-capture the live probes. The
    /// default is 256.
    pub fn set_reflection_probe_size(&mut self, size: u32) {
        self.reflection_probe_size = size.clamp(2, 4096);
        if let Some(probes) = self.reflection_probes.as_mut() {
            probes.set_size(self.reflection_probe_size);
        }
    }

    /// Returns the reflection-probe resolution. The default is 256.
    pub fn reflection_probe_size(&self) -> u32 {
        self.reflection_probe_size
    }

    /// Fills reflection probe `idx` from a baked equirectangular HDR image.
    pub fn set_reflection_probe_image(&mut self, idx: usize, img: &image::DynamicImage) {
        if let Some(probes) = self.reflection_probes.as_mut() {
            probes.set_image(idx, img);
        }
    }

    /// Mutable access to a registered probe's placement (to move/resize it; call
    /// before re-capturing).
    pub fn reflection_probe_mut(
        &mut self,
        idx: usize,
    ) -> Option<&mut crate::renderer::ReflectionProbe> {
        self.reflection_probes
            .as_mut()
            .and_then(|p| p.probe_mut(idx))
    }

    /// Queues a runtime scene capture of reflection probe `idx`: next frame the
    /// scene is rendered into the probe's six cube faces (from its center) and
    /// reprojected into its environment map, so it reflects live geometry. The
    /// captured frame omits the probe being captured (it is not yet populated) and
    /// uses the fixed-light path; re-call after the scene changes to refresh it.
    pub fn capture_reflection_probe(&mut self, idx: usize) {
        if self
            .reflection_probes
            .as_ref()
            .is_some_and(|p| idx < p.len())
            && !self.pending_probe_captures.contains(&idx)
        {
            self.pending_probe_captures.push(idx);
        }
    }

    /// Enables or disables screen-space reflections (SSR).
    ///
    /// When enabled (and supported by the backend — native/WebGPU; not WebGL2), a
    /// geometry G-buffer prepass plus a screen-space ray-march add sharp on-screen
    /// reflections to glossy/metallic surfaces, falling back to reflection probes
    /// and the skybox where the screen has no data. Disabled by default.
    pub fn set_ssr_enabled(&mut self, enabled: bool) {
        self.ssr_enabled = enabled;
    }

    /// Whether SSR is enabled (may still be inactive if the backend lacks support).
    pub fn ssr_enabled(&self) -> bool {
        self.ssr_enabled
    }

    /// Mutable access to the SSR settings, creating the SSR state if needed.
    pub fn ssr_settings_mut(&mut self) -> &mut crate::renderer::SsrSettings {
        let (w, h) = self.canvas.size();
        self.ssr
            .get_or_insert_with(|| crate::renderer::Ssr::new(w, h))
            .settings_mut()
    }

    /// Enables or disables depth of field (DoF).
    ///
    /// When enabled, the geometry G-buffer prepass (shared with SSAO/SSR) feeds a
    /// thin-lens blur that keeps surfaces near the focal plane sharp and blurs the
    /// rest, with the amount controlled by [`DofSettings`](crate::renderer::DofSettings)
    /// (focal distance, aperture, etc). Runs after SSR and before tonemapping.
    /// Disabled by default.
    pub fn set_dof_enabled(&mut self, enabled: bool) {
        self.dof_enabled = enabled;
    }

    /// Whether depth of field is enabled.
    pub fn dof_enabled(&self) -> bool {
        self.dof_enabled
    }

    /// Enables or disables refractive transmission (glass).
    ///
    /// When enabled (the default), objects with a non-zero
    /// [`set_transmission`](crate::scene::Object3d::set_transmission) refract the
    /// rendered scene behind them: after the opaque pass is resolved, its color is
    /// snapshotted (with a blurred mip chain for frosted glass) and the glass
    /// objects are drawn sampling it, bent by their IOR/thickness and tinted by the
    /// volume attenuation. Disable it to skip the extra passes (glass then renders
    /// as a plain opaque PBR surface). Has no effect on the path tracer.
    pub fn set_transmission_enabled(&mut self, enabled: bool) {
        self.transmission_enabled = enabled;
    }

    /// Whether refractive transmission (glass) is enabled.
    pub fn transmission_enabled(&self) -> bool {
        self.transmission_enabled
    }

    /// Mutable access to the refractive-transmission settings (e.g. the roughness
    /// blur quality), creating the transmission state if needed.
    pub fn transmission_settings_mut(&mut self) -> &mut crate::renderer::TransmissionSettings {
        let (w, h) = self.canvas.size();
        self.transmission
            .get_or_insert_with(|| crate::renderer::Transmission::new(w, h))
            .settings_mut()
    }

    /// Mutable access to the depth-of-field settings, creating the DoF state if
    /// needed.
    pub fn dof_settings_mut(&mut self) -> &mut crate::renderer::DofSettings {
        let (w, h) = self.canvas.size();
        self.dof
            .get_or_insert_with(|| crate::renderer::Dof::new(w, h))
            .settings_mut()
    }

    /// Enables or disables real-time shadow mapping for the rasterizer.
    ///
    /// Shadows are enabled by default. When disabled, no shadow pre-pass runs and
    /// every light illuminates surfaces as if unobstructed. This has no effect on
    /// the path tracer, which always computes ray-traced shadows.
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadow_mapper.set_enabled(enabled);
    }

    /// Returns whether real-time shadow mapping is enabled for the rasterizer.
    pub fn shadows_enabled(&self) -> bool {
        self.shadow_mapper.is_enabled()
    }

    /// Sets the per-layer resolution of the shadow atlas (square), reallocating it.
    ///
    /// Higher values yield crisper shadows at the cost of memory and fill rate.
    /// The default is 2048.
    pub fn set_shadow_resolution(&mut self, resolution: u32) {
        self.shadow_mapper.set_resolution(resolution);
    }

    /// Returns the current per-layer shadow atlas resolution.
    pub fn shadow_resolution(&self) -> u32 {
        self.shadow_mapper.resolution()
    }

    /// Sets the rasterizer shadow-edge softness (PCF blur).
    ///
    /// `1.0` (the default) is the standard penumbra; larger values blur the
    /// shadow edges more, `0.0` gives hard edges. Has no effect on the path
    /// tracer, whose shadow softness comes from each light's `radius`.
    pub fn set_shadow_softness(&mut self, softness: f32) {
        self.shadow_mapper.set_softness(softness);
    }

    /// Returns the current rasterizer shadow-edge softness (PCF blur).
    pub fn shadow_softness(&self) -> f32 {
        self.shadow_mapper.softness()
    }

    /// Caps how far directional shadows reach, in world units along the view:
    /// the cascades cover the camera's near plane to `min(far plane, distance)`.
    /// The default, `f32::INFINITY`, uses the camera's far plane.
    pub fn set_shadow_distance(&mut self, distance: f32) {
        self.shadow_mapper.set_shadow_distance(distance);
    }

    /// Returns how far directional shadows reach. The default is `f32::INFINITY`.
    pub fn shadow_distance(&self) -> f32 {
        self.shadow_mapper.shadow_distance()
    }

    /// Sets how many cascades a directional light splits its shadow range into,
    /// clamped to `1..=4`. Each cascade takes one atlas view. The default is 4.
    pub fn set_shadow_cascades(&mut self, cascades: u32) {
        self.shadow_mapper.set_num_cascades(cascades);
    }

    /// Returns the number of directional shadow cascades. The default is 4.
    pub fn shadow_cascades(&self) -> u32 {
        self.shadow_mapper.num_cascades()
    }

    /// Sets the far view distance of the first, sharpest directional cascade: it
    /// covers the camera's near plane to `distance` (at least 0.01). The default is 12.
    pub fn set_shadow_first_cascade_distance(&mut self, distance: f32) {
        self.shadow_mapper.set_first_cascade_far_bound(distance);
    }

    /// Returns the far view distance of the first directional cascade. The default is 12.
    pub fn shadow_first_cascade_distance(&self) -> f32 {
        self.shadow_mapper.first_cascade_far_bound()
    }

    /// Sets the depth bias the lighting shader applies when comparing against the
    /// shadow map (at least 0). Raise it to cure acne, lower it to keep contact
    /// shadows attached. The default is 0.0012.
    pub fn set_shadow_depth_bias(&mut self, bias: f32) {
        self.shadow_mapper.set_depth_bias(bias);
    }

    /// Returns the shadow comparison depth bias. The default is 0.0012.
    pub fn shadow_depth_bias(&self) -> f32 {
        self.shadow_mapper.depth_bias()
    }

    /// Sets the rasterizer depth bias of the shadow depth pass: `constant` in
    /// depth-buffer units plus `slope_scale` times the polygon's depth slope. A
    /// change rebuilds the shadow depth pipelines. The default is `(1, 1.75)`.
    pub fn set_shadow_raster_bias(&mut self, constant: i32, slope_scale: f32) {
        self.shadow_mapper.set_raster_bias(constant, slope_scale);
    }

    /// Returns the shadow rasterizer depth bias `(constant, slope_scale)`. The
    /// default is `(1, 1.75)`.
    pub fn shadow_raster_bias(&self) -> (i32, f32) {
        self.shadow_mapper.raster_bias()
    }

    /// Sets the shadow view budget: atlas layers shared by every shadow-casting
    /// light, clamped to `1..=MAX_SHADOW_VIEWS` (64), reallocating the atlas. A
    /// spot light takes one view, a directional light one per cascade, a point
    /// light six; lights past the budget light without shadows. Memory is
    /// `resolution² × views × 8` bytes (depth plus colored transmittance). The
    /// default is 16.
    pub fn set_max_shadow_views(&mut self, views: u32) {
        self.shadow_mapper.set_max_views(views);
    }

    /// Returns the shadow view budget. The default is 16.
    pub fn max_shadow_views(&self) -> u32 {
        self.shadow_mapper.max_views()
    }

    /// Sets the clustered-lighting grid: clusters along screen X, screen Y and
    /// view depth, each clamped to `1..=128`. Finer grids cull the many-light tier
    /// tighter at the cost of memory and culling work. The default is `[16, 9, 24]`.
    /// Clustered lighting runs where compute shaders do (not WebGL2).
    pub fn set_cluster_grid(&mut self, grid: [u32; 3]) {
        self.cluster_grid = grid.map(|n| n.clamp(1, crate::builtin::clustered::MAX_GRID_AXIS));
    }

    /// Returns the clustered-lighting grid. The default is `[16, 9, 24]`.
    pub fn cluster_grid(&self) -> [u32; 3] {
        self.cluster_grid
    }

    /// Sets how many lights one cluster records (at least 1); lights past it in a
    /// dense cluster are dropped. The light-index list holds `clusters × this`
    /// entries, so the renderer lowers it to fit the device's largest storage
    /// binding. The default is 256.
    pub fn set_max_lights_per_cluster(&mut self, lights: u32) {
        self.max_lights_per_cluster = lights.max(1);
    }

    /// Returns how many lights one cluster records. The default is 256.
    pub fn max_lights_per_cluster(&self) -> u32 {
        self.max_lights_per_cluster
    }

    #[cfg(test)]
    pub(crate) fn shadow_mapper(&self) -> &ShadowMapper {
        &self.shadow_mapper
    }

    /// The current HDR finishing settings (exposure, tonemap operator, bloom).
    ///
    /// The rasterizer renders into an HDR film and resolves it with these
    /// settings; see [`HdrSettings`] and [`HdrPipeline`].
    pub fn hdr_settings(&self) -> &HdrSettings {
        self.hdr.settings()
    }

    /// Mutable access to the HDR finishing settings.
    ///
    /// ```no_run
    /// # use kiss3d::window::Window;
    /// # use kiss3d::post_processing::Tonemap;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// let s = window.hdr_settings_mut();
    /// s.exposure = 1.5;
    /// s.tonemap = Tonemap::Aces;
    /// s.bloom_enabled = true;
    /// # }
    /// ```
    pub fn hdr_settings_mut(&mut self) -> &mut HdrSettings {
        self.hdr.settings_mut()
    }

    /// Sets the exposure multiplier applied before tonemapping (`1.0` is neutral).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.hdr.settings_mut().exposure = exposure;
    }

    /// Sets the exposure from a physically-based [`Exposure`](crate::camera::Exposure).
    ///
    /// Applies to both the rasterizer and the path tracer (they share the HDR
    /// resolve exposure).
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// // f/8, 1/125 s, ISO 100
    /// window.set_exposure_value(Exposure::from_physical(8.0, 1.0 / 125.0, 100.0));
    /// # }
    /// ```
    pub fn set_exposure_value(&mut self, exposure: crate::camera::Exposure) {
        self.hdr.settings_mut().exposure = exposure.exposure();
    }

    /// Selects the tonemapping operator used by the HDR resolve pass.
    pub fn set_tonemap(&mut self, tonemap: Tonemap) {
        self.hdr.settings_mut().tonemap = tonemap;
    }

    /// Enables or disables bloom.
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.hdr.settings_mut().bloom_enabled = enabled;
    }

    /// Compile the finishing passes the current settings will draw with.
    ///
    /// Bloom and auto-exposure are built on demand, so call this after writing
    /// the settings: a frame that waits for a shader compiler is a frame
    /// somebody sees, and this puts that wait where the settings changed.
    pub fn prepare_post(&self) {
        self.hdr.prepare();
    }

    /// Sets the bloom brightness threshold and additive intensity.
    pub fn set_bloom(&mut self, threshold: f32, intensity: f32) {
        let s = self.hdr.settings_mut();
        s.bloom_threshold = threshold;
        s.bloom_intensity = intensity;
    }

    /// Rebinds the key to close the window.
    /// Set to None to disable.
    pub fn rebind_close_key(&mut self, new_close_key: Option<Key>) {
        self.close_key = new_close_key;
    }

    /// Rebinds the modifiers to close the window.
    /// Set to None make it work with any modifiers.
    pub fn rebind_close_modifiers(&mut self, new_close_modifiers: Option<Modifiers>) {
        self.close_modifiers = new_close_modifiers;
    }

    /// Returns the current key to close the window.
    pub fn close_key(&self) -> Option<Key> {
        self.close_key
    }

    /// Returns the current modifiers to close the window.
    pub fn close_modifiers(&self) -> Option<Modifiers> {
        self.close_modifiers
    }

    /// Creates a new hidden window.
    ///
    /// The window is created but not displayed. Use [`show()`](Self::show) to make it visible.
    /// The default size is 800x600 pixels.
    ///
    /// While hidden, the window renders off-screen instead of to its surface,
    /// so [`snap`](Self::snap), [`snap_image`](Self::snap_image) and recording
    /// work without ever displaying anything — this is how kiss3d does
    /// offscreen rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_hidden(title: &str) -> Window {
        Window::do_new(title, true, DEFAULT_WIDTH, DEFAULT_HEIGHT, None).await
    }

    /// Creates a new hidden window with custom dimensions.
    ///
    /// The window is created but not displayed. Use [`show()`](Self::show) to make it visible.
    ///
    /// While hidden, the window renders off-screen instead of to its surface,
    /// so [`snap`](Self::snap), [`snap_image`](Self::snap_image) and recording
    /// work without ever displaying anything — this is how kiss3d does
    /// offscreen rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_hidden_with_size(title: &str, width: u32, height: u32) -> Window {
        Window::do_new(title, true, width, height, None).await
    }

    /// Creates a new visible window with default settings.
    ///
    /// The window is created and immediately visible with a default size of 800x600 pixels.
    /// Use this in combination with the `#[kiss3d::main]` macro for cross-platform rendering.
    ///
    /// # Arguments
    /// * `title` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    ///
    /// # Example
    /// ```no_run
    /// use kiss3d::prelude::*;
    ///
    /// #[kiss3d::main]
    /// async fn main() {
    ///     let mut window = Window::new("My Application").await;
    ///     let mut camera = OrbitCamera3d::default();
    ///     let mut scene = SceneNode3d::empty();
    ///
    ///     while window.render_3d(&mut scene, &mut camera).await {
    ///         // Your render loop code here
    ///     }
    /// }
    /// ```
    pub async fn new(title: &str) -> Window {
        Window::do_new(title, false, DEFAULT_WIDTH, DEFAULT_HEIGHT, None).await
    }

    /// Creates a new window with custom dimensions.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_size(title: &str, width: u32, height: u32) -> Window {
        Window::do_new(title, false, width, height, None).await
    }

    /// Creates a new window with custom setup options.
    ///
    /// This allows fine-grained control over window creation, including VSync and anti-aliasing settings.
    ///
    /// # Arguments
    /// * `title` - The window title
    /// * `width` - The window width in pixels
    /// * `height` - The window height in pixels
    /// * `setup` - A `CanvasSetup` struct containing the window configuration
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_setup(
        title: &str,
        width: u32,
        height: u32,
        setup: CanvasSetup,
    ) -> Window {
        Window::do_new(title, false, width, height, Some(setup)).await
    }

    /// Creates a new window with custom attributes.
    ///
    /// This allows fine-grained control over window creation.
    ///
    /// # Arguments
    /// * `window_attrs` - The window title
    ///
    /// # Returns
    /// A new `Window` instance
    pub async fn new_with_window_attributes(window_attrs: WindowAttributes) -> Window {
        Window::do_new_with_window_attributes(window_attrs, None).await
    }

    // TODO: make this pub?
    async fn do_new(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let window_attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width as f64, height as f64))
            .with_visible(!hide);
        Self::do_new_with_window_attributes(window_attrs, setup).await
    }
    async fn do_new_with_window_attributes(
        window_attrs: WindowAttributes,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let hide = !window_attrs.visible;
        let canvas = Canvas::open(window_attrs, setup, event_send).await;
        let (width, height) = canvas.size();
        // The HDR resolve pass tonemaps into the LDR swapchain. The rasterizer's
        // material pipelines are single-sampled, so the HDR film is too (see the
        // note in `render_single_frame`).
        let canvas_surface_format = canvas.surface_format();

        // Track window count for proper cleanup
        Context::increment_window_count();

        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        let mut usr_window = Window {
            should_close: false,
            first_frame: true,
            close_key: None,
            close_modifiers: None,
            last_timings: None,
            last_frame_instant: None,
            gpu_timer: GpuTimer::new(),
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ime_events: Rc::new(RefCell::new(Vec::new())),
            screen_2d: None,
            ambient_intensity: 0.2,
            ambient_color: crate::color::WHITE,
            fog: crate::light::Fog::default(),
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            hdr: HdrPipeline::new(width, height, 1, canvas_surface_format),
            skybox: crate::renderer::Skybox::new(),
            ssao: None,
            ssao_enabled: false,
            clustered: None,
            cluster_grid: crate::builtin::clustered::DEFAULT_GRID,
            max_lights_per_cluster: crate::builtin::clustered::DEFAULT_MAX_LIGHTS_PER_CLUSTER,
            reflection_probes: None,
            probe_capture: None,
            pending_probe_captures: Vec::new(),
            reflection_probe_size: crate::renderer::DEFAULT_PROBE_SIZE,
            ssr: None,
            ssr_enabled: false,
            dof: None,
            dof_enabled: false,
            transmission: None,
            transmission_enabled: true,
            reflector_oit: None,
            reflector_transmission: None,
            post_process_render_target: framebuffer_manager.new_render_target(width, height, true),
            post_process_render_target_b: framebuffer_manager
                .new_render_target(width, height, false),
            film_render_targets: None,
            scene_depth: None,
            offscreen_output_target: None,
            aov_renderer: None,
            hidden: hide,
            shadow_mapper: ShadowMapper::new(DEFAULT_SHADOW_RESOLUTION),
            framebuffer_manager,
            #[cfg(feature = "recording")]
            recording: None,
            #[cfg(feature = "rt_switcher")]
            raytracer: (None, false),
        };

        if hide {
            usr_window.canvas.hide()
        }

        usr_window
    }

    /// Creates a headless window with custom setup options: a full-featured
    /// [`Window`] backed by no OS window and no swapchain, rendering straight
    /// into an off-screen texture. Unlike [`OffscreenSurface`](crate::window::OffscreenSurface)
    /// this exposes the whole `Window` API (custom renderers, ray tracer,
    /// `snap*` readbacks, …), for callers that drive the render loop
    /// themselves and never present to a display.
    pub async fn new_headless_with_setup(width: u32, height: u32, setup: CanvasSetup) -> Window {
        Self::do_new_headless(width, height, Some(setup)).await
    }

    /// Creates a headless window: a render target backed by no actual window,
    /// for off-screen rendering. Powers [`OffscreenSurface`](crate::window::OffscreenSurface).
    pub(super) async fn do_new_headless(
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let canvas = Canvas::open_headless(width, height, setup, event_send).await;
        let (width, height) = canvas.size();
        let canvas_surface_format = canvas.surface_format();

        Context::increment_window_count();
        WindowCache::populate();

        let framebuffer_manager = FramebufferManager::new();
        Window {
            should_close: false,
            first_frame: true,
            close_key: None,
            close_modifiers: None,
            last_timings: None,
            last_frame_instant: None,
            gpu_timer: GpuTimer::new(),
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            ime_events: Rc::new(RefCell::new(Vec::new())),
            screen_2d: None,
            ambient_intensity: 0.2,
            ambient_color: crate::color::WHITE,
            fog: crate::light::Fog::default(),
            background: BLACK,
            polyline_renderer_2d: PolylineRenderer2d::new(),
            point_renderer_2d: PointRenderer2d::new(),
            point_renderer: PointRenderer3d::new(),
            polyline_renderer: PolylineRenderer3d::new(),
            text_renderer: TextRenderer::new(),
            #[cfg(feature = "egui")]
            egui_context: EguiContext::new(),
            // Offscreen rendering is single-sampled (see `render_single_frame`).
            hdr: HdrPipeline::new(width, height, 1, canvas_surface_format),
            skybox: crate::renderer::Skybox::new(),
            ssao: None,
            ssao_enabled: false,
            clustered: None,
            cluster_grid: crate::builtin::clustered::DEFAULT_GRID,
            max_lights_per_cluster: crate::builtin::clustered::DEFAULT_MAX_LIGHTS_PER_CLUSTER,
            reflection_probes: None,
            probe_capture: None,
            pending_probe_captures: Vec::new(),
            reflection_probe_size: crate::renderer::DEFAULT_PROBE_SIZE,
            ssr: None,
            ssr_enabled: false,
            dof: None,
            dof_enabled: false,
            transmission: None,
            transmission_enabled: true,
            reflector_oit: None,
            reflector_transmission: None,
            post_process_render_target: framebuffer_manager.new_render_target(width, height, true),
            post_process_render_target_b: framebuffer_manager
                .new_render_target(width, height, false),
            film_render_targets: None,
            scene_depth: None,
            offscreen_output_target: None,
            aov_renderer: None,
            // A headless window has no surface; always render off-screen.
            hidden: true,
            shadow_mapper: ShadowMapper::new(DEFAULT_SHADOW_RESOLUTION),
            framebuffer_manager,
            #[cfg(feature = "recording")]
            recording: None,
            #[cfg(feature = "rt_switcher")]
            raytracer: (None, false),
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // Only clean up GPU resources when the last window is dropped.
        // This prevents TLS access order issues with wgpu internals that can cause
        // panics during thread cleanup.
        let is_last_window = Context::decrement_window_count();

        if is_last_window {
            // The order matters: clear caches first (which hold references to GPU resources),
            // then clear the Context (which holds the wgpu Device/Queue/Instance).

            // Clear 3D resource managers
            WindowCache::reset();

            // Clear 2D resource managers
            MeshManager2d::reset_global_manager();
            MaterialManager2d::reset_global_manager();

            // Finally, clear the wgpu context itself
            Context::reset();
        }
    }
}
