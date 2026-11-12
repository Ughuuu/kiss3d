//! Unified wgpu-based canvas for both native and web platforms.

#[cfg(any(target_os = "android", test))]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::context::Context;
use crate::event::{Action, Key, Modifiers, MouseButton, TouchAction, WindowEvent};
use crate::window::access::{self, Access};
use crate::window::canvas::CanvasSetup;
use image::{GenericImage, Pixel};
#[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
use winit::application::ApplicationHandler;
#[cfg(not(target_arch = "wasm32"))]
use winit::event::{MouseScrollDelta, TouchPhase, WindowEvent as WinitWindowEvent};
#[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
use winit::event_loop::ActiveEventLoop;
#[cfg(not(target_os = "ios"))]
use winit::event_loop::EventLoop;
use winit::keyboard::ModifiersState;
#[cfg(not(target_arch = "wasm32"))]
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Icon, Window, WindowAttributes};

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
use wgpu::ExperimentalFeatures;

/// Computes the device features to request.
///
/// Opts into wgpu's experimental ray query + acceleration-structure features
/// whenever the adapter supports them, so the path tracer can use the hardware
/// backend; on platforms without support the feature is simply not requested and
/// the portable compute backend is used as a fallback.
fn raytracing_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let mut features = wgpu::Features::empty();
    let supported = adapter.features();

    // Per-pass GPU timestamp queries power the inspector's render timings. Only
    // requested when the adapter supports it; otherwise GPU timing is disabled
    // (and only the CPU submit/present/total timings are reported).
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    // Hardware ray queries for the path tracer, when the platform supports them.
    if supported.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    }

    features
}

/// The experimental-features token to pass to `request_device`.
///
/// `EXPERIMENTAL_RAY_QUERY` is gated behind wgpu's separate "I accept experimental
/// APIs" token in addition to being listed in `required_features`; requesting the
/// feature without enabling the token makes `request_device` fail. Returns the
/// enabled token only when ray query is actually being requested.
fn experimental_features(required: wgpu::Features) -> ExperimentalFeatures {
    if required.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        // SAFETY: we opt into wgpu's experimental hardware ray-query API. It may
        // still contain bugs; the path tracer's hardware backend accepts that to
        // use GPU-accelerated ray tracing where available.
        return unsafe { ExperimentalFeatures::enabled() };
    }
    ExperimentalFeatures::disabled()
}

/// Combines the features kiss3d always wants with the consumer-requested extras from
/// [`CanvasSetup::required_features`], dropping any the adapter doesn't support so
/// `request_device` never fails on an unavailable feature.
fn device_features(adapter: &wgpu::Adapter, extra: wgpu::Features) -> wgpu::Features {
    raytracing_features(adapter) | (extra & adapter.features())
}

// Thread-local EventLoop singleton for the platforms where kiss3d owns and
// pumps the loop. winit only allows one EventLoop per program, so we store it
// in thread-local storage and reuse it across window recreations. EventLoop
// is not Send/Sync, so we use thread_local! instead of a static Mutex.
// Absent on iOS, where winit owns the loop instead (see `window::ios`).
#[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

/// Ends a [`WgpuCanvas::wait_events`] from any thread.
#[derive(Clone, Debug)]
pub struct Waker(
    #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
    winit::event_loop::EventLoopProxy<()>,
);

impl Waker {
    /// Wake the waiting loop; harmless once it has gone.
    pub fn wake(&self) {
        #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
        let _ = self.0.send_event(());
    }
}

/// What ends a wait on the loop kiss3d pumps, if it pumps one here.
pub(crate) fn loop_waker() -> Option<Waker> {
    #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
    {
        EVENT_LOOP.with(|cell| cell.borrow().as_ref().map(|el| Waker(el.create_proxy())))
    }
    #[cfg(any(target_arch = "wasm32", target_os = "ios"))]
    {
        None
    }
}

/// Runs `with` on the loop's `ActiveEventLoop`, which exists only inside a
/// callback: one pump, whose window events go where every other pump's do.
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub(crate) fn with_active_loop<T>(with: impl FnOnce(&ActiveEventLoop) -> T) -> T {
    EVENT_LOOP.with(|cell| {
        let mut event_loop = cell.borrow_mut();
        let event_loop = event_loop
            .as_mut()
            .expect("a window has just been made on this loop");
        one_turn(event_loop, with)
    })
}

/// [`with_active_loop`] on a loop already in hand.
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
fn one_turn<T>(event_loop: &mut EventLoop<()>, with: impl FnOnce(&ActiveEventLoop) -> T) -> T {
    use winit::platform::pump_events::EventLoopExtPumpEvents;

    struct OneTurn<Asked, T> {
        asked: Option<Asked>,
        answer: Option<T>,
    }

    impl<Asked: FnOnce(&ActiveEventLoop) -> T, T> ApplicationHandler for OneTurn<Asked, T> {
        fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
            if let Some(asked) = self.asked.take() {
                self.answer = Some(asked(event_loop));
            }
        }

        fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

        fn window_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            window_id: winit::window::WindowId,
            event: WinitWindowEvent,
        ) {
            collect_window_event(window_id, event);
        }
    }

    let mut turn = OneTurn {
        asked: Some(with),
        answer: None,
    };
    loop {
        let _ = event_loop.pump_app_events(Some(std::time::Duration::ZERO), &mut turn);
        if let Some(answer) = turn.answer.take() {
            return answer;
        }
    }
}

/// `attrs` asking for no more than `monitor` holds. A window that does not fit
/// opens maximized, so the system sizes it to the screen less its own bars,
/// which winit cannot tell, and restores to the most the monitor holds.
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
fn fit_to_monitor(
    attrs: WindowAttributes,
    monitor: Option<winit::monitor::MonitorHandle>,
) -> WindowAttributes {
    let (Some(monitor), Some(asked)) = (monitor, attrs.inner_size) else {
        return attrs;
    };
    let scale = monitor.scale_factor();
    match held_size(asked.to_logical(scale), monitor.size().to_logical(scale)) {
        Some(held) => attrs.with_inner_size(held).with_maximized(true),
        None => attrs,
    }
}

/// The size a window asking for `asked` restores to on a monitor of `room`,
/// or `None` when it fits as asked.
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
fn held_size(
    asked: winit::dpi::LogicalSize<f64>,
    room: winit::dpi::LogicalSize<f64>,
) -> Option<winit::dpi::LogicalSize<f64>> {
    let empty = room.width <= 0.0 || room.height <= 0.0;
    if empty || (asked.width <= room.width && asked.height <= room.height) {
        return None;
    }
    Some(winit::dpi::LogicalSize::new(
        asked.width.min(room.width),
        asked.height.min(room.height),
    ))
}

/// The monitor a new window opens on: the primary one, or where a system
/// names none (Wayland), the smallest, which the window then fits anywhere.
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
fn opening_monitor(event_loop: &ActiveEventLoop) -> Option<winit::monitor::MonitorHandle> {
    event_loop.primary_monitor().or_else(|| {
        event_loop.available_monitors().min_by_key(|monitor| {
            let size = monitor.size();
            u64::from(size.width) * u64::from(size.height)
        })
    })
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    // Raw mouse motion heard since the last drain: what a held cursor moves by,
    // since its position stops at the window's edge or does not move at all.
    static RAW_MOTION: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((0.0, 0.0)) };
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    // Shared event storage for multi-window support. Events are stored per window_id
    // so each window can retrieve only its own events after pump_app_events runs
    // (or, on iOS, after winit's own loop delivers them).
    static PENDING_WINDOW_EVENTS: RefCell<std::collections::HashMap<winit::window::WindowId, Vec<PendingEvent>>> = RefCell::new(std::collections::HashMap::new());
    // Files dropped onto a window, kept out of PendingEvent because WindowEvent
    // is Copy and a PathBuf is not. Drained by `take_dropped_files`.
    static DROPPED_FILES: RefCell<Vec<(winit::window::WindowId, std::path::PathBuf)>> =
        const { RefCell::new(Vec::new()) };
    /// Composed text per window, drained by `take_ime_events` like dropped files.
    static IME_EVENTS: RefCell<Vec<(winit::window::WindowId, crate::event::ImeEvent)>> =
        const { RefCell::new(Vec::new()) };
}

// Android lifecycle plumbing. winit can only build an EventLoop from the
// `AndroidApp` that `android_main` receives, so it is stashed here first
// (see `init_android`). Resumed/Suspended arrive without a window id, so they
// are queued here rather than through PENDING_WINDOW_EVENTS; `poll_events`
// drains them to drop and recreate the surface, whose native window only
// exists between Resumed and Suspended.
#[cfg(target_os = "android")]
thread_local! {
    static ANDROID_APP: RefCell<Option<winit::platform::android::activity::AndroidApp>> =
        const { RefCell::new(None) };
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static LIFECYCLE_EVENTS: RefCell<Vec<LifecycleEvent>> = const { RefCell::new(Vec::new()) };
}

/// What the application, rather than a window, was told, kept until
/// `poll_events` hands it on.
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(not(any(target_os = "android", target_os = "ios")), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum LifecycleEvent {
    Resumed,
    Suspended,
    LowMemory,
}

/// Queue an application event for the next `poll_events`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn push_lifecycle(event: LifecycleEvent) {
    LIFECYCLE_EVENTS.with(|events| events.borrow_mut().push(event));
}

#[cfg(any(target_os = "android", test))]
thread_local! {
    // The least NativeActivity's content rect has left uncovered at each
    // window height seen: the navigation bar, where the system draws one over
    // the window. Kept per height, since a rotation moves the bar, and the
    // height seen last first.
    static ANDROID_RESTING: RefCell<Vec<(i32, i32)>> = const { RefCell::new(Vec::new()) };
    // Whether the keyboard was last asked to show.
    static KEYBOARD_ASKED: Cell<bool> = const { Cell::new(false) };
}

/// The keyboard's share of what the content rect leaves uncovered.
///
/// NativeActivity reports one rect for everything the system covers, keyboard
/// and navigation bar alike. The bar's share is the smallest cover seen at
/// this height, so what is left above it is the keyboard. A height first seen
/// with the keyboard asked for, as a rotation while typing is, borrows the
/// bar's share from the height before it rather than calling it all bar.
/// Window heights whose bar is remembered: both ways up, each in a split screen.
#[cfg(any(target_os = "android", test))]
const ANDROID_HEIGHTS: usize = 4;

#[cfg(any(target_os = "android", test))]
fn android_keyboard(height: i32, covered: i32) -> f32 {
    let asked = KEYBOARD_ASKED.with(Cell::get);
    ANDROID_RESTING.with(|resting| {
        let mut resting = resting.borrow_mut();
        let least = match resting.iter().position(|(at, _)| *at == height) {
            Some(index) => {
                let (_, least) = resting.remove(index);
                least.min(covered)
            }
            None if asked => resting.first().map_or(0, |(_, least)| *least).min(covered),
            None => covered,
        };
        resting.insert(0, (height, least));
        resting.truncate(ANDROID_HEIGHTS);
        (covered - least).max(0) as f32
    })
}

/// The navigation bar's share of the cover at `height`, as the keyboard's
/// reading last measured it.
#[cfg(target_os = "android")]
fn android_bar(height: i32) -> Option<i32> {
    ANDROID_RESTING.with(|resting| {
        resting
            .borrow()
            .iter()
            .find(|(at, _)| *at == height)
            .map(|(_, least)| *least)
    })
}

/// Stores the `AndroidApp` handle that `android_main` received, so opening a
/// window can build winit's event loop from it (a bare `EventLoop::new`
/// panics on Android by design). Must run on the `android_main` thread before
/// the first window opens; `#[kiss3d::main]` generates exactly that call.
#[cfg(target_os = "android")]
pub fn init_android(app: winit::platform::android::activity::AndroidApp) {
    ANDROID_APP.with(|cell| *cell.borrow_mut() = Some(app));
}

/// Internal event type that stores both the event data and state updates needed.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, PartialEq)]
enum PendingEvent {
    WindowEvent(WindowEvent),
    ButtonState(MouseButton, Action),
    KeyState(Key, Action),
    CursorPos(f64, f64),
    Modifiers(ModifiersState),
    /// The window changed size. The drain asks the window how big it is:
    /// winit's own figure can be stale, as one taken before a window was shown
    /// arrives after AppKit shrank it to fit the screen.
    Resize,
}

/// Translates one winit window event into the shared pending-event storage,
/// which `poll_events` drains per window. Shared by the pump collector in
/// `poll_events` (desktop, Android) and the iOS application handler
/// (`window::ios`), which receives events from winit's own loop instead of a
/// pump. Neither caller has access to the canvas, so every
/// `Modifiers::empty()` below is a placeholder: the real mask is stamped in
/// when the batch is drained.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn collect_window_event(window_id: winit::window::WindowId, event: WinitWindowEvent) {
    access::window_event(window_id, &event);
    let pending_events: Vec<PendingEvent> = match event {
        WinitWindowEvent::CloseRequested => {
            vec![PendingEvent::WindowEvent(WindowEvent::Close)]
        }
        WinitWindowEvent::Resized(_) => vec![PendingEvent::Resize],
        WinitWindowEvent::CursorEntered { .. } => {
            vec![PendingEvent::WindowEvent(WindowEvent::CursorEnter(true))]
        }
        WinitWindowEvent::CursorLeft { .. } => {
            vec![PendingEvent::WindowEvent(WindowEvent::CursorEnter(false))]
        }
        WinitWindowEvent::CursorMoved { position, .. } => {
            vec![
                PendingEvent::CursorPos(position.x, position.y),
                PendingEvent::WindowEvent(WindowEvent::CursorPos(
                    position.x,
                    position.y,
                    Modifiers::empty(), // Will be filled in when processing
                )),
            ]
        }
        WinitWindowEvent::MouseInput { state, button, .. } => {
            let action = translate_action(state);
            let button = translate_mouse_button(button);
            vec![
                PendingEvent::ButtonState(button, action),
                PendingEvent::WindowEvent(WindowEvent::MouseButton(
                    button,
                    action,
                    Modifiers::empty(),
                )),
            ]
        }
        WinitWindowEvent::Touch(touch) => {
            let action = match touch.phase {
                TouchPhase::Started => TouchAction::Start,
                TouchPhase::Ended => TouchAction::End,
                TouchPhase::Moved => TouchAction::Move,
                TouchPhase::Cancelled => TouchAction::Cancel,
            };
            vec![PendingEvent::WindowEvent(WindowEvent::Touch(
                touch.id,
                touch.location.x,
                touch.location.y,
                action,
                Modifiers::empty(),
            ))]
        }
        WinitWindowEvent::MouseWheel { delta, .. } => {
            let (x, y) = match delta {
                MouseScrollDelta::LineDelta(dx, dy) => (dx as f64 * 10.0, dy as f64 * 10.0),
                MouseScrollDelta::PixelDelta(delta) => (delta.x, delta.y),
            };
            vec![PendingEvent::WindowEvent(WindowEvent::Scroll(
                x,
                y,
                Modifiers::empty(),
            ))]
        }
        WinitWindowEvent::KeyboardInput { event, .. } => {
            let action = translate_action(event.state);
            let key = translate_key(event.physical_key);
            translate_keyboard_input(key, action, &event.logical_key, event.text.as_deref())
        }
        WinitWindowEvent::ModifiersChanged(new_modifiers) => {
            vec![PendingEvent::Modifiers(new_modifiers.state())]
        }
        WinitWindowEvent::Focused(focused) => {
            vec![PendingEvent::WindowEvent(WindowEvent::Focus(focused))]
        }
        // Minimised, covered or on another space: out of sight, as a
        // backgrounded app is.
        WinitWindowEvent::Occluded(hidden) => {
            vec![PendingEvent::WindowEvent(WindowEvent::Iconify(hidden))]
        }
        WinitWindowEvent::DroppedFile(path) => {
            DROPPED_FILES.with(|dropped| dropped.borrow_mut().push((window_id, path)));
            vec![]
        }
        WinitWindowEvent::Ime(ime) => {
            let event = crate::event::ImeEvent::from_winit(ime);
            IME_EVENTS.with(|events| events.borrow_mut().push((window_id, event)));
            vec![]
        }
        _ => vec![],
    };

    if !pending_events.is_empty() {
        PENDING_WINDOW_EVENTS.with(|storage| {
            storage
                .borrow_mut()
                .entry(window_id)
                .or_default()
                .extend(pending_events);
        });
    }
}

/// A GPU→CPU pixel readback still in flight (see `WgpuCanvas::begin_read_pixels`).
struct PendingSnap {
    buffer: wgpu::Buffer,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    submission: wgpu::SubmissionIndex,
    width: usize,
    height: usize,
    padded_bytes_per_row: usize,
}

/// A unified canvas based on wgpu that works on both native and web platforms.
#[allow(dead_code)]
pub struct WgpuCanvas {
    window: Option<Arc<Window>>,
    #[cfg(not(target_arch = "wasm32"))]
    window_id: Option<winit::window::WindowId>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: wgpu::SurfaceConfiguration,
    cursor_pos: Option<(f64, f64)>,
    held: Held,
    /// Whether a grab holds the cursor, so motion is read raw.
    cursor_held: std::cell::Cell<bool>,
    /// Between the system's `Suspended` and `Resumed`: an app in the
    /// background that keeps rendering is one iOS ends.
    suspended: bool,
    out_events: Sender<WindowEvent>,
    modifiers_state: ModifiersState,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Whether an effect reads the depth, which a multisampled one must then be
    /// a texture for.
    depth_readable: bool,
    /// Multisampling texture for MSAA (if enabled)
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    /// Number of samples for MSAA
    sample_count: u32,
    /// Texture for reading back pixels (for screenshots), made on the first
    /// capture. It is as large as the surface, and a run that never takes a
    /// shot never pays for it; a resize drops it rather than rebuilding one
    /// nothing has asked for.
    readback_texture: std::cell::RefCell<Option<wgpu::Texture>>,
    /// Staging buffer reused across `read_pixels` calls, grown on demand, so
    /// per-frame capture doesn't allocate (and free) a GPU buffer every call.
    screenshot_staging: RefCell<Option<wgpu::Buffer>>,
    /// Readback started by `begin_read_pixels`, completed by `finish_read_pixels`.
    snap_pending: RefCell<Option<PendingSnap>>,
    /// Pending events from web callbacks (WASM only)
    #[cfg(target_arch = "wasm32")]
    pending_events: Rc<RefCell<Vec<WindowEvent>>>,
    /// Keep closures alive (WASM only)
    #[cfg(target_arch = "wasm32")]
    _event_closures: Vec<wasm_bindgen::JsValue>,
    /// A screen reader's way in, for a window opened with `accessible`.
    access: Option<Access>,
}

impl WgpuCanvas {
    /// Opens a new window and initializes the wgpu context.
    pub async fn open(
        window_attrs: WindowAttributes,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        let canvas_setup = canvas_setup.unwrap_or_default();

        // An adapter has to exist before the window is first shown.
        let accessible = canvas_setup.accessible && access::AVAILABLE;
        let shown = window_attrs.visible;
        let window_attrs = window_attrs.with_visible(shown && !accessible);

        // Create the window.
        //
        // iOS first: winit's loop owns the thread there (see `window::ios`),
        // so the window comes from the callback-scoped ActiveEventLoop rather
        // than from a pumpable EventLoop of our own.
        #[cfg(target_os = "ios")]
        let window = super::ios::create_window(window_attrs);

        #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
        let window = {
            // Get or create the thread-local EventLoop (winit only allows one per program)
            EVENT_LOOP.with(|event_loop_cell| {
                let mut event_loop_opt = event_loop_cell.borrow_mut();
                if event_loop_opt.is_none() {
                    // Android refuses a bare EventLoop::new: the loop must be
                    // built from android_main's AndroidApp handle.
                    #[cfg(target_os = "android")]
                    {
                        use winit::platform::android::EventLoopBuilderExtAndroid;
                        let app = ANDROID_APP.with(|cell| cell.borrow().clone()).expect(
                            "call kiss3d::window::init_android(app) from android_main \
                             before opening a window (#[kiss3d::main] does this)",
                        );
                        *event_loop_opt = Some(
                            EventLoop::builder()
                                .with_android_app(app)
                                .build()
                                .expect("Failed to create event loop"),
                        );
                    }
                    #[cfg(not(target_os = "android"))]
                    {
                        *event_loop_opt =
                            Some(EventLoop::new().expect("Failed to create event loop"));
                    }
                }

                // Android's native window only exists once the activity is
                // resumed; a window created before that has no handle wgpu can
                // make a surface from. Pump until the first Resumed arrives.
                #[cfg(target_os = "android")]
                {
                    use winit::platform::pump_events::EventLoopExtPumpEvents;

                    struct WaitForResume(bool);

                    impl ApplicationHandler for WaitForResume {
                        fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
                            self.0 = true;
                        }

                        fn window_event(
                            &mut self,
                            _event_loop: &ActiveEventLoop,
                            _window_id: winit::window::WindowId,
                            _event: WinitWindowEvent,
                        ) {
                        }
                    }

                    let event_loop = event_loop_opt.as_mut().unwrap();
                    let mut waiter = WaitForResume(false);
                    while !waiter.0 {
                        let _ = event_loop.pump_app_events(
                            Some(std::time::Duration::from_millis(16)),
                            &mut waiter,
                        );
                    }
                }

                #[cfg(not(target_os = "android"))]
                let window_attrs = {
                    let monitor = one_turn(event_loop_opt.as_mut().unwrap(), opening_monitor);
                    fit_to_monitor(window_attrs, monitor)
                };

                let event_loop = event_loop_opt.as_ref().unwrap();
                #[allow(deprecated)]
                event_loop
                    .create_window(window_attrs)
                    .expect("Failed to create window")
            })
        };

        #[cfg(target_arch = "wasm32")]
        let window = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            // For WASM, we create a local EventLoop (single-threaded environment)
            let events = EventLoop::new().expect("Failed to create event loop");

            let web_window = web_sys::window().expect("Failed to get web_sys window");
            let document = web_window.document().expect("Failed to get document");

            // Try to find an existing canvas with the configured id, or create one
            let page_canvas = document
                .get_element_by_id(&canvas_setup.canvas_id)
                .and_then(|elem| elem.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            let created = page_canvas.is_none();
            let canvas = page_canvas.unwrap_or_else(|| {
                // Create a new canvas element
                let canvas = document
                    .create_element("canvas")
                    .expect("Failed to create canvas element")
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .expect("Failed to cast to HtmlCanvasElement");
                canvas.set_id(&canvas_setup.canvas_id);

                // Append to body
                if let Some(body) = document.body() {
                    body.append_child(&canvas)
                        .expect("Failed to append canvas to body");
                }

                canvas
            });

            // A canvas kiss3d had to create is the whole page: size the
            // document around it. One the page supplied sits in a layout the
            // page owns, which keeps its margins and its scrolling.
            if created {
                if let Some(html) = document.document_element() {
                    if let Some(html) = html.dyn_ref::<web_sys::HtmlElement>() {
                        let style = html.style();
                        let _ = style.set_property("margin", "0");
                        let _ = style.set_property("padding", "0");
                        let _ = style.set_property("width", "100%");
                        let _ = style.set_property("height", "100%");
                    }
                }
                if let Some(body) = document.body() {
                    let style = body.style();
                    let _ = style.set_property("margin", "0");
                    let _ = style.set_property("padding", "0");
                    let _ = style.set_property("width", "100%");
                    let _ = style.set_property("height", "100%");
                    let _ = style.set_property("overflow", "hidden");
                }
            }

            let window_attrs = window_attrs.with_canvas(Some(canvas));

            #[allow(deprecated)]
            let window = events
                .create_window(window_attrs)
                .expect("Failed to create window");

            // Style the canvas AFTER winit creates the window (winit may overwrite styles)
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let style = canvas.style();
                let _ = style.set_property("display", "block");
                let _ = style.set_property("width", "100%");
                let _ = style.set_property("height", "100%");
            }

            window
        };

        let window = Arc::new(window);
        let access = accessible.then(|| Access::attach(&window));
        if accessible {
            // In a loop turn, so the resize AppKit reports as it shrinks a
            // window too big for the screen reaches the canvas.
            #[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
            with_active_loop(|_| window.set_visible(shown));
            #[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
            window.set_visible(shown);
        }

        // Check if we already have a context initialized (multi-window case)
        let (surface, surface_format) = if Context::is_initialized() {
            // Reuse the existing context - create a new surface using the shared instance
            let ctxt = Context::get();

            let surface = ctxt
                .instance
                .create_surface(window.clone())
                .expect("Failed to create surface");

            // Configure surface with existing device
            let surface_caps = surface.get_capabilities(&ctxt.adapter);
            let enabled_features = ctxt.device.features();
            let surface_format = surface_caps
                .formats
                .iter()
                .find(|f| !f.is_srgb() && enabled_features.contains(f.required_features()))
                .copied()
                .unwrap_or(surface_caps.formats[0]);

            (surface, surface_format)
        } else {
            // First window - create the full wgpu context.
            //
            // Instance, surface and adapter are requested together because on
            // Android the backends are tried one at a time: an all-backends
            // instance creates a Vulkan and a GL surface for the same
            // ANativeWindow, and vkCreateAndroidSurfaceKHR connects the
            // window's buffer queue even when the Vulkan adapter turns out
            // unusable (emulators without GPU passthrough), after which the
            // GL fallback's eglCreateWindowSurface fails with "already
            // connected to another API". Separate instances mean the failed
            // candidate's surface is dropped — and the window disconnected —
            // before the next backend touches it.
            let asked = web_time::Instant::now();
            let request = |backends| {
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends,
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                let window = window.clone();
                async move {
                    let surface = instance.create_surface(window).ok()?;
                    let adapter = instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::default(),
                            compatible_surface: Some(&surface),
                            force_fallback_adapter: false,
                            apply_limit_buckets: false,
                        })
                        .await
                        .ok()?;
                    Some((instance, surface, adapter))
                }
            };

            #[cfg(target_os = "android")]
            let picked = match request(wgpu::Backends::VULKAN).await {
                Some(picked) => Some(picked),
                None => request(wgpu::Backends::GL).await,
            };
            #[cfg(not(target_os = "android"))]
            let picked = request(wgpu::Backends::all()).await;

            let (instance, surface, adapter) =
                picked.expect("Failed to create a surface and find an adapter for it");

            // Request the adapter's full limits on every platform. The path tracer,
            // the shadow-mapped material, and the storage-backed point/wireframe
            // renderers need more bind groups and per-stage storage buffers (and, for
            // the path tracer, compute) than wgpu's conservative cross-platform
            // defaults allow. On native and on WebGPU browsers `adapter.limits()`
            // grants these. On a WebGL2-only browser the adapter reports the (much
            // lower) WebGL2 caps, so requesting them is still valid and the basic
            // rasterizer keeps working — only the storage/compute shaders are
            // unavailable there, which is an inherent WebGL2 limitation.
            let limits = adapter.limits();

            let required_features = device_features(&adapter, canvas_setup.required_features);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("kiss3d device"),
                    required_features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: experimental_features(required_features),
                })
                .await
                .expect("Failed to create device");
            Context::note_device_wait(asked.elapsed());

            // Get surface capabilities
            // We explicitly prefer non-sRGB formats for consistent behavior across platforms.
            // WebGL2 often doesn't support sRGB framebuffers, so we do manual gamma correction
            // in shaders instead. This ensures colors look the same on native and web.
            let surface_caps = surface.get_capabilities(&adapter);
            let enabled_features = device.features();
            let surface_format = surface_caps
                .formats
                .iter()
                .find(|f| !f.is_srgb() && enabled_features.contains(f.required_features()))
                .copied()
                .unwrap_or(surface_caps.formats[0]);

            // Initialize the global context (only for first window)
            Context::init(instance, device, queue, adapter, surface_format);

            (surface, surface_format)
        };

        let ctxt = Context::get();

        // Get surface capabilities for alpha mode
        let surface_caps = surface.get_capabilities(&ctxt.adapter);

        let size = drawable_size(&window);
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Configure surface
        let present_mode = if canvas_setup.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        // COPY_SRC powers surface screenshots, but downlevel surfaces (GLES on
        // Android) only offer RENDER_ATTACHMENT, and asking for more is a
        // validation error. Request it only where the surface supports it;
        // reading the surface back is simply unavailable elsewhere.
        let mut surface_usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if surface_caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            surface_usage |= wgpu::TextureUsages::COPY_SRC;
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: surface_usage,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode,
            // Opaque where offered: a WebGPU canvas lists premultiplied first,
            // and the page would show through every pixel left at alpha 0.
            alpha_mode: if surface_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::Opaque)
            {
                wgpu::CompositeAlphaMode::Opaque
            } else {
                surface_caps.alpha_modes[0]
            },
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctxt.device, &surface_config);

        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(
            &ctxt.device,
            width,
            height,
            canvas_setup.samples as u32,
            false,
        );

        // Create MSAA texture if needed
        let sample_count = canvas_setup.samples as u32;
        let (msaa_texture, msaa_view) = if sample_count > 1 {
            let (tex, view) = Self::create_msaa_texture(
                &ctxt.device,
                width,
                height,
                surface_format,
                sample_count,
            );
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        // The readback texture is built on the first capture, not here.
        let readback_texture = std::cell::RefCell::new(None);

        // Set up WASM event listeners
        #[cfg(target_arch = "wasm32")]
        let (pending_events, _event_closures) = {
            use winit::platform::web::WindowExtWebSys;

            let pending_events = Rc::new(RefCell::new(Vec::new()));
            let mut closures: Vec<wasm_bindgen::JsValue> = Vec::new();

            if let Some(canvas) = window.canvas() {
                // Pointer move (using pointer events for consistency)
                {
                    let pending = pending_events.clone();
                    let canvas_clone = canvas.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            let (x, y) = canvas_point(&canvas_clone, &event);
                            let mut pending = pending.borrow_mut();
                            pending.push(WindowEvent::CursorPos(x, y, event.modifiers()));
                            // A locked pointer keeps its place; only the
                            // movement says where the mouse went.
                            let locked = web_sys::window()
                                .and_then(|page| page.document())
                                .and_then(|document| document.pointer_lock_element())
                                .is_some();
                            if locked {
                                pending.push(WindowEvent::MouseMotion(
                                    f64::from(event.movement_x()),
                                    f64::from(event.movement_y()),
                                ));
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointermove",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Pointer down
                {
                    let pending = pending_events.clone();
                    let canvas_clone = canvas.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            // Only handle mouse pointer type (not touch - that's handled separately)
                            if event.pointer_type() == "mouse" {
                                let button = translate_web_mouse_button(event.button());
                                let (x, y) = canvas_point(&canvas_clone, &event);
                                // The release then arrives here even off the canvas.
                                let _ = canvas_clone.set_pointer_capture(event.pointer_id());
                                let mut pending = pending.borrow_mut();
                                // Where the press happened, which no move may
                                // have reported: a tap that never moved.
                                pending.push(WindowEvent::CursorPos(x, y, event.modifiers()));
                                pending.push(WindowEvent::MouseButton(
                                    button,
                                    Action::Press,
                                    event.modifiers(),
                                ));
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointerdown",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Pointer up
                {
                    let pending = pending_events.clone();
                    let canvas_clone = canvas.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::PointerEvent| {
                            // Only handle mouse pointer type (not touch - that's handled separately)
                            if event.pointer_type() == "mouse" {
                                let button = translate_web_mouse_button(event.button());
                                let (x, y) = canvas_point(&canvas_clone, &event);
                                let mut pending = pending.borrow_mut();
                                pending.push(WindowEvent::CursorPos(x, y, event.modifiers()));
                                pending.push(WindowEvent::MouseButton(
                                    button,
                                    Action::Release,
                                    event.modifiers(),
                                ));
                            }
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "pointerup",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Wheel
                {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
                            // Prevent default scrolling behavior
                            event.prevent_default();
                            // Scale based on delta mode to match native behavior:
                            // Browsers report much larger pixel deltas than native platforms,
                            // so we normalize them to produce similar scroll behavior.
                            // 0 = DOM_DELTA_PIXEL, 1 = DOM_DELTA_LINE, 2 = DOM_DELTA_PAGE
                            let scale = match event.delta_mode() {
                                0 => 0.1,  // Pixel mode - scale down (browsers report ~100px per tick)
                                1 => 1.0,  // Line mode - use as-is (browsers report ~1-3 lines)
                                _ => 10.0, // Page mode - scale up slightly
                            };
                            let dx = event.delta_x() * scale;
                            let dy = -event.delta_y() * scale; // Invert for natural scrolling
                            pending.borrow_mut().push(WindowEvent::Scroll(
                                dx,
                                dy,
                                event.modifiers(),
                            ));
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "wheel",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Context menu (prevent right-click menu)
                {
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                            event.prevent_default();
                        });
                    let _ = canvas.add_event_listener_with_callback(
                        "contextmenu",
                        closure.as_ref().unchecked_ref(),
                    );
                    closures.push(closure.into_js_value());
                }

                // Touch, in the same canvas pixels as the pointer.
                for (name, action) in [
                    ("touchstart", TouchAction::Start),
                    ("touchmove", TouchAction::Move),
                    ("touchend", TouchAction::End),
                    ("touchcancel", TouchAction::Cancel),
                ] {
                    let pending = pending_events.clone();
                    let canvas_clone = canvas.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
                            event.prevent_default();
                            let modifiers = event.modifiers();
                            let touches = event.changed_touches();
                            for i in 0..touches.length() {
                                if let Some(touch) = touches.get(i) {
                                    let (x, y) = canvas_point_at(
                                        &canvas_clone,
                                        f64::from(touch.client_x()),
                                        f64::from(touch.client_y()),
                                    );
                                    pending.borrow_mut().push(WindowEvent::Touch(
                                        touch.identifier() as u64,
                                        x,
                                        y,
                                        action,
                                        modifiers,
                                    ));
                                }
                            }
                        });
                    let _ = canvas
                        .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref());
                    closures.push(closure.into_js_value());
                }

                // The pointer leaving the canvas ends hover, as `CursorLeft` does.
                for (name, inside) in [("pointerenter", true), ("pointerleave", false)] {
                    let pending = pending_events.clone();
                    let closure =
                        Closure::<dyn FnMut(_)>::new(move |_event: web_sys::PointerEvent| {
                            pending.borrow_mut().push(WindowEvent::CursorEnter(inside));
                        });
                    let _ = canvas
                        .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref());
                    closures.push(closure.into_js_value());
                }
            }

            // Keyboard events on window (document level)
            let web_window = web_sys::window().expect("Failed to get web_sys window");
            {
                let pending = pending_events.clone();
                let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
                    let key = translate_web_key(&event.code());
                    let modifiers = event.modifiers();
                    pending
                        .borrow_mut()
                        .push(WindowEvent::Key(key, Action::Press, modifiers));
                    // macOS sends no keyup for a key released while ⌘ is held,
                    // which would leave that key pressed for good. Remember it
                    // and release it when ⌘ itself goes up, so it still reads
                    // as held for as long as it really is.
                    if modifiers.contains(Modifiers::Super)
                        && apple_platform()
                        && !is_modifier_key(key)
                    {
                        SUPER_HELD_KEYS.with(|held| {
                            let mut held = held.borrow_mut();
                            if !held.contains(&key) {
                                held.push(key);
                            }
                        });
                    }
                    // Emit a Char event for single-character (printable) keys so
                    // egui text fields receive text input.
                    let key_string = event.key();
                    if !is_shortcut(modifiers) && key_string.chars().count() == 1 {
                        if let Some(ch) = key_string.chars().next() {
                            pending.borrow_mut().push(WindowEvent::Char(ch));
                        }
                    }
                });
                let _ = web_window
                    .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
                closures.push(closure.into_js_value());
            }

            {
                let pending = pending_events.clone();
                let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
                    let key = translate_web_key(&event.code());
                    let modifiers = event.modifiers();
                    pending
                        .borrow_mut()
                        .push(WindowEvent::Key(key, Action::Release, modifiers));
                    // ⌘ is up, so the keys it swallowed the keyup of are too.
                    if !modifiers.contains(Modifiers::Super) {
                        let swallowed = SUPER_HELD_KEYS.with(|held| held.take());
                        let mut pending = pending.borrow_mut();
                        for key in swallowed {
                            pending.push(WindowEvent::Key(key, Action::Release, modifiers));
                        }
                    }
                });
                let _ = web_window
                    .add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref());
                closures.push(closure.into_js_value());
            }

            // A key or button let go while the page is not focused sends no
            // release here, so losing focus lets go of everything held.
            for (name, focused) in [("blur", false), ("focus", true)] {
                let pending = pending_events.clone();
                let closure = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::Event| {
                    if !focused {
                        SUPER_HELD_KEYS.with(|held| held.take());
                    }
                    pending.borrow_mut().push(WindowEvent::Focus(focused));
                });
                let _ = web_window
                    .add_event_listener_with_callback(name, closure.as_ref().unchecked_ref());
                closures.push(closure.into_js_value());
            }

            (pending_events, closures)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let window_id = window.id();

        WgpuCanvas {
            access,
            window: Some(window),
            #[cfg(not(target_arch = "wasm32"))]
            window_id: Some(window_id),
            surface: Some(surface),
            surface_config,
            cursor_pos: None,
            held: Held::default(),
            cursor_held: std::cell::Cell::new(false),
            suspended: false,
            out_events,
            modifiers_state: ModifiersState::default(),
            depth_texture,
            depth_view,
            depth_readable: false,
            msaa_texture,
            msaa_view,
            sample_count,
            readback_texture,
            screenshot_staging: RefCell::new(None),
            snap_pending: RefCell::new(None),
            #[cfg(target_arch = "wasm32")]
            pending_events,
            #[cfg(target_arch = "wasm32")]
            _event_closures,
        }
    }

    /// Opens a headless canvas: a wgpu context with no window and no surface,
    /// for off-screen rendering. Works without a display server.
    pub async fn open_headless(
        width: u32,
        height: u32,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        let canvas_setup = canvas_setup.unwrap_or_default();
        let width = width.max(1);
        let height = height.max(1);

        // Reuse the wgpu context if one already exists (e.g. a window was
        // created first); otherwise create a surface-less one.
        let surface_format = if Context::is_initialized() {
            Context::get().surface_format
        } else {
            let asked = web_time::Instant::now();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });

            let ask = async |fallback: bool| {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::default(),
                        compatible_surface: None,
                        force_fallback_adapter: fallback,
                        apply_limit_buckets: false,
                    })
                    .await
                    .ok()
            };
            // A machine with no GPU driver still has a software one: WARP on
            // Windows, lavapipe where Mesa is installed. Slow, and the only
            // way a surface-less render runs on a CI box at all.
            let adapter = match ask(false).await {
                Some(adapter) => adapter,
                None => ask(true)
                    .await
                    .expect("Failed to find an appropriate adapter, software one included"),
            };

            let required_features = device_features(&adapter, canvas_setup.required_features);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("kiss3d headless device"),
                    required_features,
                    // See the windowed path: request the adapter's full limits so
                    // the path tracer and shadow-mapped material have enough bind
                    // groups and per-stage storage buffers.
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: experimental_features(required_features),
                })
                .await
                .expect("Failed to create device");
            Context::note_device_wait(asked.elapsed());

            // No surface to query for a preferred format; pick a widely
            // supported non-sRGB format (gamma is handled in shaders).
            let surface_format = wgpu::TextureFormat::Rgba8Unorm;
            Context::init(instance, device, queue, adapter, surface_format);
            surface_format
        };

        let ctxt = Context::get();
        let sample_count = canvas_setup.samples as u32;

        // Kept only to carry the size and format; no surface is configured.
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let (depth_texture, depth_view) =
            Self::create_depth_texture(&ctxt.device, width, height, sample_count, false);
        let (msaa_texture, msaa_view) = if sample_count > 1 {
            let (tex, view) = Self::create_msaa_texture(
                &ctxt.device,
                width,
                height,
                surface_format,
                sample_count,
            );
            (Some(tex), Some(view))
        } else {
            (None, None)
        };
        let readback_texture = std::cell::RefCell::new(None);

        WgpuCanvas {
            access: None,
            window: None,
            #[cfg(not(target_arch = "wasm32"))]
            window_id: None,
            surface: None,
            surface_config,
            cursor_pos: None,
            held: Held::default(),
            cursor_held: std::cell::Cell::new(false),
            suspended: false,
            out_events,
            modifiers_state: ModifiersState::default(),
            depth_texture,
            depth_view,
            depth_readable: false,
            msaa_texture,
            msaa_view,
            sample_count,
            readback_texture,
            screenshot_staging: RefCell::new(None),
            snap_pending: RefCell::new(None),
            #[cfg(target_arch = "wasm32")]
            pending_events: Rc::new(RefCell::new(Vec::new())),
            #[cfg(target_arch = "wasm32")]
            _event_closures: Vec::new(),
        }
    }

    /// Resizes the canvas render targets.
    ///
    /// For an off-screen (headless) canvas this is the only way to change the
    /// render size, since there is no window to emit resize events.
    /// Whether vsync (the `AutoVsync` present mode) is currently enabled.
    pub fn vsync(&self) -> bool {
        self.surface_config.present_mode == wgpu::PresentMode::AutoVsync
    }

    /// Enables/disables vsync at runtime by switching the surface present mode
    /// (`AutoVsync` ↔ `AutoNoVsync`) and reconfiguring the surface. With vsync off,
    /// frames present as fast as the GPU produces them (uncapped), which is what you
    /// want when measuring GPU-bound throughput; on, presentation is paced to the
    /// display refresh. No-op on a headless/offscreen canvas (no surface).
    pub fn set_vsync(&mut self, enabled: bool) {
        let present_mode = if enabled {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if self.surface_config.present_mode == present_mode {
            return;
        }
        self.surface_config.present_mode = present_mode;
        if let Some(surface) = &self.surface {
            let ctxt = Context::get();
            surface.configure(&ctxt.device, &self.surface_config);
        }
    }

    /// Brings the surface to the size the window has now, read from the
    /// window rather than from an event, and tells the app. Whether it changed.
    fn fit_to_window(&mut self) -> bool {
        let Some(size) = self.window.as_deref().map(drawable_size) else {
            return false;
        };
        let current = (self.surface_config.width, self.surface_config.height);
        if size.width == 0 || size.height == 0 || (size.width, size.height) == current {
            return false;
        }
        self.resize(size.width, size.height);
        let _ = self
            .out_events
            .send(WindowEvent::FramebufferSize(size.width, size.height));
        true
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.surface_config.width == width && self.surface_config.height == height {
            return;
        }

        let ctxt = Context::get();
        self.surface_config.width = width;
        self.surface_config.height = height;
        if let Some(surface) = &self.surface {
            surface.configure(&ctxt.device, &self.surface_config);
        }

        let (depth_texture, depth_view) = Self::create_depth_texture(
            &ctxt.device,
            width,
            height,
            self.sample_count,
            self.depth_readable,
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;

        if self.sample_count > 1 {
            let (msaa_texture, msaa_view) = Self::create_msaa_texture(
                &ctxt.device,
                width,
                height,
                self.surface_config.format,
                self.sample_count,
            );
            self.msaa_texture = Some(msaa_texture);
            self.msaa_view = Some(msaa_view);
        }

        // Dropped, not rebuilt: the next capture makes one at the new size.
        self.readback_texture.replace(None);
    }

    /// Make the depth buffer one an effect can read, or stop; a multisampled
    /// one is rebuilt to match.
    pub(crate) fn set_depth_readable(&mut self, readable: bool) {
        if self.depth_readable == readable {
            return;
        }
        self.depth_readable = readable;
        if self.sample_count > 1 {
            let ctxt = Context::get();
            let (depth_texture, depth_view) = Self::create_depth_texture(
                &ctxt.device,
                self.surface_config.width.max(1),
                self.surface_config.height.max(1),
                self.sample_count,
                readable,
            );
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    /// Changes the MSAA sample count, recreating the size-dependent attachments
    /// (depth + MSAA color) to match.
    ///
    /// The swapchain is single-sample regardless, so it is left untouched. The HDR
    /// film, OIT targets and the rasterization pipelines re-derive the new count on
    /// the next frame (the pipelines are cached per sample count), so no other state
    /// needs to be rebuilt here.
    pub fn set_sample_count(&mut self, sample_count: u32) {
        let sample_count = sample_count.max(1);
        if self.sample_count == sample_count {
            return;
        }
        self.sample_count = sample_count;

        let ctxt = Context::get();
        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);

        let (depth_texture, depth_view) = Self::create_depth_texture(
            &ctxt.device,
            width,
            height,
            sample_count,
            self.depth_readable,
        );
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;

        if sample_count > 1 {
            let (msaa_texture, msaa_view) = Self::create_msaa_texture(
                &ctxt.device,
                width,
                height,
                self.surface_config.format,
                sample_count,
            );
            self.msaa_texture = Some(msaa_texture);
            self.msaa_view = Some(msaa_view);
        } else {
            // Drop the (now unused) multisampled color attachment.
            self.msaa_texture = None;
            self.msaa_view = None;
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        readable: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let sample_count = sample_count.max(1);
        // A multisampled depth is a texture only while an effect reads it: GLES
        // makes it a multisampled texture then, which some drivers cannot attach.
        let usage = if sample_count == 1 || readable {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: Context::depth_format(),
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// The readback texture, built at the surface's current size the first
    /// time a capture asks for it. Cloning is cheap: a `wgpu::Texture` is a
    /// handle.
    fn readback_texture(&self) -> wgpu::Texture {
        let mut slot = self.readback_texture.borrow_mut();
        if slot.is_none() {
            *slot = Some(Self::create_readback_texture(
                &Context::get().device,
                self.surface_config.width,
                self.surface_config.height,
                self.surface_config.format,
            ));
        }
        slot.as_ref().expect("just built").clone()
    }

    fn create_readback_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        // Ensure minimum dimensions of 1x1 to avoid wgpu validation errors
        let width = width.max(1);
        let height = height.max(1);
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("readback_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Polls events from the window system. Answers whether any arrived.
    pub fn poll_events(&mut self) -> bool {
        self.pump_events(Some(std::time::Duration::ZERO))
    }

    /// Blocks until the window system has an event, a [`Waker`] wakes it, or
    /// `timeout` passes, then polls as [`Self::poll_events`] does; `None`
    /// waits for as long as it takes. Answers whether anything arrived. The
    /// web and iOS own their loops and cannot block here, so there it polls.
    pub fn wait_events(&mut self, timeout: Option<std::time::Duration>) -> bool {
        self.pump_events(timeout)
    }

    /// Something that ends a [`Self::wait_events`] from another thread.
    /// `None` where the loop is not kiss3d's to pump.
    pub fn waker(&self) -> Option<Waker> {
        loop_waker()
    }

    /// What a screen reader asked since the last call.
    #[cfg(feature = "egui")]
    pub fn take_access_events(&self) -> Vec<crate::window::AccessEvent> {
        self.access
            .as_ref()
            .map_or_else(Vec::new, Access::take_events)
    }

    /// Hand a screen reader what changed. The platform drops it while none has
    /// asked; the first update after [`AccessEvent::Activated`] has to be the
    /// whole tree.
    ///
    /// [`AccessEvent::Activated`]: crate::window::AccessEvent::Activated
    #[cfg(feature = "egui")]
    pub fn update_access(&self, update: egui::accesskit::TreeUpdate) {
        if let Some(access) = &self.access {
            access.update(update);
        }
    }

    fn pump_events(&mut self, timeout: Option<std::time::Duration>) -> bool {
        // A headless canvas has no window and no event loop; nothing to poll.
        if self.window.is_none() {
            return false;
        }
        let mut arrived = false;
        #[cfg(any(target_arch = "wasm32", target_os = "ios"))]
        let _ = timeout;

        #[cfg(not(target_arch = "wasm32"))]
        {
            // iOS never pumps: winit owns the loop there (see `window::ios`)
            // and feeds `collect_window_event` itself, so only the drain below
            // runs. Everywhere else, pump all events into the shared storage
            // first.
            #[cfg(not(target_os = "ios"))]
            {
                use winit::platform::pump_events::EventLoopExtPumpEvents;

                struct EventCollector {
                    arrived: bool,
                }

                impl ApplicationHandler for EventCollector {
                    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
                        // Only Android acts on these: its surface dies with the
                        // activity. Desktop delivers a Resumed at startup and
                        // nothing on minimize, so there is nothing to do there.
                        #[cfg(target_os = "android")]
                        LIFECYCLE_EVENTS.with(|events| {
                            events.borrow_mut().push(LifecycleEvent::Resumed);
                        });
                    }

                    #[cfg(target_os = "android")]
                    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
                        LIFECYCLE_EVENTS.with(|events| {
                            events.borrow_mut().push(LifecycleEvent::Suspended);
                        });
                    }

                    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
                        push_lifecycle(LifecycleEvent::LowMemory);
                    }

                    fn window_event(
                        &mut self,
                        _event_loop: &ActiveEventLoop,
                        window_id: winit::window::WindowId,
                        event: WinitWindowEvent,
                    ) {
                        self.arrived = true;
                        collect_window_event(window_id, event);
                    }

                    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
                        self.arrived = true;
                    }

                    fn device_event(
                        &mut self,
                        _event_loop: &ActiveEventLoop,
                        _device_id: winit::event::DeviceId,
                        event: winit::event::DeviceEvent,
                    ) {
                        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                            RAW_MOTION.with(|motion| {
                                let (x, y) = motion.get();
                                motion.set((x + delta.0, y + delta.1));
                            });
                        }
                    }
                }

                EVENT_LOOP.with(|event_loop_cell| {
                    if let Some(ref mut event_loop) = *event_loop_cell.borrow_mut() {
                        let mut collector = EventCollector { arrived: false };
                        let _ = event_loop.pump_app_events(timeout, &mut collector);
                        arrived = collector.arrived;
                    }
                });
            }

            // Told to the app as window events: an app in the background is
            // an iconified window. Android: the native window lives only
            // between Resumed and Suspended, so the surface must die and be
            // reborn with it; while it is gone frames skip.
            #[cfg(not(target_arch = "wasm32"))]
            {
                let lifecycle: Vec<LifecycleEvent> =
                    LIFECYCLE_EVENTS.with(|events| events.borrow_mut().drain(..).collect());
                for event in lifecycle {
                    match event {
                        LifecycleEvent::Suspended => self.suspended = true,
                        LifecycleEvent::Resumed => self.suspended = false,
                        LifecycleEvent::LowMemory => {}
                    }
                    let _ = self.out_events.send(match event {
                        LifecycleEvent::LowMemory => WindowEvent::LowMemory,
                        LifecycleEvent::Suspended => WindowEvent::Iconify(true),
                        LifecycleEvent::Resumed => WindowEvent::Iconify(false),
                    });
                    #[cfg(target_os = "android")]
                    match event {
                        LifecycleEvent::LowMemory => {}
                        LifecycleEvent::Suspended => {
                            self.surface = None;
                        }
                        LifecycleEvent::Resumed => {
                            if self.surface.is_none() {
                                if let Some(window) = &self.window {
                                    let ctxt = Context::get();
                                    let surface = ctxt
                                        .instance
                                        .create_surface(window.clone())
                                        .expect("Failed to recreate surface after resume");
                                    surface.configure(&ctxt.device, &self.surface_config);
                                    self.surface = Some(surface);
                                }
                                // A rotation while away changed the size, and the
                                // depth and MSAA targets have to follow it too.
                                self.fit_to_window();
                            }
                        }
                    }
                }
            }

            // iOS: text the system keyboard produced (window::ios) joins the
            // stream as ordinary typing, so text fields cannot tell a soft
            // keyboard from a hardware one.
            #[cfg(target_os = "ios")]
            for text_event in super::ios::take_text_events() {
                match text_event {
                    super::ios::TextEvent::Char(c) => {
                        let _ = self.out_events.send(WindowEvent::Char(c));
                    }
                    super::ios::TextEvent::Backspace => {
                        let _ = self.out_events.send(WindowEvent::Key(
                            Key::Back,
                            Action::Press,
                            Modifiers::empty(),
                        ));
                        let _ = self.out_events.send(WindowEvent::Key(
                            Key::Back,
                            Action::Release,
                            Modifiers::empty(),
                        ));
                    }
                }
            }

            // Now process only this window's events
            let events = PENDING_WINDOW_EVENTS.with(|storage| {
                storage
                    .borrow_mut()
                    .remove(&self.window_id.unwrap())
                    .unwrap_or_default()
            });
            arrived |= !events.is_empty();

            for event in events {
                match event {
                    PendingEvent::WindowEvent(we) => {
                        self.note(&we);
                        // Winit reports modifiers separately from the events they apply to,
                        // so the mask is stamped in here, once the preceding
                        // `PendingEvent::Modifiers` of this batch have been applied.
                        let modifiers = translate_modifiers(self.modifiers_state);
                        if let Some(we) = stamped(we, modifiers) {
                            let _ = self.out_events.send(we);
                        }
                    }
                    PendingEvent::ButtonState(button, action) => {
                        self.held.set_button(button, action);
                    }
                    PendingEvent::KeyState(key, action) => {
                        self.held.set_key(key, action);
                    }
                    PendingEvent::CursorPos(x, y) => {
                        self.cursor_pos = Some((x, y));
                    }
                    PendingEvent::Modifiers(m) => {
                        self.modifiers_state = m;
                    }
                    PendingEvent::Resize => {
                        self.fit_to_window();
                    }
                }
            }

            let motion = RAW_MOTION.with(|motion| motion.replace((0.0, 0.0)));
            if self.cursor_held.get() && motion != (0.0, 0.0) {
                let _ = self
                    .out_events
                    .send(WindowEvent::MouseMotion(motion.0, motion.1));
                arrived = true;
            }
        }

        // A browser has no pump: it reports a canvas late, so every poll looks.
        #[cfg(target_arch = "wasm32")]
        {
            arrived |= self.fit_to_window();
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Process pending events from web callbacks
            let events: Vec<WindowEvent> = self.pending_events.borrow_mut().drain(..).collect();
            arrived |= !events.is_empty();
            for event in events {
                match &event {
                    WindowEvent::CursorPos(x, y, _) => {
                        self.cursor_pos = Some((*x, *y));
                    }
                    WindowEvent::MouseButton(button, action, _) => {
                        self.held.set_button(*button, *action);
                    }
                    WindowEvent::Key(key, action, _) => {
                        self.held.set_key(*key, *action);
                    }
                    other => self.note(other),
                }
                let _ = self.out_events.send(event);
            }
        }
        arrived
    }

    /// Gets the current surface texture for rendering.
    pub fn get_current_texture(&mut self) -> Option<wgpu::SurfaceTexture> {
        if self.suspended {
            return None;
        }
        let surface = self.surface.as_ref()?;
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some(texture),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // The window may have changed size with no event saying so:
                // take its size before reconfiguring, then retry once.
                self.fit_to_window();
                let surface = self.surface.as_ref()?;
                let ctxt = Context::get();
                surface.configure(&ctxt.device, &self.surface_config);
                match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some(texture),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Copies the surface frame texture into the readback texture for later
    /// reading via [`read_pixels`](Self::read_pixels). A no-op when the
    /// surface has no `COPY_SRC` usage (downlevel GL; see the surface
    /// configuration in `open`) — snapshots of on-screen frames are simply
    /// unavailable there, and attempting the copy is a validation error.
    pub fn copy_frame_to_readback(&self, frame: &wgpu::SurfaceTexture) {
        if !self
            .surface_config
            .usage
            .contains(wgpu::TextureUsages::COPY_SRC)
        {
            return;
        }
        self.copy_texture_to_readback(&frame.texture);
    }

    /// Copies an arbitrary texture into the readback texture used by
    /// [`read_pixels`](Self::read_pixels).
    ///
    /// The source texture must have `COPY_SRC` usage and the same size and
    /// format as the surface. This is how a hidden window, which renders into
    /// an offscreen texture rather than a surface, makes its frame available
    /// to `snap`/`snap_rect`.
    pub fn copy_texture_to_readback(&self, src: &wgpu::Texture) {
        let ctxt = Context::get();
        let mut encoder = ctxt.create_command_encoder(Some("readback_copy_encoder"));

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.readback_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
        );

        ctxt.submit(std::iter::once(encoder.finish()));
    }

    /// Presents the current frame.
    pub fn present(&self, frame: wgpu::SurfaceTexture) {
        Context::get().queue.present(frame);
    }

    /// Reads pixels from the readback texture into the provided buffer.
    /// Returns RGB data (3 bytes per pixel).
    pub fn read_pixels(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        self.begin_read_pixels(x, y, width, height);
        self.finish_read_pixels(out);
    }

    /// Starts an asynchronous GPU→CPU readback of the readback texture and
    /// returns immediately: it enqueues the texture→buffer copy and the buffer
    /// map, but never waits on the GPU. Complete it — typically one frame
    /// later, once the GPU has long finished the copy — with
    /// [`Self::finish_read_pixels`]. Pipelining capture this way hides the
    /// full CPU↔GPU sync that a blocking [`Self::read_pixels`] pays every
    /// frame.
    ///
    /// A second `begin` before the previous readback was finished completes
    /// and discards the previous one first (one readback in flight at a time).
    pub fn begin_read_pixels(&self, x: usize, y: usize, width: usize, height: usize) {
        if self.snap_pending.borrow().is_some() {
            self.finish_read_pixels(&mut Vec::new());
        }

        let ctxt = Context::get();

        // Calculate buffer size with alignment
        // wgpu requires rows to be aligned to 256 bytes
        let bytes_per_pixel = 4; // RGBA or BGRA
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row * height;

        // Reuse the cached staging buffer when it is large enough; (re)create
        // it otherwise. Capturing every frame (video export) then allocates
        // exactly once instead of once per call.
        let mut staging_slot = self.screenshot_staging.borrow_mut();
        let staging_buffer = match staging_slot.take() {
            Some(buffer) if buffer.size() >= buffer_size as u64 => buffer,
            _ => ctxt.create_buffer(&wgpu::BufferDescriptor {
                label: Some("screenshot_staging_buffer"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        };

        // Copy from readback texture to staging buffer
        let mut encoder = ctxt.create_command_encoder(Some("screenshot_copy_encoder"));

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.readback_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(height as u32),
                },
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        let submission = ctxt.submit_indexed(std::iter::once(encoder.finish()));

        // Queue the map; completion is observed in `finish_read_pixels`.
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        *self.snap_pending.borrow_mut() = Some(PendingSnap {
            buffer: staging_buffer,
            rx,
            submission,
            width,
            height,
            padded_bytes_per_row,
        });
    }

    /// Completes a readback started by [`Self::begin_read_pixels`], filling
    /// `out` with RGB data (3 bytes per pixel, rows bottom-to-top like
    /// [`Self::read_pixels`]) and returning the captured `(width, height)`.
    /// Returns `None` (and leaves `out` untouched) when no readback is in
    /// flight. Blocks only until the copy's submission completes — a no-op
    /// when a frame of GPU work has been submitted since the `begin`.
    pub fn finish_read_pixels(&self, out: &mut Vec<u8>) -> Option<(u32, u32)> {
        let PendingSnap {
            buffer: staging_buffer,
            rx,
            submission,
            width,
            height,
            padded_bytes_per_row,
        } = self.snap_pending.borrow_mut().take()?;
        let ctxt = Context::get();

        // Wait only for the copy submission (and, transitively, the work it
        // depends on) instead of polling the device indefinitely.
        let _ = ctxt.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        rx.recv().unwrap().unwrap();

        let bytes_per_pixel = 4; // RGBA or BGRA
        let unpadded_bytes_per_row = width * bytes_per_pixel;

        // Read the data
        let buffer_slice = staging_buffer.slice(..);
        let data = buffer_slice
            .get_mapped_range()
            .expect("staging buffer is mapped");

        // Convert from BGRA/RGBA to RGB and handle row padding
        let rgb_size = width * height * 3;
        out.clear();
        out.reserve(rgb_size);

        let is_bgra = matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        // wgpu has origin at top-left, but we want bottom-left origin for OpenGL compatibility
        // So we read rows in reverse order.
        //
        // The mapped range is uncached (write-combined) memory: scalar reads
        // from it run at ~10 MB/s. memcpy each row into a cached local buffer
        // first, then convert — orders of magnitude faster than indexing the
        // mapped slice per byte.
        let mut row_buf = vec![0u8; unpadded_bytes_per_row];
        for row in (0..height).rev() {
            let row_start = row * padded_bytes_per_row;
            row_buf.copy_from_slice(&data[row_start..row_start + unpadded_bytes_per_row]);
            for px in row_buf.chunks_exact(bytes_per_pixel) {
                if is_bgra {
                    out.extend_from_slice(&[px[2], px[1], px[0]]);
                } else {
                    out.extend_from_slice(&[px[0], px[1], px[2]]);
                }
            }
        }

        drop(data);
        staging_buffer.unmap();
        // Return the buffer to the cache so the next `begin_read_pixels`
        // (or blocking `read_pixels`) reuses it instead of allocating.
        *self.screenshot_staging.borrow_mut() = Some(staging_buffer);
        Some((width as u32, height as u32))
    }

    /// Gets the depth texture view for rendering.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Gets the MSAA texture view if MSAA is enabled.
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }

    /// Gets the sample count for MSAA.
    pub fn sample_count(&self) -> u32 {
        self.sample_count.max(1)
    }

    /// Gets the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// The size of the render surface.
    ///
    /// This returns the configured surface size, which matches the depth texture
    /// and is guaranteed to be consistent with the render targets.
    pub fn size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    /// The current position of the cursor, if known.
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.cursor_pos
    }

    /// The scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.window
            .as_ref()
            .map_or(1.0, |window| window.scale_factor())
    }

    /// The refresh rate of the display the window is on, in hertz, where the
    /// system says.
    pub fn refresh_rate(&self) -> Option<f32> {
        let millihertz = self
            .window
            .as_ref()?
            .current_monitor()?
            .refresh_rate_millihertz()?;
        #[allow(clippy::cast_precision_loss)]
        Some(millihertz as f32 / 1000.0)
    }

    /// Set the window title.
    pub fn set_title(&mut self, title: &str) {
        if let Some(window) = &self.window {
            window.set_title(title);
        }
    }

    /// Set the window icon.
    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        let (width, height) = icon.dimensions();
        let mut rgba = Vec::with_capacity((width * height) as usize * 4);
        for (_, _, pixel) in icon.pixels() {
            rgba.extend_from_slice(&pixel.to_rgba().0);
        }
        let icon = Icon::from_rgba(rgba, width, height).unwrap();
        if let Some(window) = &self.window {
            window.set_window_icon(Some(icon));
        }
    }

    /// Show or hide the platform's on-screen keyboard.
    ///
    /// Android asks the activity (`show_soft_input`, working since
    /// android-activity 0.6.1); iOS goes through the hidden UIKeyInput view in
    /// `window::ios`, since winit has no IME support there. Desktop and web
    /// no-op: hardware keyboards need no summoning and the browser manages
    /// its own.
    pub fn set_keyboard_visible(&self, visible: bool) {
        #[cfg(target_os = "android")]
        {
            KEYBOARD_ASKED.with(|asked| asked.set(visible));
            ANDROID_APP.with(|app| {
                if let Some(app) = &*app.borrow() {
                    if visible {
                        app.show_soft_input(true);
                    } else {
                        app.hide_soft_input(false);
                    }
                }
            });
        }
        #[cfg(target_os = "ios")]
        if let Some(window) = &self.window {
            super::ios::set_keyboard_visible(window, visible);
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        let _ = visible;
    }

    /// Enter or leave borderless fullscreen on the current monitor.
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.change_window(|window| {
            let mode = fullscreen.then_some(winit::window::Fullscreen::Borderless(None));
            window.set_fullscreen(mode);
        });
    }

    /// Run `change` on the window inside a turn of its event loop where kiss3d
    /// pumps one: winit on macOS drops an event raised while no handler is set,
    /// and a resize or a focus change raised between pumps would be lost.
    fn change_window(&self, change: impl FnOnce(&Window)) {
        let Some(window) = &self.window else {
            return;
        };
        #[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
        with_active_loop(|_| change(window));
        #[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
        change(window);
    }

    /// Enter exclusive fullscreen on the current monitor, in its largest
    /// video mode at the highest refresh rate, or leave fullscreen entirely.
    /// Falls back to borderless where the platform offers no video modes, as
    /// the web does.
    pub fn set_exclusive_fullscreen(&self, exclusive: bool) {
        self.change_window(|window| {
            if exclusive {
                window.set_fullscreen(Some(exclusive_mode(window)));
            } else {
                window.set_fullscreen(None);
            }
        });
    }
}

/// The current monitor's largest video mode at its highest refresh rate, or
/// borderless where the platform offers no video modes, as the web does.
fn exclusive_mode(window: &Window) -> winit::window::Fullscreen {
    let mode = window.current_monitor().and_then(|monitor| {
        monitor.video_modes().max_by_key(|mode| {
            let size = mode.size();
            (
                u64::from(size.width) * u64::from(size.height),
                mode.refresh_rate_millihertz(),
            )
        })
    });
    match mode {
        Some(mode) => winit::window::Fullscreen::Exclusive(mode),
        None => winit::window::Fullscreen::Borderless(None),
    }
}

impl WgpuCanvas {
    /// Maximize the window, or restore it.
    pub fn set_maximized(&self, maximized: bool) {
        self.change_window(|window| window.set_maximized(maximized));
    }

    /// Whether the window is currently maximized.
    pub fn is_maximized(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.is_maximized())
    }

    /// Whether the window is currently fullscreen.
    pub fn is_fullscreen(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.fullscreen().is_some())
    }

    /// Files dropped onto the window since the last call, in drop order.
    /// Always empty on the web: browsers deliver file drops to the page, not
    /// the canvas, so a web build wanting them needs DOM listeners instead.
    pub fn take_dropped_files(&self) -> Vec<std::path::PathBuf> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(window_id) = self.window_id else {
                return Vec::new();
            };
            DROPPED_FILES.with(|dropped| {
                let mut dropped = dropped.borrow_mut();
                let mut taken = Vec::new();
                dropped.retain(|(id, path)| {
                    if *id == window_id {
                        taken.push(path.clone());
                        false
                    } else {
                        true
                    }
                });
                taken
            })
        }
        #[cfg(target_arch = "wasm32")]
        Vec::new()
    }

    /// Composed text since the last call; empty until `set_ime_allowed(true)`.
    pub fn take_ime_events(&self) -> Vec<crate::event::ImeEvent> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(window) = &self.window else {
                return Vec::new();
            };
            let id = window.id();
            IME_EVENTS.with(|events| {
                let mut events = events.borrow_mut();
                let (mine, others): (Vec<_>, Vec<_>) =
                    events.drain(..).partition(|(window, _)| *window == id);
                *events = others;
                mine.into_iter().map(|(_, event)| event).collect()
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            Vec::new()
        }
    }

    /// Let the platform compose text through its input method. Not on Android
    /// or iOS, where winit's switch is the on-screen keyboard itself (on iOS
    /// it makes winit's `UIKeyInput` view first responder), which
    /// [`set_keyboard_visible`](Self::set_keyboard_visible) shows with focus.
    pub fn set_ime_allowed(&self, allowed: bool) {
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        if let Some(window) = &self.window {
            window.set_ime_allowed(allowed);
        }
        #[cfg(any(target_os = "android", target_os = "ios"))]
        let _ = allowed;
    }

    /// The window's width and height and NativeActivity's content rect,
    /// `[left, top, right, bottom]`, in pixels.
    #[cfg(target_os = "android")]
    fn android_content(&self) -> Option<(i32, i32, [i32; 4])> {
        let size = self.window.as_ref()?.inner_size();
        let content = ANDROID_APP.with(|app| app.borrow().as_ref().map(|a| a.content_rect()))?;
        Some((
            size.width as i32,
            size.height as i32,
            [content.left, content.top, content.right, content.bottom],
        ))
    }

    /// How many pixels of the window the on-screen keyboard covers, from the
    /// bottom; zero with it down, and zero everywhere but Android and iOS.
    pub fn keyboard_height(&self) -> f32 {
        #[cfg(target_os = "android")]
        {
            let Some((_, height, [_, _, _, bottom])) = self.android_content() else {
                return 0.0;
            };
            android_keyboard(height, height - bottom)
        }
        #[cfg(target_os = "ios")]
        {
            let Some(window) = &self.window else {
                return 0.0;
            };
            (super::ios::keyboard_height(window) * window.scale_factor()) as f32
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            0.0
        }
    }

    /// `[left, top, right, bottom]` insets in pixels; zero on a desktop and
    /// the web.
    pub fn safe_area(&self) -> [f32; 4] {
        // What the content rect leaves uncovered, the bar's share alone at the
        // bottom: the keyboard is `keyboard_height`'s.
        #[cfg(target_os = "android")]
        {
            let Some((width, height, [left, top, right, bottom])) = self.android_content() else {
                return [0.0; 4];
            };
            let covered = height - bottom;
            let bar = android_bar(height).unwrap_or(covered).min(covered);
            [left, top, width - right, bar].map(|inset| inset.max(0) as f32)
        }
        #[cfg(target_os = "ios")]
        {
            let Some(window) = &self.window else {
                return [0.0; 4];
            };
            let scale = window.scale_factor();
            super::ios::safe_area(window).map(|points| (points * scale) as f32)
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            [0.0; 4]
        }
    }

    /// How much larger than standard the reader asked their text to be: iOS
    /// Dynamic Type today, and 1.0 on every platform not yet asked.
    pub fn text_scale(&self) -> f32 {
        #[cfg(target_os = "ios")]
        {
            super::ios::text_scale()
        }
        #[cfg(not(target_os = "ios"))]
        {
            1.0
        }
    }

    /// Hold the cursor for mouse-look: kept inside the window where the
    /// system allows that, else locked in place (macOS, the web). While it is
    /// held the mouse's raw motion arrives as [`WindowEvent::MouseMotion`],
    /// which keeps counting at the window's edge.
    pub fn set_cursor_grab(&self, grab: bool) {
        use winit::window::CursorGrabMode;
        let Some(window) = &self.window else {
            return;
        };
        let held = if grab {
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
                .is_ok()
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            false
        };
        self.cursor_held.set(held);
    }

    /// Whether a grab holds the cursor, so its motion arrives raw.
    pub fn is_cursor_held(&self) -> bool {
        self.cursor_held.get()
    }

    /// Where the text being composed sits, in physical pixels, so the system's
    /// candidate window opens beside it rather than in a corner.
    pub fn set_ime_cursor_area(&self, x: f64, y: f64, width: f64, height: f64) {
        if let Some(window) = &self.window {
            window.set_ime_cursor_area(
                winit::dpi::PhysicalPosition::new(x, y),
                winit::dpi::PhysicalSize::new(width, height),
            );
        }
    }

    /// Set the cursor position.
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        if let Some(window) = &self.window {
            let _ = window.set_cursor_position(winit::dpi::PhysicalPosition::new(x, y));
        }
    }

    /// Toggle the cursor visibility.
    pub fn hide_cursor(&self, hide: bool) {
        if let Some(window) = &self.window {
            window.set_cursor_visible(!hide);
        }
    }

    /// Set the shape the pointer takes over this window.
    pub fn set_cursor_icon(&self, icon: winit::window::CursorIcon) {
        if let Some(window) = &self.window {
            window.set_cursor(icon);
        }
    }

    /// Hide the window.
    pub fn hide(&mut self) {
        self.change_window(|window| window.set_visible(false));
    }

    /// Show the window.
    pub fn show(&mut self) {
        self.change_window(|window| window.set_visible(true));
    }

    /// The state of a mouse button.
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.held.button(button)
    }

    /// The state of a key.
    pub fn get_key(&self, key: Key) -> Action {
        self.held.key(key)
    }

    /// What an event changes in what the canvas holds besides keys and
    /// buttons: losing focus lets go of everything, a pointer outside has no
    /// position.
    fn note(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Focus(false) => self.let_go(),
            WindowEvent::CursorEnter(false) => self.cursor_pos = None,
            _ => {}
        }
    }

    /// Releases everything still held, which no event will once the window
    /// has lost focus: winit sends no releases on macOS, Wayland or the web.
    fn let_go(&mut self) {
        self.modifiers_state = ModifiersState::default();
        for release in self.held.let_go() {
            let _ = self.out_events.send(release);
        }
    }
}

/// The keys and mouse buttons held down, as their events left them.
#[derive(Default)]
struct Held {
    keys: BTreeSet<Key>,
    buttons: BTreeSet<MouseButton>,
}

fn held_state<T: Ord>(held: &BTreeSet<T>, item: &T) -> Action {
    if held.contains(item) {
        Action::Press
    } else {
        Action::Release
    }
}

fn hold<T: Ord>(held: &mut BTreeSet<T>, item: T, action: Action) {
    match action {
        Action::Press => held.insert(item),
        Action::Release => held.remove(&item),
    };
}

impl Held {
    fn key(&self, key: Key) -> Action {
        held_state(&self.keys, &key)
    }

    fn button(&self, button: MouseButton) -> Action {
        held_state(&self.buttons, &button)
    }

    fn set_key(&mut self, key: Key, action: Action) {
        hold(&mut self.keys, key, action);
    }

    fn set_button(&mut self, button: MouseButton, action: Action) {
        hold(&mut self.buttons, button, action);
    }

    /// A release for each thing held, leaving nothing held.
    fn let_go(&mut self) -> Vec<WindowEvent> {
        let buttons = std::mem::take(&mut self.buttons)
            .into_iter()
            .map(|button| WindowEvent::MouseButton(button, Action::Release, Modifiers::empty()));
        let keys = std::mem::take(&mut self.keys)
            .into_iter()
            .map(|key| WindowEvent::Key(key, Action::Release, Modifiers::empty()));
        buttons.chain(keys).collect()
    }
}

// Translate a winit `KeyboardInput` event into the kiss3d-internal stream of
// pending events. `Char` events are only emitted on key press — emitting them
// on release as well caused egui textboxes to receive every character twice
// (see https://github.com/dimforge/kiss3d/issues/380).
#[cfg(not(target_arch = "wasm32"))]
fn translate_keyboard_input(
    key: Key,
    action: Action,
    logical_key: &winit::keyboard::Key,
    text: Option<&str>,
) -> Vec<PendingEvent> {
    let mut events = vec![
        PendingEvent::KeyState(key, action),
        PendingEvent::WindowEvent(WindowEvent::Key(key, action, Modifiers::empty())),
    ];

    if action == Action::Press {
        // Prefer winit's `text` field, falling back to the logical key's
        // character form where it is not populated. A shortcut's letter is
        // dropped by `stamped`, once the modifiers are known.
        let chars = text.or_else(|| match logical_key {
            winit::keyboard::Key::Character(s) => Some(s.as_str()),
            _ => None,
        });
        if let Some(s) = chars {
            for ch in s.chars() {
                events.push(PendingEvent::WindowEvent(WindowEvent::Char(ch)));
            }
        }
    }

    events
}

/// Whether a key pressed under `modifiers` is a shortcut, whose letter is not
/// typed. winit has already taken AltGr out of Control on Windows.
fn is_shortcut(modifiers: Modifiers) -> bool {
    modifiers.intersects(Modifiers::Control | Modifiers::Super)
}

/// `event` as it leaves under `modifiers`, or nothing for a shortcut's letter:
/// on macOS winit's `text` still holds the `c` of ⌘C.
#[cfg(not(target_arch = "wasm32"))]
fn stamped(event: WindowEvent, modifiers: Modifiers) -> Option<WindowEvent> {
    if matches!(event, WindowEvent::Char(_)) && is_shortcut(modifiers) {
        return None;
    }
    Some(event.with_modifiers(modifiers))
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_action(action: winit::event::ElementState) -> Action {
    use winit::event::ElementState;
    match action {
        ElementState::Pressed => Action::Press,
        ElementState::Released => Action::Release,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_modifiers(modifiers: ModifiersState) -> Modifiers {
    let mut res = Modifiers::empty();
    if modifiers.shift_key() {
        res.insert(Modifiers::Shift)
    }
    if modifiers.control_key() {
        res.insert(Modifiers::Control)
    }
    if modifiers.alt_key() {
        res.insert(Modifiers::Alt)
    }
    if modifiers.super_key() {
        res.insert(Modifiers::Super)
    }
    res
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_mouse_button(button: winit::event::MouseButton) -> MouseButton {
    match button {
        winit::event::MouseButton::Left => MouseButton::Button1,
        winit::event::MouseButton::Right => MouseButton::Button2,
        winit::event::MouseButton::Middle => MouseButton::Button3,
        winit::event::MouseButton::Back => MouseButton::Button4,
        winit::event::MouseButton::Forward => MouseButton::Button5,
        winit::event::MouseButton::Other(_) => MouseButton::Button6,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn translate_key(physical_key: PhysicalKey) -> Key {
    if let PhysicalKey::Code(key_code) = physical_key {
        match key_code {
            KeyCode::Digit1 => Key::Key1,
            KeyCode::Digit2 => Key::Key2,
            KeyCode::Digit3 => Key::Key3,
            KeyCode::Digit4 => Key::Key4,
            KeyCode::Digit5 => Key::Key5,
            KeyCode::Digit6 => Key::Key6,
            KeyCode::Digit7 => Key::Key7,
            KeyCode::Digit8 => Key::Key8,
            KeyCode::Digit9 => Key::Key9,
            KeyCode::Digit0 => Key::Key0,
            KeyCode::KeyA => Key::A,
            KeyCode::KeyB => Key::B,
            KeyCode::KeyC => Key::C,
            KeyCode::KeyD => Key::D,
            KeyCode::KeyE => Key::E,
            KeyCode::KeyF => Key::F,
            KeyCode::KeyG => Key::G,
            KeyCode::KeyH => Key::H,
            KeyCode::KeyI => Key::I,
            KeyCode::KeyJ => Key::J,
            KeyCode::KeyK => Key::K,
            KeyCode::KeyL => Key::L,
            KeyCode::KeyM => Key::M,
            KeyCode::KeyN => Key::N,
            KeyCode::KeyO => Key::O,
            KeyCode::KeyP => Key::P,
            KeyCode::KeyQ => Key::Q,
            KeyCode::KeyR => Key::R,
            KeyCode::KeyS => Key::S,
            KeyCode::KeyT => Key::T,
            KeyCode::KeyU => Key::U,
            KeyCode::KeyV => Key::V,
            KeyCode::KeyW => Key::W,
            KeyCode::KeyX => Key::X,
            KeyCode::KeyY => Key::Y,
            KeyCode::KeyZ => Key::Z,
            KeyCode::Escape => Key::Escape,
            KeyCode::F1 => Key::F1,
            KeyCode::F2 => Key::F2,
            KeyCode::F3 => Key::F3,
            KeyCode::F4 => Key::F4,
            KeyCode::F5 => Key::F5,
            KeyCode::F6 => Key::F6,
            KeyCode::F7 => Key::F7,
            KeyCode::F8 => Key::F8,
            KeyCode::F9 => Key::F9,
            KeyCode::F10 => Key::F10,
            KeyCode::F11 => Key::F11,
            KeyCode::F12 => Key::F12,
            KeyCode::F13 => Key::F13,
            KeyCode::F14 => Key::F14,
            KeyCode::F15 => Key::F15,
            KeyCode::F16 => Key::F16,
            KeyCode::F17 => Key::F17,
            KeyCode::F18 => Key::F18,
            KeyCode::F19 => Key::F19,
            KeyCode::F20 => Key::F20,
            KeyCode::F21 => Key::F21,
            KeyCode::F22 => Key::F22,
            KeyCode::F23 => Key::F23,
            KeyCode::F24 => Key::F24,
            KeyCode::PrintScreen => Key::Snapshot,
            KeyCode::ScrollLock => Key::Scroll,
            KeyCode::Pause => Key::Pause,
            KeyCode::Insert => Key::Insert,
            KeyCode::Home => Key::Home,
            KeyCode::Delete => Key::Delete,
            KeyCode::End => Key::End,
            KeyCode::PageDown => Key::PageDown,
            KeyCode::PageUp => Key::PageUp,
            KeyCode::ArrowLeft => Key::Left,
            KeyCode::ArrowUp => Key::Up,
            KeyCode::ArrowRight => Key::Right,
            KeyCode::ArrowDown => Key::Down,
            KeyCode::Backspace => Key::Back,
            KeyCode::Enter => Key::Return,
            KeyCode::Space => Key::Space,
            KeyCode::NumLock => Key::Numlock,
            KeyCode::Numpad0 => Key::Numpad0,
            KeyCode::Numpad1 => Key::Numpad1,
            KeyCode::Numpad2 => Key::Numpad2,
            KeyCode::Numpad3 => Key::Numpad3,
            KeyCode::Numpad4 => Key::Numpad4,
            KeyCode::Numpad5 => Key::Numpad5,
            KeyCode::Numpad6 => Key::Numpad6,
            KeyCode::Numpad7 => Key::Numpad7,
            KeyCode::Numpad8 => Key::Numpad8,
            KeyCode::Numpad9 => Key::Numpad9,
            KeyCode::NumpadAdd => Key::Add,
            KeyCode::Quote => Key::Apostrophe,
            KeyCode::Backslash => Key::Backslash,
            KeyCode::NumpadClear => Key::NumpadEquals,
            KeyCode::Comma => Key::Comma,
            KeyCode::Convert => Key::Convert,
            KeyCode::NumpadDecimal => Key::Decimal,
            KeyCode::NumpadDivide => Key::Divide,
            KeyCode::NumpadMultiply => Key::Multiply,
            KeyCode::Equal => Key::Equals,
            KeyCode::Backquote => Key::Grave,
            KeyCode::KanaMode => Key::Kana,
            KeyCode::AltLeft => Key::LAlt,
            KeyCode::BracketLeft => Key::LBracket,
            KeyCode::ControlLeft => Key::LControl,
            KeyCode::ShiftLeft => Key::LShift,
            KeyCode::SuperLeft => Key::LWin,
            KeyCode::LaunchMail => Key::Mail,
            KeyCode::MediaSelect => Key::MediaSelect,
            KeyCode::MediaStop => Key::MediaStop,
            KeyCode::Minus => Key::Minus,
            KeyCode::AudioVolumeMute => Key::Mute,
            KeyCode::BrowserForward => Key::NavigateForward,
            KeyCode::BrowserBack => Key::NavigateBackward,
            KeyCode::MediaTrackNext => Key::NextTrack,
            KeyCode::NonConvert => Key::NoConvert,
            KeyCode::NumpadComma => Key::NumpadComma,
            KeyCode::NumpadEnter => Key::NumpadEnter,
            KeyCode::IntlBackslash => Key::OEM102,
            KeyCode::Period => Key::Period,
            KeyCode::MediaPlayPause => Key::PlayPause,
            KeyCode::Power => Key::Power,
            KeyCode::MediaTrackPrevious => Key::PrevTrack,
            KeyCode::AltRight => Key::RAlt,
            KeyCode::BracketRight => Key::RBracket,
            KeyCode::ControlRight => Key::RControl,
            KeyCode::ShiftRight => Key::RShift,
            KeyCode::SuperRight => Key::RWin,
            KeyCode::Semicolon => Key::Semicolon,
            KeyCode::Slash => Key::Slash,
            KeyCode::Sleep => Key::Sleep,
            KeyCode::NumpadSubtract => Key::Subtract,
            KeyCode::Tab => Key::Tab,
            KeyCode::AudioVolumeDown => Key::VolumeDown,
            KeyCode::AudioVolumeUp => Key::VolumeUp,
            KeyCode::WakeUp => Key::Wake,
            KeyCode::BrowserHome => Key::WebHome,
            KeyCode::BrowserRefresh => Key::WebRefresh,
            KeyCode::BrowserSearch => Key::WebSearch,
            KeyCode::IntlYen => Key::Yen,
            KeyCode::Copy => Key::Copy,
            KeyCode::Paste => Key::Paste,
            KeyCode::Cut => Key::Cut,
            KeyCode::CapsLock => Key::Capital,
            KeyCode::IntlRo => Key::AbntC1,
            KeyCode::ContextMenu => Key::Apps,
            KeyCode::NumpadEqual => Key::NumpadEquals,
            _ => Key::Unknown,
        }
    } else {
        Key::Unknown
    }
}

/// Reads the modifier keys held during a DOM event.
///
/// The web backend has no equivalent of winit's `ModifiersChanged`; each DOM
/// input event carries its own modifier state instead.
#[cfg(target_arch = "wasm32")]
trait WebModifiers {
    fn modifiers(&self) -> Modifiers;
}

/// Builds a mask from a `getModifierState`-style query.
///
/// See <https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/getModifierState>.
#[cfg(any(target_arch = "wasm32", test))]
fn web_modifiers(state: impl Fn(&str) -> bool) -> Modifiers {
    let mut res = Modifiers::empty();
    if state("Shift") {
        res.insert(Modifiers::Shift)
    }
    if state("Control") {
        res.insert(Modifiers::Control)
    }
    if state("Alt") {
        res.insert(Modifiers::Alt)
    }
    // The web calls the Super key "Meta" (winit 0.31 renames it to Meta too).
    if state("Meta") {
        res.insert(Modifiers::Super)
    }
    res
}

#[cfg(target_arch = "wasm32")]
impl WebModifiers for web_sys::KeyboardEvent {
    fn modifiers(&self) -> Modifiers {
        web_modifiers(|key| self.get_modifier_state(key))
    }
}

/// Also covers `PointerEvent` and `WheelEvent`, which deref to `MouseEvent`.
#[cfg(target_arch = "wasm32")]
impl WebModifiers for web_sys::MouseEvent {
    fn modifiers(&self) -> Modifiers {
        web_modifiers(|key| self.get_modifier_state(key))
    }
}

/// `TouchEvent` has no `getModifierState`, so the individual flags are read instead.
#[cfg(target_arch = "wasm32")]
impl WebModifiers for web_sys::TouchEvent {
    fn modifiers(&self) -> Modifiers {
        web_modifiers(|key| match key {
            "Shift" => self.shift_key(),
            "Control" => self.ctrl_key(),
            "Alt" => self.alt_key(),
            _ => self.meta_key(),
        })
    }
}

/// Where a pointer event landed on the canvas, in canvas pixels.
#[cfg(target_arch = "wasm32")]
fn canvas_point(canvas: &web_sys::HtmlCanvasElement, event: &web_sys::MouseEvent) -> (f64, f64) {
    canvas_point_at(
        canvas,
        f64::from(event.client_x()),
        f64::from(event.client_y()),
    )
}

/// A page position on the canvas, in canvas pixels: the CSS offset from its
/// corner, scaled by how much the canvas is stretched.
#[cfg(target_arch = "wasm32")]
fn canvas_point_at(
    canvas: &web_sys::HtmlCanvasElement,
    client_x: f64,
    client_y: f64,
) -> (f64, f64) {
    let rect = canvas.get_bounding_client_rect();
    let css_x = client_x - rect.left();
    let css_y = client_y - rect.top();
    let scale_x = canvas.width() as f64 / rect.width();
    let scale_y = canvas.height() as f64 / rect.height();
    (css_x * scale_x, css_y * scale_y)
}

#[cfg(target_arch = "wasm32")]
fn translate_web_mouse_button(button: i16) -> MouseButton {
    match button {
        0 => MouseButton::Button1, // Left
        1 => MouseButton::Button3, // Middle
        2 => MouseButton::Button2, // Right
        3 => MouseButton::Button4,
        4 => MouseButton::Button5,
        _ => MouseButton::Button1,
    }
}

/// Whether the browser runs on an Apple platform, where ⌘ is the command
/// key. A page can only tell from the user agent; iPadOS reports itself as a
/// Mac, which is right, since its keyboards carry ⌘ too.
#[cfg(target_arch = "wasm32")]
pub(crate) fn apple_platform() -> bool {
    thread_local! {
        static APPLE: bool = web_sys::window()
            .and_then(|w| w.navigator().user_agent().ok())
            .is_some_and(|ua| ua.contains("Mac"));
    }
    APPLE.with(|apple| *apple)
}

thread_local! {
    /// Keys pressed while ⌘ was held on an Apple platform. macOS never sends
    /// their keyup, so their release is synthesized when ⌘ goes up.
    #[cfg(target_arch = "wasm32")]
    static SUPER_HELD_KEYS: RefCell<Vec<Key>> = const { RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "wasm32")]
fn is_modifier_key(key: Key) -> bool {
    matches!(
        key,
        Key::LShift
            | Key::RShift
            | Key::LControl
            | Key::RControl
            | Key::LAlt
            | Key::RAlt
            | Key::LWin
            | Key::RWin
    )
}

#[cfg(any(target_arch = "wasm32", test))]
fn translate_web_key(code: &str) -> Key {
    match code {
        "Digit1" => Key::Key1,
        "Digit2" => Key::Key2,
        "Digit3" => Key::Key3,
        "Digit4" => Key::Key4,
        "Digit5" => Key::Key5,
        "Digit6" => Key::Key6,
        "Digit7" => Key::Key7,
        "Digit8" => Key::Key8,
        "Digit9" => Key::Key9,
        "Digit0" => Key::Key0,
        "KeyA" => Key::A,
        "KeyB" => Key::B,
        "KeyC" => Key::C,
        "KeyD" => Key::D,
        "KeyE" => Key::E,
        "KeyF" => Key::F,
        "KeyG" => Key::G,
        "KeyH" => Key::H,
        "KeyI" => Key::I,
        "KeyJ" => Key::J,
        "KeyK" => Key::K,
        "KeyL" => Key::L,
        "KeyM" => Key::M,
        "KeyN" => Key::N,
        "KeyO" => Key::O,
        "KeyP" => Key::P,
        "KeyQ" => Key::Q,
        "KeyR" => Key::R,
        "KeyS" => Key::S,
        "KeyT" => Key::T,
        "KeyU" => Key::U,
        "KeyV" => Key::V,
        "KeyW" => Key::W,
        "KeyX" => Key::X,
        "KeyY" => Key::Y,
        "KeyZ" => Key::Z,
        "Escape" => Key::Escape,
        "F1" => Key::F1,
        "F2" => Key::F2,
        "F3" => Key::F3,
        "F4" => Key::F4,
        "F5" => Key::F5,
        "F6" => Key::F6,
        "F7" => Key::F7,
        "F8" => Key::F8,
        "F9" => Key::F9,
        "F10" => Key::F10,
        "F11" => Key::F11,
        "F12" => Key::F12,
        "Insert" => Key::Insert,
        "Home" => Key::Home,
        "Delete" => Key::Delete,
        "End" => Key::End,
        "PageDown" => Key::PageDown,
        "PageUp" => Key::PageUp,
        "ArrowLeft" => Key::Left,
        "ArrowUp" => Key::Up,
        "ArrowRight" => Key::Right,
        "ArrowDown" => Key::Down,
        "Backspace" => Key::Back,
        "Enter" => Key::Return,
        "Space" => Key::Space,
        "NumLock" => Key::Numlock,
        "Numpad0" => Key::Numpad0,
        "Numpad1" => Key::Numpad1,
        "Numpad2" => Key::Numpad2,
        "Numpad3" => Key::Numpad3,
        "Numpad4" => Key::Numpad4,
        "Numpad5" => Key::Numpad5,
        "Numpad6" => Key::Numpad6,
        "Numpad7" => Key::Numpad7,
        "Numpad8" => Key::Numpad8,
        "Numpad9" => Key::Numpad9,
        "NumpadAdd" => Key::Add,
        "NumpadSubtract" => Key::Subtract,
        "NumpadMultiply" => Key::Multiply,
        "NumpadDivide" => Key::Divide,
        "NumpadDecimal" => Key::Decimal,
        "NumpadEnter" => Key::NumpadEnter,
        "Quote" => Key::Apostrophe,
        "Backslash" => Key::Backslash,
        "Comma" => Key::Comma,
        "Equal" => Key::Equals,
        "Backquote" => Key::Grave,
        "AltLeft" => Key::LAlt,
        "BracketLeft" => Key::LBracket,
        "ControlLeft" => Key::LControl,
        "ShiftLeft" => Key::LShift,
        "MetaLeft" => Key::LWin,
        "Minus" => Key::Minus,
        "Period" => Key::Period,
        "AltRight" => Key::RAlt,
        "BracketRight" => Key::RBracket,
        "ControlRight" => Key::RControl,
        "ShiftRight" => Key::RShift,
        "MetaRight" => Key::RWin,
        "Semicolon" => Key::Semicolon,
        "Slash" => Key::Slash,
        "Tab" => Key::Tab,
        "F13" => Key::F13,
        "F14" => Key::F14,
        "F15" => Key::F15,
        "F16" => Key::F16,
        "F17" => Key::F17,
        "F18" => Key::F18,
        "F19" => Key::F19,
        "F20" => Key::F20,
        "F21" => Key::F21,
        "F22" => Key::F22,
        "F23" => Key::F23,
        "F24" => Key::F24,
        "CapsLock" => Key::Capital,
        "ScrollLock" => Key::Scroll,
        "PrintScreen" => Key::Snapshot,
        "Pause" => Key::Pause,
        "IntlBackslash" => Key::OEM102,
        "IntlYen" => Key::Yen,
        "IntlRo" => Key::AbntC1,
        "ContextMenu" => Key::Apps,
        "NumpadEqual" => Key::NumpadEquals,
        "NumpadComma" => Key::NumpadComma,
        "MediaPlayPause" => Key::PlayPause,
        "MediaStop" => Key::MediaStop,
        "MediaTrackNext" => Key::NextTrack,
        "MediaTrackPrevious" => Key::PrevTrack,
        "MediaSelect" => Key::MediaSelect,
        "AudioVolumeMute" => Key::Mute,
        "AudioVolumeDown" => Key::VolumeDown,
        "AudioVolumeUp" => Key::VolumeUp,
        "BrowserBack" => Key::NavigateBackward,
        "BrowserForward" => Key::NavigateForward,
        "BrowserHome" => Key::WebHome,
        "BrowserRefresh" => Key::WebRefresh,
        "BrowserSearch" => Key::WebSearch,
        "LaunchMail" => Key::Mail,
        "Power" => Key::Power,
        "Sleep" => Key::Sleep,
        "WakeUp" => Key::Wake,
        "KanaMode" => Key::Kana,
        "Convert" => Key::Convert,
        "NonConvert" => Key::NoConvert,
        "Copy" => Key::Copy,
        "Paste" => Key::Paste,
        "Cut" => Key::Cut,
        _ => Key::Unknown,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use winit::dpi::LogicalSize;
    use winit::keyboard::{Key as WinitKey, NamedKey, SmolStr};

    #[test]
    fn the_pointer_leaving_and_entering_the_window_is_told() {
        let window = winit::window::WindowId::dummy();
        let device_id = winit::event::DeviceId::dummy();
        collect_window_event(window, WinitWindowEvent::CursorLeft { device_id });
        collect_window_event(window, WinitWindowEvent::CursorEntered { device_id });
        let pending = PENDING_WINDOW_EVENTS
            .with(|storage| storage.borrow_mut().remove(&window).unwrap_or_default());
        let told: Vec<bool> = pending
            .iter()
            .filter_map(|event| match event {
                PendingEvent::WindowEvent(WindowEvent::CursorEnter(inside)) => Some(*inside),
                _ => None,
            })
            .collect();
        assert_eq!(told, [false, true]);
    }

    #[test]
    fn losing_focus_releases_every_held_key_and_button_once() {
        let mut held = Held::default();
        held.set_key(Key::W, Action::Press);
        held.set_key(Key::LWin, Action::Press);
        held.set_key(Key::A, Action::Press);
        held.set_key(Key::A, Action::Release);
        held.set_button(MouseButton::Button1, Action::Press);

        let released = held.let_go();
        assert_eq!(released.len(), 3, "{released:?}");
        assert!(released.iter().all(|event| matches!(
            event,
            WindowEvent::Key(_, Action::Release, _)
                | WindowEvent::MouseButton(_, Action::Release, _)
        )));
        assert_eq!(held.key(Key::W), Action::Release);
        assert_eq!(held.key(Key::LWin), Action::Release);
        assert_eq!(held.button(MouseButton::Button1), Action::Release);
        assert!(
            held.let_go().is_empty(),
            "a second loss has nothing to release"
        );
    }

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    #[test]
    fn a_window_too_big_for_its_monitor_restores_to_what_the_monitor_holds() {
        let laptop = LogicalSize::new(1512.0, 982.0);
        assert!(held_size(LogicalSize::new(1280.0, 800.0), laptop).is_none());
        assert_eq!(
            held_size(LogicalSize::new(1600.0, 1000.0), laptop),
            Some(laptop)
        );
        assert_eq!(
            held_size(LogicalSize::new(1400.0, 1080.0), laptop),
            Some(LogicalSize::new(1400.0, 982.0)),
            "only the side that does not fit is cut"
        );
        assert!(
            held_size(LogicalSize::new(1600.0, 1000.0), LogicalSize::new(0.0, 0.0)).is_none(),
            "a monitor that reports no size leaves the window as asked"
        );
    }

    fn char_events(events: &[PendingEvent]) -> Vec<char> {
        events
            .iter()
            .filter_map(|e| match e {
                PendingEvent::WindowEvent(WindowEvent::Char(c)) => Some(*c),
                _ => None,
            })
            .collect()
    }

    /// A height first seen with the keyboard asked for borrows the bar's share
    /// from the height before it, so a turn while typing is not all bar.
    #[test]
    fn a_rotation_while_typing_still_reads_the_keyboard() {
        assert_eq!(android_keyboard(1920, 60), 0.0, "upright, the bar alone");
        KEYBOARD_ASKED.with(|asked| asked.set(true));
        assert_eq!(android_keyboard(1920, 660), 600.0);
        assert_eq!(
            android_keyboard(1080, 460),
            400.0,
            "turned with the keyboard up, the cover is not all bar"
        );
        KEYBOARD_ASKED.with(|asked| asked.set(false));
        assert_eq!(android_keyboard(1080, 60), 0.0, "down again");
        assert_eq!(
            android_keyboard(1920, 660),
            600.0,
            "and the first height is remembered"
        );
    }

    /// The navigation bar covers the window whether or not the keyboard is up,
    /// so the first cover seen at a height is taken as the bar alone and only
    /// what a later one adds is the keyboard.
    #[test]
    fn the_android_keyboard_is_what_covers_more_than_the_navigation_bar() {
        // 1080-tall window whose navigation bar takes the bottom 60.
        assert_eq!(android_keyboard(1080, 60), 0.0, "the bar alone is not it");
        assert_eq!(android_keyboard(1080, 660), 600.0, "the keyboard is up");
        assert_eq!(android_keyboard(1080, 60), 0.0, "and down again");
        // A rotation: the new height is measured from scratch, so its first
        // cover is the bar again rather than 600 pixels of phantom keyboard.
        assert_eq!(android_keyboard(1920, 40), 0.0);
        assert_eq!(android_keyboard(1920, 440), 400.0);
    }

    /// The editor's chords are punctuation and function keys, and both
    /// backends read the physical code, so a shifted `\` is still `Backslash`.
    #[test]
    fn a_chord_key_arrives_as_its_physical_code() {
        for (code, web, want) in [
            (KeyCode::Backslash, "Backslash", Key::Backslash),
            (KeyCode::Comma, "Comma", Key::Comma),
            (KeyCode::Equal, "Equal", Key::Equals),
            (KeyCode::Slash, "Slash", Key::Slash),
            (KeyCode::F5, "F5", Key::F5),
        ] {
            assert_eq!(translate_key(PhysicalKey::Code(code)), want);
            assert_eq!(translate_web_key(web), want);
        }
    }

    /// Both keys reach the event as themselves. Folding them into one command
    /// modifier is the egui layer's job, in `egui_modifiers`.
    #[test]
    fn control_and_super_both_reach_the_event() {
        let held = translate_modifiers(ModifiersState::SUPER | ModifiersState::SHIFT);
        assert!(held.contains(Modifiers::Super) && held.contains(Modifiers::Shift));
        assert!(translate_modifiers(ModifiersState::CONTROL).contains(Modifiers::Control));
        let web = web_modifiers(|key| key == "Meta" || key == "Shift");
        assert!(web.contains(Modifiers::Super) && web.contains(Modifiers::Shift));
    }

    #[test]
    fn a_shortcut_types_no_letter_and_still_reaches_the_keys() {
        for held in [
            Modifiers::Super,
            Modifiers::Control,
            Modifiers::Super | Modifiers::Shift,
        ] {
            assert!(stamped(WindowEvent::Char('c'), held).is_none());
            let key = stamped(
                WindowEvent::Key(Key::C, Action::Press, Modifiers::empty()),
                held,
            );
            assert!(matches!(
                key,
                Some(WindowEvent::Key(Key::C, Action::Press, mask)) if mask == held
            ));
        }
    }

    #[test]
    fn shift_and_option_still_type() {
        for held in [Modifiers::empty(), Modifiers::Shift, Modifiers::Alt] {
            assert!(matches!(
                stamped(WindowEvent::Char('c'), held),
                Some(WindowEvent::Char('c'))
            ));
        }
    }

    // Regression test for https://github.com/dimforge/kiss3d/issues/380:
    // a single key press/release cycle must only produce a single Char event,
    // otherwise egui textboxes (and other text consumers) insert each typed
    // character twice.
    #[test]
    fn issue_380_press_release_emits_single_char() {
        let logical = WinitKey::Character(SmolStr::new("a"));

        let press = translate_keyboard_input(Key::A, Action::Press, &logical, Some("a"));
        let release = translate_keyboard_input(Key::A, Action::Release, &logical, Some("a"));

        // Before the fix, both press and release emitted a Char event,
        // so a single keystroke produced "aa" in egui textboxes.
        assert_eq!(char_events(&press), vec!['a']);
        assert_eq!(char_events(&release), Vec::<char>::new());

        // Key + KeyState events must still fire on both press and release,
        // so consumers that track key state (e.g. cameras) keep working.
        assert!(press.iter().any(|e| matches!(
            e,
            PendingEvent::WindowEvent(WindowEvent::Key(Key::A, Action::Press, _))
        )));
        assert!(release.iter().any(|e| matches!(
            e,
            PendingEvent::WindowEvent(WindowEvent::Key(Key::A, Action::Release, _))
        )));
        assert!(press
            .iter()
            .any(|e| matches!(e, PendingEvent::KeyState(Key::A, Action::Press))));
        assert!(release
            .iter()
            .any(|e| matches!(e, PendingEvent::KeyState(Key::A, Action::Release))));
    }

    // Named keys (Enter, Shift, ...) without text must not generate Char events
    // on either press or release.
    #[test]
    fn named_key_emits_no_char() {
        let logical = WinitKey::Named(NamedKey::Shift);
        let press = translate_keyboard_input(Key::LShift, Action::Press, &logical, None);
        let release = translate_keyboard_input(Key::LShift, Action::Release, &logical, None);
        assert_eq!(char_events(&press), Vec::<char>::new());
        assert_eq!(char_events(&release), Vec::<char>::new());
    }

    // If winit reports `text` (the canonical source), use it verbatim — it
    // already accounts for shift/altgr layouts.
    #[test]
    fn uses_text_field_when_provided() {
        let logical = WinitKey::Character(SmolStr::new("a"));
        let press = translate_keyboard_input(Key::A, Action::Press, &logical, Some("A"));
        assert_eq!(char_events(&press), vec!['A']);
    }

    // Some platforms may leave `text` empty for a printable key; fall back to
    // the logical key's character form so users still get text input.
    #[test]
    fn falls_back_to_logical_key_when_text_missing() {
        let logical = WinitKey::Character(SmolStr::new("z"));
        let press = translate_keyboard_input(Key::Z, Action::Press, &logical, None);
        assert_eq!(char_events(&press), vec!['z']);
    }
}

/// The canvas's size in device pixels from its own layout. Winit learns the
/// size from a `ResizeObserver`, which the browser delivers only when it next
/// paints the page, and `inner_size` is zero until then; a page it has not
/// painted yet has laid the canvas out all the same. The first frames went
/// to a 1 by 1 surface before this, and a browser given a 1 by 1 present had
/// nothing to paint again, so a page nothing touched stayed on the clear
/// colour until the pointer moved (2026-10-20).
#[cfg(target_arch = "wasm32")]
fn canvas_layout_size(window: &winit::window::Window) -> Option<winit::dpi::PhysicalSize<u32>> {
    use winit::platform::web::WindowExtWebSys;
    let canvas = window.canvas()?;
    let scale = window.scale_factor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let width = (f64::from(canvas.client_width()) * scale).round() as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let height = (f64::from(canvas.client_height()) * scale).round() as u32;
    (width > 0 && height > 0).then_some(winit::dpi::PhysicalSize::new(width, height))
}

/// The device pixels the window's surface covers. iOS's `inner_size` is the
/// safe area; the surface covers the whole window, which `outer_size` is.
fn drawable_size(window: &winit::window::Window) -> winit::dpi::PhysicalSize<u32> {
    #[cfg(target_arch = "wasm32")]
    {
        measured_size(window)
    }
    #[cfg(target_os = "ios")]
    {
        window.outer_size()
    }
    #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
    {
        window.inner_size()
    }
}

/// What winit says the window is, or what the canvas measures while winit has
/// not heard from its observer yet.
#[cfg(target_arch = "wasm32")]
fn measured_size(window: &winit::window::Window) -> winit::dpi::PhysicalSize<u32> {
    let size = window.inner_size();
    if size.width > 0 && size.height > 0 {
        size
    } else {
        canvas_layout_size(window).unwrap_or(size)
    }
}
