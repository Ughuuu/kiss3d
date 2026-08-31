//! iOS support: the inverted main loop.
//!
//! iOS forbids the pump model the other native platforms use. winit's
//! `run_app` calls `UIApplicationMain`, which owns the main thread and never
//! returns, and `pump_events` does not exist on the platform. So the roles
//! flip, the same way they do on the web: winit drives, and the user's
//! `async fn main` becomes a future polled once per loop turn. The renderer's
//! per-frame [`next_frame`]`().await` is the seam that makes the user's
//! `while window.render().await` loop yield back to UIKit between frames —
//! the iOS analogue of the wasm path awaiting `requestAnimationFrame`.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::task::{Context as TaskContext, Poll, Waker};

use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

thread_local! {
    // The ActiveEventLoop of the winit callback currently on the stack. Only
    // valid inside that callback: IosApp sets it before polling the app
    // future and clears it after, and `create_window` reads it when the
    // future opens a window mid-poll.
    static ACTIVE: Cell<*const ActiveEventLoop> = const { Cell::new(ptr::null()) };
    // Turn counter that resumes stalled `next_frame` futures: a frame that
    // yielded in turn N completes its await in turn N+1.
    static FRAME: Cell<u64> = const { Cell::new(0) };
}

/// Creates a winit window from inside the running app future.
///
/// # Panics
/// Outside a `run_ios` poll — on iOS a window can only be created while
/// winit's loop is on the stack, which `#[kiss3d::main]` guarantees.
pub(crate) fn create_window(attrs: WindowAttributes) -> Window {
    ACTIVE.with(|active| {
        let ptr = active.get();
        assert!(
            !ptr.is_null(),
            "on iOS, windows can only be created inside #[kiss3d::main]'s event loop"
        );
        // Valid for the duration of the callback that set it; this is only
        // reachable from the app future, which is polled inside one.
        let event_loop = unsafe { &*ptr };
        event_loop
            .create_window(attrs)
            .expect("Failed to create window")
    })
}

/// Completes on the loop turn after its first poll. One await per rendered
/// frame is what paces the app future to the display.
pub(crate) async fn next_frame() {
    struct NextFrame(u64);

    impl Future for NextFrame {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<()> {
            FRAME.with(|frame| {
                if frame.get() == self.0 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
        }
    }

    NextFrame(FRAME.with(Cell::get)).await;
}

struct IosApp {
    // None once the future completes; the loop is asked to exit then, but
    // UIApplicationMain does not oblige, so this also guards re-polling.
    fut: Option<Pin<Box<dyn Future<Output = ()>>>>,
    // The future first runs in `resumed` — UIKit forbids UI work before the
    // application is active — and `about_to_wait` fires earlier than that.
    started: bool,
}

impl IosApp {
    fn poll(&mut self, event_loop: &ActiveEventLoop) {
        let Some(fut) = self.fut.as_mut() else {
            return;
        };
        ACTIVE.with(|active| active.set(ptr::from_ref(event_loop)));
        // A no-op waker: the loop re-polls every turn regardless, which is
        // exactly the pacing `next_frame` encodes.
        let mut cx = TaskContext::from_waker(Waker::noop());
        let done = fut.as_mut().poll(&mut cx).is_ready();
        ACTIVE.with(|active| active.set(ptr::null()));
        if done {
            self.fut = None;
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for IosApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        self.started = true;
        self.poll(event_loop);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        super::wgpu_canvas::collect_window_event(window_id, event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.started {
            return;
        }
        FRAME.with(|frame| frame.set(frame.get() + 1));
        self.poll(event_loop);
    }
}

/// Runs a kiss3d app future under winit's iOS main loop. `#[kiss3d::main]`
/// expands to a call to this. In practice it never returns: once started,
/// `UIApplicationMain` owns the process.
pub fn run_ios(fut: impl Future<Output = ()> + 'static) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = IosApp {
        fut: Some(Box::pin(fut)),
        started: false,
    };
    let _ = event_loop.run_app(&mut app);
}
