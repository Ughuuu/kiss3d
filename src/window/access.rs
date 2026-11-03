//! A screen reader's way into a window: AccessKit's adapter for the platform,
//! for a window opened with [`CanvasSetup::accessible`].
//!
//! The adapter has to exist before the window is first shown, so
//! `WgpuCanvas::open` makes it there, and it hears every window event, so it
//! lives beside the loop that delivers them. It answers the platform by itself:
//! until a reader asks, a frame looks at an empty queue and nothing else runs.
//!
//! A reader's requests arrive on whichever thread the platform calls from, so
//! they wait in a queue and wake the loop, and the app reads them between frames
//! through [`Window::take_access_events`]. What the app says back goes through
//! [`Window::update_access`].
//!
//! No adapter on the web (a page would need a DOM to mirror the tree into) or on
//! Android (AccessKit's reaches for `GameActivity`, and kiss3d runs on
//! `NativeActivity`): there a window hears nothing and an update goes nowhere.
//!
//! [`CanvasSetup::accessible`]: super::CanvasSetup::accessible
//! [`Window::take_access_events`]: super::Window::take_access_events
//! [`Window::update_access`]: super::Window::update_access

#[cfg(feature = "egui")]
use egui::accesskit::{ActionRequest, TreeUpdate};

/// What a screen reader asked of a window.
#[cfg(feature = "egui")]
#[derive(Clone, Debug, PartialEq)]
pub enum AccessEvent {
    /// A reader is listening: the next update has to be the whole tree, and
    /// later ones what changed.
    Activated,
    /// A reader asked something of a node.
    Action(ActionRequest),
    /// The last reader left, so updates can stop. Only Linux and iOS tell.
    Deactivated,
}

#[cfg(all(
    feature = "egui",
    not(any(target_arch = "wasm32", target_os = "android"))
))]
pub(crate) use adapter::{window_event, Access, AVAILABLE};
#[cfg(not(any(
    target_arch = "wasm32",
    all(feature = "egui", not(target_os = "android"))
)))]
pub(crate) use no_adapter::window_event;
#[cfg(not(all(
    feature = "egui",
    not(any(target_arch = "wasm32", target_os = "android"))
)))]
pub(crate) use no_adapter::{Access, AVAILABLE};

#[cfg(all(
    feature = "egui",
    not(any(target_arch = "wasm32", target_os = "android"))
))]
mod adapter {
    use super::{AccessEvent, TreeUpdate};
    use crate::window::wgpu_canvas::loop_waker;
    use crate::window::Waker;
    #[cfg(not(target_os = "ios"))]
    use accesskit_winit::Adapter;
    use egui::accesskit::{ActionHandler, ActionRequest, ActivationHandler, DeactivationHandler};
    #[cfg(target_os = "ios")]
    use ios::Adapter;
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex, Weak};
    use winit::event::WindowEvent;
    use winit::window::{Window, WindowId};

    pub(crate) const AVAILABLE: bool = true;

    struct Attached {
        id: WindowId,
        window: Weak<Window>,
        adapter: Adapter,
    }

    thread_local! {
        // By window, like the pending events: the handler that receives a
        // window event has no canvas to ask. An adapter is not `Send`.
        static ATTACHED: RefCell<Vec<Attached>> = const { RefCell::new(Vec::new()) };
    }

    /// The handlers AccessKit calls, each a way into the queue.
    #[derive(Clone)]
    struct Heard {
        queue: Arc<Mutex<Vec<AccessEvent>>>,
        waker: Option<Waker>,
    }

    impl Heard {
        fn push(&self, event: AccessEvent) {
            self.queue
                .lock()
                .expect("the access queue is only pushed to and drained")
                .push(event);
            if let Some(waker) = &self.waker {
                waker.wake();
            }
        }
    }

    impl ActivationHandler for Heard {
        // No tree yet: the app builds one in its next frame, and the platform
        // holds a placeholder until `update_access` hands it over.
        fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
            self.push(AccessEvent::Activated);
            None
        }
    }

    impl ActionHandler for Heard {
        fn do_action(&mut self, request: ActionRequest) {
            self.push(AccessEvent::Action(request));
        }
    }

    impl DeactivationHandler for Heard {
        fn deactivate_accessibility(&mut self) {
            self.push(AccessEvent::Deactivated);
        }
    }

    /// A window's adapter, for as long as its canvas lives.
    pub(crate) struct Access {
        id: WindowId,
        queue: Arc<Mutex<Vec<AccessEvent>>>,
    }

    impl Access {
        /// Makes `window`'s adapter. `window` must not have been shown yet.
        pub(crate) fn attach(window: &Arc<Window>) -> Self {
            let queue = Arc::new(Mutex::new(Vec::new()));
            let heard = Heard {
                queue: Arc::clone(&queue),
                waker: loop_waker(),
            };
            #[cfg(not(target_os = "ios"))]
            let adapter = crate::window::wgpu_canvas::with_active_loop(|event_loop| {
                Adapter::with_direct_handlers(
                    event_loop,
                    window,
                    heard.clone(),
                    heard.clone(),
                    heard,
                )
            });
            #[cfg(target_os = "ios")]
            let adapter = Adapter::new(window, heard);
            let id = window.id();
            ATTACHED.with(|attached| {
                attached.borrow_mut().push(Attached {
                    id,
                    window: Arc::downgrade(window),
                    adapter,
                });
            });
            Self { id, queue }
        }

        /// What a reader asked since the last call.
        pub(crate) fn take_events(&self) -> Vec<AccessEvent> {
            std::mem::take(
                &mut *self
                    .queue
                    .lock()
                    .expect("the access queue is only pushed to and drained"),
            )
        }

        /// Hands `update` to the platform, which drops it while no reader asked.
        pub(crate) fn update(&self, update: TreeUpdate) {
            ATTACHED.with(|attached| {
                let mut attached = attached.borrow_mut();
                if let Some(attached) = attached.iter_mut().find(|one| one.id == self.id) {
                    attached.adapter.update_if_active(|| update);
                }
            });
        }
    }

    impl Drop for Access {
        fn drop(&mut self) {
            ATTACHED.with(|attached| attached.borrow_mut().retain(|one| one.id != self.id));
        }
    }

    /// UIKit's adapter, made as `accesskit_winit` makes it but without asking
    /// the window whether it is visible, which winit answers on iOS with a
    /// warning and nothing else.
    #[cfg(target_os = "ios")]
    mod ios {
        use super::Heard;
        use accesskit_ios::SubclassingAdapter;
        use egui::accesskit::TreeUpdate;
        use winit::event::WindowEvent;
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        use winit::window::Window;

        pub(super) struct Adapter(SubclassingAdapter);

        impl Adapter {
            pub(super) fn new(window: &Window, heard: Heard) -> Self {
                let handle = window
                    .window_handle()
                    .expect("an open window has a handle")
                    .as_raw();
                let RawWindowHandle::UiKit(handle) = handle else {
                    unreachable!("an iOS window is a UIView");
                };
                // The view is the window's own, alive as long as it is.
                let adapter = unsafe {
                    SubclassingAdapter::new(
                        handle.ui_view.as_ptr(),
                        heard.clone(),
                        heard.clone(),
                        heard,
                    )
                };
                Self(adapter)
            }

            pub(super) fn update_if_active(&mut self, update: impl FnOnce() -> TreeUpdate) {
                if let Some(events) = self.0.update_if_active(update) {
                    events.raise();
                }
            }

            pub(super) fn process_event(&mut self, _window: &Window, _event: &WindowEvent) {}
        }
    }

    /// What every window event has to pass through before the app handles it.
    pub(crate) fn window_event(id: WindowId, event: &WindowEvent) {
        ATTACHED.with(|attached| {
            for one in attached.borrow_mut().iter_mut().filter(|one| one.id == id) {
                if let Some(window) = one.window.upgrade() {
                    one.adapter.process_event(&window, event);
                }
            }
        });
    }
}

#[cfg(not(all(
    feature = "egui",
    not(any(target_arch = "wasm32", target_os = "android"))
)))]
mod no_adapter {
    #[cfg(feature = "egui")]
    use super::{AccessEvent, TreeUpdate};
    use std::sync::Arc;
    use winit::window::Window;

    pub(crate) const AVAILABLE: bool = false;

    pub(crate) struct Access;

    impl Access {
        pub(crate) fn attach(_window: &Arc<Window>) -> Self {
            Self
        }

        #[cfg(feature = "egui")]
        pub(crate) fn take_events(&self) -> Vec<AccessEvent> {
            Vec::new()
        }

        #[cfg(feature = "egui")]
        pub(crate) fn update(&self, _update: TreeUpdate) {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn window_event(_id: winit::window::WindowId, _event: &winit::event::WindowEvent) {}
}
