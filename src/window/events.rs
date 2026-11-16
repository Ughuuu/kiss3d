//! Event handling functionality.

use crate::camera::Camera2d;
use crate::camera::Camera3d;
use crate::event::{Action, EventManager, Key, MouseButton, WindowEvent};

use super::Window;

#[cfg(feature = "rt_switcher")]
use crate::prelude::{RayTracer, RayTracerPreset};

impl Window {
    /// Returns an event manager for accessing window events.
    ///
    /// The event manager provides an iterator over events that occurred since the last frame,
    /// such as keyboard input, mouse movement, and window resizing.
    ///
    /// # Returns
    /// An `EventManager` that can be iterated to process events
    ///
    /// # Example
    /// ```no_run
    /// # use kiss3d::prelude::*;
    /// # #[kiss3d::main]
    /// # async fn main() {
    /// # let mut window = Window::new("Example").await;
    /// # let mut camera = OrbitCamera3d::default();
    /// # let mut scene = SceneNode3d::empty();
    /// # while window.render_3d(&mut scene, &mut camera).await {
    /// for event in window.events().iter() {
    ///     match event.value {
    ///         WindowEvent::Key(Key::Escape, Action::Release, _) => {
    ///             println!("Escape pressed!");
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// # }
    /// # }
    /// ```
    pub fn events(&self) -> EventManager {
        EventManager::new(self.events.clone(), self.unhandled_events.clone())
    }

    /// Sleeps until the window system has an event, a [`Waker`] from
    /// [`Self::waker`] fires, or `timeout` passes, and queues what arrived for
    /// [`Self::events`] without drawing. `None` waits for as long as it takes.
    /// Answers whether anything arrived. The web and iOS cannot block here.
    ///
    /// [`Waker`]: crate::window::Waker
    pub fn wait_events(&mut self, timeout: Option<std::time::Duration>) -> bool {
        // Events the last frame's poll queued are already here to read; a
        // wait would sleep until something newer, leaving them unread.
        if std::mem::take(&mut self.events_queued) {
            self.canvas.poll_events();
            return true;
        }
        self.canvas.wait_events(timeout)
    }

    /// What a screen reader asked since the last call: a reader arriving or
    /// leaving, or asking something of a node. Read between frames; an empty answer
    /// is all a window pays until one asks, and only one opened with
    /// [`CanvasSetup::accessible`](crate::window::CanvasSetup::accessible) hears
    /// anything. A request wakes [`Self::wait_events`].
    #[cfg(feature = "egui")]
    pub fn take_access_events(&self) -> Vec<crate::window::AccessEvent> {
        self.canvas.take_access_events()
    }

    /// Hand a screen reader what changed: the whole tree first, after
    /// [`AccessEvent::Activated`](crate::window::AccessEvent::Activated), then
    /// only what moved. Goes nowhere while no reader has asked.
    #[cfg(feature = "egui")]
    pub fn update_access(&self, update: egui::accesskit::TreeUpdate) {
        self.canvas.update_access(update);
    }

    /// What ends a [`Self::wait_events`] from another thread; `None` where
    /// the platform owns the loop.
    pub fn waker(&self) -> Option<crate::window::Waker> {
        self.canvas.waker()
    }

    /// Gets the current state of a keyboard key.
    ///
    /// # Arguments
    /// * `key` - The key to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    /// Gets the current state of a mouse button.
    ///
    /// # Arguments
    /// * `button` - The mouse button to check
    ///
    /// # Returns
    /// The current `Action` state (e.g., `Action::Press`, `Action::Release`)
    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }

    /// Gets the last known position of the mouse cursor.
    ///
    /// The position is automatically updated when the mouse moves over the window.
    /// Coordinates are in pixels, with (0, 0) at the top-left corner.
    ///
    /// # Returns
    /// `Some((x, y))` with the cursor position, or `None` if the cursor position is unknown
    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    #[inline]
    pub(crate) fn handle_events(
        &mut self,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
    ) {
        let unhandled_events = self.unhandled_events.clone(); // TODO: could we avoid the clone?
        let events = self.events.clone(); // TODO: could we avoid the clone?

        // The first `fed` were handed to the pass `draw_ui` opened this frame.
        #[cfg(feature = "egui")]
        let fed = std::mem::take(&mut self.egui_context.fed);
        #[cfg(not(feature = "egui"))]
        let fed = 0;
        for (n, event) in unhandled_events.borrow().iter().enumerate() {
            self.handle_event(camera, camera_2d, event, n >= fed)
        }

        for event in events.try_iter() {
            self.handle_event(camera, camera_2d, &event, true)
        }

        unhandled_events.borrow_mut().clear();
        self.events_queued = self.canvas.poll_events();
        let ime = self.canvas.take_ime_events();
        #[cfg(feature = "egui")]
        for event in &ime {
            use crate::event::ImeEvent;
            // egui counts the caret in characters where winit counts bytes.
            let event = match event {
                ImeEvent::Preedit { text, cursor } => egui::ImeEvent::Preedit {
                    text: text.clone(),
                    active_range_chars: cursor.map(|(start, end)| {
                        // Counting the chars that start before the byte index
                        // rather than slicing at it: an index that is not a
                        // char boundary rounds down instead of panicking.
                        let chars = |byte: usize| {
                            text.char_indices().take_while(|(at, _)| *at < byte).count()
                        };
                        chars(start)..chars(end)
                    }),
                },
                ImeEvent::Commit(text) => egui::ImeEvent::Commit(text.clone()),
                ImeEvent::Enabled | ImeEvent::Disabled => continue,
            };
            self.egui_context
                .raw_input
                .events
                .push(egui::Event::Ime(event));
        }
        *self.ime_events.borrow_mut() = ime;
    }

    pub(crate) fn handle_event(
        &mut self,
        camera: &mut dyn Camera3d,
        camera_2d: &mut dyn Camera2d,
        event: &WindowEvent,
        feed_ui: bool,
    ) {
        if let Some(binding_key) = self.close_key {
            if let WindowEvent::Key(key, Action::Release, modifiers) = event {
                if binding_key == *key
                    && (Some(*modifiers) == self.close_modifiers || self.close_modifiers.is_none())
                {
                    self.close()
                }
            }
        }

        #[cfg(feature = "rt_switcher")]
        match event {
            WindowEvent::Key(Key::F4, Action::Release, _) => {
                self.raytracer.1 = false;
            }
            WindowEvent::Key(Key::F5, Action::Release, _) => {
                self.raytracer.0 = Some(RayTracer::preset(RayTracerPreset::Low));
            }
            WindowEvent::Key(Key::F6, Action::Release, _) => {
                self.raytracer.0 = Some(RayTracer::preset(RayTracerPreset::Medium));
            }
            WindowEvent::Key(Key::F7, Action::Release, _) => {
                self.raytracer.0 = Some(RayTracer::preset(RayTracerPreset::High));
            }
            WindowEvent::Key(Key::F8, Action::Release, _) => {
                self.raytracer.0 = Some(RayTracer::preset(RayTracerPreset::Ultra));
            }
            _ => {}
        }

        if *event == WindowEvent::Close {
            self.close();
        }

        // Feed events to egui and check if it wants to capture input
        #[cfg(not(feature = "egui"))]
        let _ = feed_ui;
        #[cfg(feature = "egui")]
        {
            if feed_ui {
                self.feed_egui_event(event);
            }

            if event.is_keyboard_event() && self.is_egui_capturing_keyboard() {
                return;
            }

            if event.is_mouse_event() && self.is_egui_capturing_mouse() {
                return;
            }
        }

        camera.handle_event(&self.canvas, event);
        camera_2d.handle_event(&self.canvas, event);
    }
}
