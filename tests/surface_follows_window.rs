//! The surface is the size of the window AppKit gave, on a real display, when
//! that is not the size asked for: a window bigger than the screen, shrunk when
//! shown, and a window maximized after it opened. A surface that misses either
//! is squeezed into the window, and the pointer misses what it points at.
//! AppKit wants its main thread, so this has no test harness.

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
    println!("surface_follows_window: ok");
}

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
mod macos {
    use kiss3d::window::{CanvasSetup, NumSamples, Window};
    use objc2::encode::{Encode, Encoding};
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use std::ffi::{c_char, CStr};
    use std::time::Duration;

    const OVERSIZED: &str = "surface_follows_window_oversized";
    const MAXIMIZED: &str = "surface_follows_window_maximized";
    // Past any screen, and under wgpu's 16384 texture limit at a scale of 2.
    const TOO_BIG: u32 = 6_000;
    const SMALL: u32 = 320;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Pair(f64, f64);

    unsafe impl Encode for Pair {
        const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Size(f64, f64);

    unsafe impl Encode for Size {
        const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Rect(Pair, Size);

    unsafe impl Encode for Rect {
        const ENCODING: Encoding = Encoding::Struct("CGRect", &[Pair::ENCODING, Size::ENCODING]);
    }

    fn named(window: *mut AnyObject) -> String {
        let title: *mut AnyObject = unsafe { msg_send![window, title] };
        let bytes: *const c_char = unsafe { msg_send![title, UTF8String] };
        unsafe { CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned()
    }

    fn window_titled(title: &str) -> *mut AnyObject {
        let app: *mut AnyObject = unsafe { msg_send![class!(NSApplication), sharedApplication] };
        let windows: *mut AnyObject = unsafe { msg_send![app, windows] };
        let count: usize = unsafe { msg_send![windows, count] };
        (0..count)
            .map(|index| unsafe { msg_send![windows, objectAtIndex: index] })
            .find(|&window: &*mut AnyObject| named(window) == title)
            .expect("the window is open")
    }

    /// The content view's size in device pixels: what the window really got.
    fn content_pixels(title: &str) -> (u32, u32) {
        let window = window_titled(title);
        let view: *mut AnyObject = unsafe { msg_send![window, contentView] };
        let frame: Rect = unsafe { msg_send![view, frame] };
        let scale: f64 = unsafe { msg_send![window, backingScaleFactor] };
        let Size(width, height) = frame.1;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        (
            (width * scale).round() as u32,
            (height * scale).round() as u32,
        )
    }

    async fn open(title: &str, size: u32) -> Window {
        // `accessible` opens the window hidden and pumps once before showing
        // it: what Balaur opens, and when winit reports the size asked for.
        Window::new_with_setup(
            title,
            size,
            size,
            CanvasSetup {
                accessible: true,
                samples: NumSamples::One,
                ..CanvasSetup::default()
            },
        )
        .await
    }

    fn settle(window: &mut Window) {
        for _ in 0..5 {
            window.wait_events(Some(Duration::from_millis(100)));
        }
    }

    pub(super) fn run() {
        pollster::block_on(async {
            let mut oversized = open(OVERSIZED, TOO_BIG).await;
            settle(&mut oversized);
            let got = content_pixels(OVERSIZED);
            assert!(
                got.0 < TOO_BIG,
                "the screen is smaller than the window asked for"
            );
            assert_eq!(
                (oversized.width(), oversized.height()),
                got,
                "the surface is the size of the window the screen allowed"
            );

            let mut maximized = open(MAXIMIZED, SMALL).await;
            settle(&mut maximized);
            maximized.set_maximized(true);
            settle(&mut maximized);
            let got = content_pixels(MAXIMIZED);
            assert!(got.0 > SMALL, "maximizing grew the window");
            assert_eq!(
                (maximized.width(), maximized.height()),
                got,
                "the surface is the size of the maximized window"
            );
        });
    }
}
