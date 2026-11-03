//! The platform adapter on a real, hidden window, asked what macOS's accessibility
//! API asks: the same calls a screen reader's arrive as, in process, so no reader
//! and no permission are needed. AppKit wants its main thread, so this has no
//! test harness.

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
    println!("access_adapter: ok");
}

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
mod macos {
    use egui::accesskit::{
        Action, ActionRequest, Node, NodeId, Role, TreeId, TreeInfo, TreeUpdate,
    };
    use kiss3d::window::{AccessEvent, CanvasSetup, NumSamples, Window};
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use std::ffi::{c_char, CStr};
    use std::time::Duration;

    const ROOT: NodeId = NodeId(1);
    const BUTTON: NodeId = NodeId(2);
    const ASKING: &str = "access_adapter_asking";
    const SILENT: &str = "access_adapter_silent";

    fn text(string: *mut AnyObject) -> String {
        assert!(!string.is_null(), "no string to read");
        let bytes: *const c_char = unsafe { msg_send![string, UTF8String] };
        unsafe { CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned()
    }

    /// The content view of the window with this title, hidden or not.
    fn view_of(title: &str) -> *mut AnyObject {
        let app: *mut AnyObject = unsafe { msg_send![class!(NSApplication), sharedApplication] };
        let windows: *mut AnyObject = unsafe { msg_send![app, windows] };
        let count: usize = unsafe { msg_send![windows, count] };
        for index in 0..count {
            let window: *mut AnyObject = unsafe { msg_send![windows, objectAtIndex: index] };
            let named: *mut AnyObject = unsafe { msg_send![window, title] };
            if text(named) == title {
                return unsafe { msg_send![window, contentView] };
            }
        }
        panic!("no window titled {}", title);
    }

    fn children(of: *mut AnyObject) -> Vec<*mut AnyObject> {
        let list: *mut AnyObject = unsafe { msg_send![of, accessibilityChildren] };
        if list.is_null() {
            return Vec::new();
        }
        let count: usize = unsafe { msg_send![list, count] };
        (0..count)
            .map(|index| unsafe { msg_send![list, objectAtIndex: index] })
            .collect()
    }

    fn tree() -> TreeUpdate {
        let mut root = Node::new(Role::Window);
        root.set_children(vec![BUTTON]);
        let mut button = Node::new(Role::Button);
        button.set_label("Play");
        button.add_action(Action::Click);
        TreeUpdate {
            nodes: vec![(ROOT, root), (BUTTON, button)],
            tree: Some(TreeInfo::new(ROOT)),
            tree_id: TreeId::ROOT,
            focus: ROOT,
        }
    }

    async fn window(title: &str, accessible: bool) -> Window {
        Window::new_hidden_with_setup(
            title,
            64,
            64,
            CanvasSetup {
                accessible,
                samples: NumSamples::One,
                ..CanvasSetup::default()
            },
        )
        .await
    }

    pub(super) fn run() {
        pollster::block_on(async {
            let mut asking = window(ASKING, true).await;
            let silent = window(SILENT, false).await;
            asking.wait_events(Some(Duration::ZERO));

            assert!(
                asking.take_access_events().is_empty(),
                "no one has asked yet"
            );
            asking.update_access(tree());
            assert!(
                asking.take_access_events().is_empty(),
                "an update before a reader asks goes nowhere and asks nothing"
            );

            let view = view_of(ASKING);
            let placeholder = children(view);
            assert_eq!(
                asking.take_access_events(),
                [AccessEvent::Activated],
                "the platform's first question is a reader arriving"
            );
            assert_eq!(placeholder.len(), 1, "a placeholder until the tree comes");
            assert!(children(placeholder[0]).is_empty());

            asking.update_access(tree());
            let root = children(view);
            assert_eq!(root.len(), 1, "the tree's root");
            let under_root = children(root[0]);
            assert_eq!(under_root.len(), 1, "the root's one child");
            let button = under_root[0];
            let role: *mut AnyObject = unsafe { msg_send![button, accessibilityRole] };
            assert_eq!(text(role), "AXButton");
            let title: *mut AnyObject = unsafe { msg_send![button, accessibilityTitle] };
            assert_eq!(text(title), "Play");

            // A press wakes a loop asleep in `wait_events`: an idle window draws
            // nothing, and a reader's press has to reach it anyway.
            asking.wait_events(Some(Duration::ZERO));
            let pressed: bool = unsafe { msg_send![button, accessibilityPerformPress] };
            assert!(pressed, "a node that says it can be pressed takes a press");
            assert!(
                asking.wait_events(Some(Duration::from_secs(30))),
                "the press wakes the loop"
            );
            assert_eq!(
                asking.take_access_events(),
                [AccessEvent::Action(ActionRequest {
                    action: Action::Click,
                    target_tree: TreeId::ROOT,
                    target_node: BUTTON,
                    data: None,
                })]
            );
            assert!(asking.take_access_events().is_empty(), "read once");

            // Without the opt-in, the same questions meet AppKit's own view.
            let plain = children(view_of(SILENT));
            assert!(plain.is_empty());
            silent.update_access(tree());
            assert!(silent.take_access_events().is_empty());
        });
    }
}
