//! Composed text input: what an input method editor reports while a word is
//! being assembled, and what it commits.
//!
//! Delivered beside the event stream rather than inside it, as dropped
//! files are: [`WindowEvent`](super::WindowEvent) is `Copy`, and composed
//! text is not. Nothing arrives until
//! [`Window::set_ime_allowed`](crate::window::Window::set_ime_allowed) has
//! been called with `true`.

/// One report from the platform's input method editor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImeEvent {
    /// The editor took the keyboard: composition may follow.
    Enabled,
    /// The text being composed so far, with the caret range in bytes of that
    /// text when the editor places one. Shown in place of nothing; the next
    /// `Preedit` or a `Commit` replaces it.
    Preedit {
        text: String,
        cursor: Option<(usize, usize)>,
    },
    /// Composed text the editor settled on: what a text field appends.
    Commit(String),
    /// The editor let go; any preedit shown is withdrawn.
    Disabled,
}

impl ImeEvent {
    /// From winit's own report.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_winit(ime: winit::event::Ime) -> Self {
        match ime {
            winit::event::Ime::Enabled => Self::Enabled,
            winit::event::Ime::Preedit(text, cursor) => Self::Preedit { text, cursor },
            winit::event::Ime::Commit(text) => Self::Commit(text),
            winit::event::Ime::Disabled => Self::Disabled,
        }
    }
}
