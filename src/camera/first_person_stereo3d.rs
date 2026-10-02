use std::f32;

use glamx::glam::camera::rh::proj::opengl;
use glamx::{Mat4, Pose3, Vec2, Vec3};

use crate::camera::Camera3d;
use crate::event::{Action, Key, MouseButton, WindowEvent};
use crate::window::Canvas;

/// Stereo first-person camera: two eyes `ipd` apart, the left eye drawn into the
/// left half of the frame and the right eye into the right half (pair it with
/// [`OculusStereo`](crate::post_processing::OculusStereo) for a headset).
///
///   * Left button press + drag - look around
///   * Right button press + drag - translates the camera position on the plane orthogonal to the
///     view direction
///   * Scroll in/out - zoom in/out
#[derive(Copy, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FirstPersonCamera3dStereo {
    /// The camera position
    eye: Vec3,
    eye_left: Vec3,
    eye_right: Vec3,

    /// Inter Pupilary Distance
    ipd: f32,
    /// How far ahead the two eyes converge: the distance to the point
    /// `look_at` was given.
    focus: f32,

    /// Yaw of the camera (rotation along the y axis).
    yaw: f32,
    /// Pitch of the camera (rotation along the x axis).
    pitch: f32,

    /// Increment of the yaw per unit mouse movement. The default value is 0.005.
    yaw_step: f32,
    /// Increment of the pitch per unit mouse movement. The default value is 0.005.
    pitch_step: f32,
    /// Increment of the translation per arrow press. The default value is 0.5.
    move_step: f32,
    #[cfg_attr(feature = "serde", serde(default = "super::all_render_layers"))]
    render_layers: u32,

    /// Low level data
    fov: f32,
    znear: f32,
    zfar: f32,
    view_left: Mat4,
    view_right: Mat4,
    proj: Mat4,
    proj_view: Mat4,
    inverse_proj_view: Mat4,
    last_cursor_pos: Vec2,
    last_framebuffer_size: Vec2,
}

impl FirstPersonCamera3dStereo {
    /// Creates a first person camera with default sensitivity values.
    pub fn new(eye: Vec3, at: Vec3, ipd: f32) -> FirstPersonCamera3dStereo {
        FirstPersonCamera3dStereo::new_with_frustum(
            f32::consts::PI / 4.0,
            0.1,
            1024.0,
            eye,
            at,
            ipd,
        )
    }

    /// Creates a new first person camera with default sensitivity values.
    pub fn new_with_frustum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Vec3,
        at: Vec3,
        ipd: f32,
    ) -> FirstPersonCamera3dStereo {
        let mut res = FirstPersonCamera3dStereo {
            eye: Vec3::ZERO,
            // left & right are initially wrong, don't take ipd into account
            eye_left: Vec3::ZERO,
            eye_right: Vec3::ZERO,
            ipd,
            focus: 1.0,
            yaw: 0.0,
            pitch: 0.0,
            yaw_step: 0.005,
            pitch_step: 0.005,
            move_step: 0.5,
            render_layers: u32::MAX,
            fov,
            znear,
            zfar,
            proj_view: Mat4::IDENTITY,
            inverse_proj_view: Mat4::IDENTITY,
            last_cursor_pos: Vec2::ZERO,
            last_framebuffer_size: Vec2::new(800.0, 600.0),
            proj: Mat4::IDENTITY,
            view_left: Mat4::IDENTITY,
            view_right: Mat4::IDENTITY,
        };

        res.look_at(eye, at);

        res
    }

    /// Changes the orientation and position of the camera to look at the specified point.
    pub fn look_at(&mut self, eye: Vec3, at: Vec3) {
        let dist = (eye - at).length();

        let pitch = ((at.y - eye.y) / dist).acos();
        let yaw = (at.z - eye.z).atan2(at.x - eye.x);

        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        if dist > 0.0 {
            self.focus = dist;
        }
        self.update_eyes_location();
        self.update_projviews();
    }

    /// The point the camera is looking at.
    pub fn at(&self) -> Vec3 {
        let ax = self.eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = self.eye.y + self.pitch.cos();
        let az = self.eye.z + self.yaw.sin() * self.pitch.sin();

        Vec3::new(ax, ay, az)
    }

    fn update_restrictions(&mut self) {
        if self.pitch <= 0.0001 {
            self.pitch = 0.0001
        }

        let _pi: f32 = f32::consts::PI;
        if self.pitch > _pi - 0.0001 {
            self.pitch = _pi - 0.0001
        }
    }

    #[doc(hidden)]
    pub fn handle_left_button_displacement(&mut self, dpos: Vec2) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch += dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_eyes_location(&mut self) {
        // left and right are on a line perpendicular to both up and the target
        // up is always y
        let dir = (self.at() - self.eye).normalize();
        let tangent = Vec3::Y.cross(dir).normalize();
        self.eye_left = self.eye - tangent * (self.ipd / 2.0);
        self.eye_right = self.eye + tangent * (self.ipd / 2.0);
        //println(fmt!("eye_left = %f,%f,%f", self.eye_left.x as float, self.eye_left.y as float, self.eye_left.z as float));
        //println(fmt!("eye_right = %f,%f,%f", self.eye_right.x as float, self.eye_right.y as float, self.eye_right.z as float));
        // TODO: verify with an assert or something that the distance between the eyes is ipd, just to make me feel good.
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: Vec2) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = Vec3::Y.cross(dir).normalize();
        let bitangent = dir.cross(tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        // TODO: ugly - should move eye update to where eye_left & eye_right are updated
        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front = self.view_transform().rotation * Vec3::Z;

        self.eye += front * (self.move_step * yoff);

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        // Each eye draws into half the frame's width.
        let aspect = self.last_framebuffer_size.x * 0.5 / self.last_framebuffer_size.y;
        self.proj = opengl::perspective(self.fov, aspect, self.znear, self.zfar);
        self.proj_view = self.proj * self.view_transform().to_mat4();
        self.inverse_proj_view = self.proj_view.inverse();
        self.view_left = self.view_transform_left().to_mat4();
        self.view_right = self.view_transform_right().to_mat4();
    }

    #[allow(dead_code)]
    fn view_eye(&self, eye: usize) -> Mat4 {
        match eye {
            0usize => self.view_left,
            1usize => self.view_right,
            _ => panic!("bad eye index"),
        }
    }

    /// The left eye camera view transformation
    fn view_transform_left(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye_left, self.focus_point(), Vec3::Y)
    }

    /// Where the two eyes converge.
    fn focus_point(&self) -> Vec3 {
        self.eye + (self.at() - self.eye) * self.focus
    }

    /// The right eye camera view transformation
    fn view_transform_right(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye_right, self.focus_point(), Vec3::Y)
    }

    /// return Inter Pupilary Distance
    pub fn ipd(&self) -> f32 {
        self.ipd
    }

    /// change Inter Pupilary Distance
    pub fn set_ipd(&mut self, ipd: f32) {
        self.ipd = ipd;

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    /// Sets the render-layer bitmask this camera draws (see
    /// [`Camera3d::render_layers`]). The default, `u32::MAX`, draws every layer.
    pub fn set_render_layers(&mut self, layers: u32) {
        self.render_layers = layers;
    }
}

impl Camera3d for FirstPersonCamera3dStereo {
    fn clip_planes(&self) -> (f32, f32) {
        (self.znear, self.zfar)
    }

    /// The imaginary middle eye camera view transformation (i-e transformation without projection).
    fn view_transform(&self) -> Pose3 {
        Pose3::look_at_rh(self.eye, self.at(), Vec3::Y)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vec2::new(x as f32, y as f32);

                if canvas.get_mouse_button(MouseButton::Button1) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_left_button_displacement(dpos)
                }

                if canvas.get_mouse_button(MouseButton::Button2) == Action::Press {
                    let dpos = curr_pos - self.last_cursor_pos;
                    self.handle_right_button_displacement(dpos)
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.last_framebuffer_size = Vec2::new(w as f32, h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    fn eye(&self) -> Vec3 {
        self.eye
    }

    fn transformation(&self) -> Mat4 {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.inverse_proj_view
    }

    fn update(&mut self, canvas: &Canvas) {
        let t = self.view_transform();
        let front = t.rotation * Vec3::Z;
        let right = t.rotation * Vec3::X;

        if canvas.get_key(Key::Up) == Action::Press {
            self.eye += front * self.move_step
        }

        if canvas.get_key(Key::Down) == Action::Press {
            self.eye += front * (-self.move_step)
        }

        if canvas.get_key(Key::Right) == Action::Press {
            self.eye += right * (-self.move_step)
        }

        if canvas.get_key(Key::Left) == Action::Press {
            self.eye += right * self.move_step
        }

        self.update_eyes_location();
        self.update_restrictions();
        self.update_projviews();
    }

    fn view_transform_pair(&self, pass: usize) -> (Pose3, Mat4) {
        let view = match pass {
            0 => self.view_transform_left(),
            1 => self.view_transform_right(),
            _ => self.view_transform(),
        };
        (view, self.proj)
    }

    fn num_passes(&self) -> usize {
        2usize
    }

    fn render_layers(&self) -> u32 {
        self.render_layers
    }

    fn pass_viewport(&self, pass: usize, width: u32, height: u32) -> [f32; 4] {
        let left = width / 2;
        match pass {
            0 => [0.0, 0.0, left as f32, height as f32],
            _ => [left as f32, 0.0, (width - left) as f32, height as f32],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FirstPersonCamera3dStereo;
    use crate::camera::{Camera3d, OrbitCamera3d};
    use crate::color::Color;
    use crate::scene::{AlphaMode, SceneNode3d};
    use crate::test_gpu::{luma_at, on_gpu};
    use glamx::Vec3;

    fn glowing_ball(alpha: f32) -> SceneNode3d {
        let mut scene = SceneNode3d::empty();
        let mut ball = scene.add_sphere(0.6);
        ball.set_color(Color::new(1.0, 1.0, 1.0, alpha));
        ball.set_emissive(Color::new(4.0, 4.0, 4.0, 1.0));
        if alpha < 1.0 {
            ball.set_alpha_mode(AlphaMode::Blend);
        }
        scene
    }

    #[test]
    fn each_eye_draws_into_its_own_half() {
        on_gpu(128, 64, async |surface| {
            for alpha in [1.0, 0.6] {
                let mut scene = glowing_ball(alpha);
                let mut camera =
                    FirstPersonCamera3dStereo::new(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, 0.2);
                surface.render_3d(&mut scene, &mut camera).await;
                let (left, middle, right) = (
                    luma_at(surface, 32, 32),
                    luma_at(surface, 64, 32),
                    luma_at(surface, 96, 32),
                );
                assert!(left > 0.3 && right > 0.3, "{} {} {}", left, middle, right);
                assert!(middle < 0.1, "{} {} {}", left, middle, right);
            }
        });
    }

    #[test]
    fn both_eyes_converge_on_the_point_looked_at() {
        on_gpu(128, 64, async |surface| {
            let mut scene = glowing_ball(1.0);
            let mut camera =
                FirstPersonCamera3dStereo::new(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, 2.0);
            surface.render_3d(&mut scene, &mut camera).await;
            assert!(
                luma_at(surface, 32, 32) > 0.3,
                "{}",
                luma_at(surface, 32, 32)
            );
            assert!(
                luma_at(surface, 96, 32) > 0.3,
                "{}",
                luma_at(surface, 96, 32)
            );
        });
    }

    #[test]
    fn a_mono_camera_still_draws_across_the_whole_frame() {
        on_gpu(128, 64, async |surface| {
            let mut scene = glowing_ball(1.0);
            let mut camera = OrbitCamera3d::new(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
            surface.render_3d(&mut scene, &mut camera).await;
            assert!(luma_at(surface, 64, 32) > 0.3);
            assert!(luma_at(surface, 32, 32) < 0.1);
        });
    }

    #[test]
    fn the_stereo_layers_are_settable() {
        let mut camera = FirstPersonCamera3dStereo::new(Vec3::Z, Vec3::ZERO, 0.1);
        assert_eq!(camera.render_layers(), u32::MAX);
        camera.set_render_layers(4);
        assert_eq!(camera.render_layers(), 4);
    }
}
