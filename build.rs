//! Links every built-in shader with `wesl` when kiss3d compiles, so a program
//! carries finished WGSL and no linker. `builtin::linked` looks them up.

use std::fmt::Write as _;
use std::path::PathBuf;

const COMMON: (&str, &str) = ("package::common", "builtin/common.wgsl");
const PBR_ENV: (&str, &str) = ("package::pbr_env", "builtin/pbr_env.wgsl");
const TONEMAP_OPS: (&str, &str) = ("package::tonemap_ops", "builtin/tonemap_ops.wgsl");
const ENV_DOWNSAMPLE: (&str, &str) = ("package::env_downsample", "builtin/env_downsample.wgsl");

/// One shader: the module it starts from, every module it may import, and
/// each set of `@if` flags a caller links it with.
struct Shader {
    root: &'static str,
    modules: Vec<(&'static str, &'static str)>,
    variants: Vec<Vec<(&'static str, bool)>>,
}

/// A shader importing nothing but `package::common`, linked once.
fn with_common(root: &'static str, file: &'static str) -> Shader {
    Shader {
        root,
        modules: vec![(root, file), COMMON],
        variants: vec![vec![]],
    }
}

/// Every combination of `flags`.
fn every(flags: &[&'static str]) -> Vec<Vec<(&'static str, bool)>> {
    (0..1u32 << flags.len())
        .map(|bits| {
            flags
                .iter()
                .enumerate()
                .map(|(i, flag)| (*flag, bits & (1 << i) != 0))
                .collect()
        })
        .collect()
}

fn shaders() -> Vec<Shader> {
    let mut all = vec![
        // The object shader's other flags are `override` constants a pipeline
        // sets; these three change what it declares.
        Shader {
            root: "package::default",
            modules: vec![("package::default", "builtin/default.wgsl"), PBR_ENV],
            variants: every(&["clustered", "deform", "vertex_colors"]),
        },
        Shader {
            root: "package::object2d",
            modules: vec![("package::object2d", "builtin/object2d.wgsl"), COMMON],
            variants: every(&["textured"]),
        },
        Shader {
            root: "package::shadow_depth",
            modules: vec![("package::shadow_depth", "builtin/shadow_depth.wgsl")],
            variants: every(&["skinned"]),
        },
        Shader {
            root: "package::shadow_transmittance",
            modules: vec![(
                "package::shadow_transmittance",
                "builtin/shadow_transmittance.wgsl",
            )],
            variants: every(&["skinned"]),
        },
        Shader {
            root: "package::hdr_tonemap",
            modules: vec![
                TONEMAP_OPS,
                ("package::hdr_tonemap", "builtin/hdr_tonemap.wgsl"),
                COMMON,
            ],
            variants: vec![vec![]],
        },
        Shader {
            root: "package::rt_tonemap",
            modules: vec![
                TONEMAP_OPS,
                ("package::rt_tonemap", "builtin/raytrace/tonemap.wgsl"),
            ],
            variants: vec![vec![]],
        },
        Shader {
            root: "package::skybox",
            modules: vec![("package::skybox", "builtin/skybox.wgsl"), PBR_ENV, COMMON],
            variants: vec![vec![]],
        },
        Shader {
            root: "package::ssr",
            modules: vec![("package::ssr", "builtin/ssr.wgsl"), PBR_ENV, COMMON],
            variants: vec![vec![]],
        },
        Shader {
            root: "package::env_downsample",
            modules: vec![ENV_DOWNSAMPLE, COMMON],
            variants: vec![vec![]],
        },
    ];
    // The path tracer mounts a different intersection module per backend.
    for (hardware, intersect) in [
        (false, "builtin/raytrace/rt_intersect_bvh.wgsl"),
        (true, "builtin/raytrace/rt_intersect_rayquery.wgsl"),
    ] {
        all.push(Shader {
            root: "package::rt_kernel",
            modules: vec![
                ("package::rt_preamble", "builtin/raytrace/rt_preamble.wgsl"),
                ("package::rt_intersect", intersect),
                ("package::rt_kernel", "builtin/raytrace/rt_kernel.wgsl"),
                PBR_ENV,
                COMMON,
            ],
            variants: vec![vec![("hardware", hardware)]],
        });
    }
    for (root, file) in [
        ("package::lit2d", "builtin/lit2d.wgsl"),
        (
            "package::wireframe_polyline2d",
            "builtin/wireframe_polyline2d.wgsl",
        ),
        (
            "package::wireframe_points2d",
            "builtin/wireframe_points2d.wgsl",
        ),
        ("package::skinned2d", "builtin/skinned2d.wgsl"),
        ("package::cas", "builtin/cas.wgsl"),
        ("package::crt", "builtin/crt.wgsl"),
        ("package::fxaa", "builtin/fxaa.wgsl"),
        ("package::gi2d_field", "builtin/gi2d_field.wgsl"),
        ("package::gi2d_composite", "builtin/gi2d_composite.wgsl"),
        ("package::gi2d_jfa_seed", "builtin/gi2d_jfa_seed.wgsl"),
        ("package::gi2d_jfa", "builtin/gi2d_jfa.wgsl"),
        ("package::gi2d_cascade", "builtin/gi2d_cascade.wgsl"),
        (
            "package::gi2d_cascade_composite",
            "builtin/gi2d_cascade_composite.wgsl",
        ),
        ("package::grayscales", "builtin/grayscales.wgsl"),
        ("package::hdr_bloom", "builtin/hdr_bloom.wgsl"),
        (
            "package::auto_exposure_meter",
            "builtin/auto_exposure_meter.wgsl",
        ),
        (
            "package::auto_exposure_adapt",
            "builtin/auto_exposure_adapt.wgsl",
        ),
        ("package::loupe", "builtin/loupe.wgsl"),
        ("package::oculus", "builtin/oculus.wgsl"),
        ("package::depth_resolve", "builtin/depth_resolve.wgsl"),
        ("package::sobel", "builtin/sobel.wgsl"),
        ("package::waves", "builtin/waves.wgsl"),
        ("package::dof", "builtin/dof.wgsl"),
        ("package::points2d", "builtin/points2d.wgsl"),
        ("package::polyline2d", "builtin/polyline2d.wgsl"),
        ("package::denoise", "builtin/raytrace/denoise.wgsl"),
        ("package::cube_to_equirect", "builtin/cube_to_equirect.wgsl"),
        ("package::ssao", "builtin/ssao.wgsl"),
        (
            "package::transmission_downsample",
            "builtin/transmission_downsample.wgsl",
        ),
    ] {
        all.push(with_common(root, file));
    }
    all
}

fn main() {
    let src = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/builtin");
    let mut table = String::from("&[\n");
    for shader in shaders() {
        for features in &shader.variants {
            let mut resolver = wesl::resolver::VirtualResolver::new();
            for (path, file) in &shader.modules {
                let text = std::fs::read_to_string(src.join(file))
                    .unwrap_or_else(|e| panic!("reading {}: {}", file, e));
                resolver.add_module(path.parse().expect("a module path"), text.into());
            }
            // naga validates the result when kiss3d's own tests build it.
            let mut options = wesl::CompileOptions {
                validate: false,
                ..Default::default()
            };
            for (name, on) in features {
                options.features.set(*name, *on);
            }
            let wgsl = wesl::Compiler::new_with_resolver(options, resolver)
                .compile_module(&shader.root.parse().expect("a module path"))
                .unwrap_or_else(|e| panic!("linking {} {:?}: {}", shader.root, features, e))
                .to_string();
            let mut name = shader.root.replace("::", "_");
            for (flag, on) in features {
                let _ = write!(name, "__{flag}_{}", u8::from(*on));
            }
            let file = out.join(format!("{name}.wgsl"));
            std::fs::write(&file, wgsl).expect("writing a linked shader");
            let flags: Vec<String> = features
                .iter()
                .map(|(flag, on)| format!("({flag:?}, {on})"))
                .collect();
            let _ = writeln!(
                table,
                "    ({:?}, &[{}], include_str!({:?})),",
                shader.root,
                flags.join(", "),
                file.display().to_string()
            );
        }
    }
    table.push(']');
    std::fs::write(out.join("linked.rs"), table).expect("writing the shader table");
}
