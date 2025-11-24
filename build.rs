use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Local definition of SplatGeometry to avoid dependency cycle
#[repr(C)]
#[derive(Copy, Clone)]
struct SplatGeometry {
    pub position: [f32; 3],     // 12 bytes
    pub scale: [f32; 3],        // 12 bytes
    pub rotation: [f32; 4],     // 16 bytes
    pub color_rgba: [u8; 4],    // 4 bytes
    pub physics_props: [u8; 4], // 4 bytes
}

// Local definition of SplatSemantics for size estimation
// Note: This is only accurate for in-memory layout, not bincode on-disk.
struct SplatSemantics {
    pub payload_id: u64,
    pub birth_time: f64,
    pub confidence: f32,
    pub embedding: [f32; 384],
    pub emotional_state: Option<()>,
    pub fitness_metadata: Option<()>,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/structs.rs");

    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=kernels/reduce.cu");

        std::fs::create_dir_all("target/nvptx").unwrap();

        if std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .is_ok()
        {
            let status = std::process::Command::new("nvcc")
                .args(&[
                    "--ptx",
                    // Try targeting native architecture of the machine
                    // or fallback to sm_86 (RTX 30 series) which runs on everything modern
                    "-arch=sm_86", 
                    "kernels/distance_matrix.cu", // Make sure to include the new kernel
                    "-o",
                    "target/nvptx/distance_matrix.ptx",
                ])
                .status()
                .expect("Failed to execute nvcc. Is CUDA Toolkit installed?");

            // Compile reduce.cu
            let status_reduce = std::process::Command::new("nvcc")
                .args(&[
                    "--ptx",
                    "-arch=sm_86",
                    "kernels/reduce.cu",
                    "-o",
                    "target/nvptx/reduce.ptx",
                ])
                .status()
                .expect("Failed to execute nvcc for reduce.cu");

            if !status.success() || !status_reduce.success() {
                let compute_cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "86".to_string());
                println!("cargo:rustc-env=CUDA_COMPUTE_CAP={}", compute_cap);
                println!("cargo:warning=CUDA Kernel compilation failed. Check nvcc installation.");
            }
        } else {
            println!("cargo:warning=nvcc not found. GPU features will be runtime disabled.");
        }
    }

    let geom_size = std::mem::size_of::<SplatGeometry>();
    // Semantics size is tricky due to Bincode variable length.
    // We provide a rough estimate or the in-memory size, but python script should be careful.
    // We'll use a placeholder or calculated size.
    // SplatSemantics (Rust) is roughly 8+8+4+1536 + options.
    let sem_size = 8 + 8 + 4 + (384 * 4) + 64; // Approx

    // For Rust
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_sizes.rs");
    let mut f = File::create(&dest_path).unwrap();
    writeln!(f, "pub const GEOMETRY_STRIDE: usize = {};", geom_size).unwrap();
    writeln!(f, "pub const SEM_STRIDE: usize = {};", sem_size).unwrap();

    // For Python
    let json_path = Path::new("splat_sizes.json");
    let json = format!(
        r#"{{"geometry_stride": {}, "semantics_stride": {}}}"#,
        geom_size, sem_size
    );
    std::fs::write(json_path, json).unwrap();

    println!("cargo:warning=Splat sizes written: geom={geom_size}, sem={sem_size}");
}
