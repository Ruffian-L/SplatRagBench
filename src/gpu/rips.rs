use anyhow::Result;

#[cfg(feature = "gpu-acceleration")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "gpu-acceleration")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "gpu-acceleration")]
use std::sync::Arc;

// Helper for Rips Complex structure
pub struct RipsComplex {
    pub distances: Vec<f32>, // CHANGED: Now f32, not u8
    pub num_points: usize,
}

#[cfg(feature = "gpu-acceleration")]
pub fn compute_distances_gpu(
    device: &Arc<CudaDevice>, 
    points: &[[f32; 3]], 
    threshold: f32
) -> Result<cudarc::driver::CudaSlice<f32>> { // CHANGED: Return CudaSlice<f32>
    let n = points.len();
    if n == 0 {
         return device.alloc_zeros::<f32>(0).map_err(Into::into);
    }
    
    // 1. Upload points
    let points_flat: Vec<f32> = points.iter().flat_map(|p| p.as_slice()).cloned().collect();
    let d_points = device.htod_copy(points_flat)?;
    
    // 2. Allocate Distance Matrix on GPU (float)
    let mut d_dists = device.alloc_zeros::<f32>(n * n)?;

    // 3. Launch Kernel
    let ptx = compile_ptx(include_str!("kernels/distance_matrix.cu"))?;
    
    device.load_ptx(ptx, "distance_module", &["compute_distances"])?;
    let f = device.get_func("distance_module", "compute_distances").unwrap();

    let cfg = LaunchConfig::for_num_elems((n * n) as u32);
    unsafe { f.launch(cfg, (&d_points, &mut d_dists, n as i32, threshold)) }?;

    Ok(d_dists)
}

#[cfg(feature = "gpu-acceleration")]
pub fn build_rips_complex_gpu(
    device: &Arc<CudaDevice>,
    points: &[[f32; 3]],
    threshold: f32,
) -> Result<RipsComplex> {
    let n = points.len();
    let d_dists = compute_distances_gpu(device, points, threshold)?;

    // 4. Download Distances
    let dists_host = device.dtoh_sync_copy(&d_dists)?;

    Ok(RipsComplex {
        distances: dists_host,
        num_points: n,
    })
}

#[cfg(not(feature = "gpu-acceleration"))]
pub fn build_rips_complex_gpu(
    _device: &(), // dummy
    _points: &[[f32; 3]],
    _threshold: f32,
) -> Result<RipsComplex> {
    anyhow::bail!("GPU acceleration not enabled. Compile with --features gpu-acceleration")
}
