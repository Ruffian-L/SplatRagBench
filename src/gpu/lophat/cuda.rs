use super::MatrixDecomposer;
use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// We use a flattened Compressed Sparse Row (CSR) format for the GPU
// It's much faster than pointer chasing on a 5080.
pub struct CudaDecomposer {
    // Use generic Arc or fully qualified if importing fails
    device: Arc<cudarc::driver::CudaDevice>,
    // We keep these CPU-side for quick lookups if the GPU is busy
    cpu_fallback_cache: Option<Vec<Vec<usize>>>,
    num_cols: usize,
    num_rows: usize,
    pivots: Vec<Option<usize>>,
}

impl CudaDecomposer {
    pub fn new(boundary_matrix: Vec<Vec<usize>>) -> Result<Self> {
        let dev = cudarc::driver::CudaDevice::new(0)?;

        // Load the PTX (compiled CUDA code)
        // We assume build.rs compiles 'kernels/reduce.cu' to 'reduce.ptx'
        dev.load_ptx(
            Ptx::from_file("./target/nvptx/reduce.ptx"),
            "persistence",
            &["lock_free_reduction"],
        )?;

        let rows = boundary_matrix.len(); // logic approximation
        let cols = boundary_matrix.len();

        Ok(Self {
            device: dev,
            cpu_fallback_cache: Some(boundary_matrix), // Keep copy for now
            num_cols: cols,
            num_rows: rows,
            pivots: vec![None; cols],
        })
    }

    /// Flattens the matrix and sends it to the GPU
    fn upload_matrix(
        &self,
    ) -> Result<(
        cudarc::driver::CudaSlice<usize>,
        cudarc::driver::CudaSlice<usize>,
    )> {
        let matrix = self.cpu_fallback_cache.as_ref().unwrap();

        let mut col_ptr = Vec::with_capacity(self.num_cols + 1);
        let mut row_indices = Vec::new();

        let mut current_ptr = 0;
        col_ptr.push(current_ptr);

        for col in matrix {
            for &row_idx in col {
                row_indices.push(row_idx);
                current_ptr += 1;
            }
            col_ptr.push(current_ptr);
        }

        let dev_col_ptr = self.device.htod_copy(col_ptr)?;
        let dev_row_idx = self.device.htod_copy(row_indices)?;

        Ok((dev_col_ptr, dev_row_idx))
    }
}

impl MatrixDecomposer for CudaDecomposer {
    fn add_entries(&mut self, _target: usize, _source: usize) {
        // On GPU, we don't do single adds. We batch reduce.
    }

    fn get_pivot(&self, col_idx: usize) -> Option<usize> {
        // Return the cached pivot from GPU reduction
        if col_idx < self.pivots.len() {
            self.pivots[col_idx]
        } else {
            None
        }
    }

    fn get_r_col(&self, col_idx: usize) -> Vec<usize> {
        // In production: Copy back specific slice from GPU
        // For now, fallback to cache (Note: This is unreduced! But get_pivot is correct)
        self.cpu_fallback_cache.as_ref().unwrap()[col_idx].clone()
    }

    fn reduce(&mut self) {
        println!("⚡ 5080-Q: Dispatching Reduction Kernel...");

        // 1. Upload Data
        let (mut d_col_ptr, mut d_row_idx) = self.upload_matrix().unwrap();

        // 2. Allocate Output Buffer (Pivots)
        // Kernel expects i32 (standard int in CUDA)
        // Using i32 to allow -1 for "None"
        let mut d_pivots = self.device.alloc_zeros::<i32>(self.num_cols).unwrap();

        // Initialize with -1
        let init_pivots = vec![-1i32; self.num_cols];
        self.device
            .htod_sync_copy_into(&init_pivots, &mut d_pivots)
            .unwrap();

        // Allocate auxiliary buffers
        let mut d_is_cleared = self.device.alloc_zeros::<bool>(self.num_cols).unwrap();

        let heap_capacity = self.num_cols * 500; // Heuristic size (increased for dense reductions)
        let mut d_heap = self.device.alloc_zeros::<i32>(heap_capacity).unwrap();
        let mut d_heap_ptr = self.device.alloc_zeros::<i32>(1).unwrap();

        // 3. Launch Config
        let cfg = LaunchConfig::for_num_elems(self.num_cols as u32);
        let func = self
            .device
            .get_func("persistence", "lock_free_reduction")
            .unwrap();

        // 4. FIRE
        // Params: (pivots, col_ptr, row_idx, is_cleared, heap, heap_ptr, num_cols, heap_capacity)
        unsafe {
            func.launch(
                cfg,
                (
                    &mut d_pivots,
                    &d_col_ptr,
                    &d_row_idx,
                    &d_is_cleared,
                    &mut d_heap,
                    &mut d_heap_ptr,
                    self.num_cols as i32,
                    heap_capacity as i32,
                ),
            )
        }
        .unwrap();

        // 5. Sync (Wait for the 5080 to chew through the topology)
        self.device.synchronize().unwrap();

        println!("⚡ 5080-Q: Reduction Complete.");

        // 6. Download Pivots
        let raw_pivots = self.device.dtoh_sync_copy(&d_pivots).unwrap();

        // Update host cache
        for (i, &p) in raw_pivots.iter().enumerate() {
            if p >= 0 {
                self.pivots[i] = Some(p as usize);
            } else {
                self.pivots[i] = None;
            }
        }
    }
}
