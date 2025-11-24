use anyhow::Result;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

/// A transactional wrapper for splat file operations.
/// Ensures that writes to geometry and semantics files are atomic-ish.
/// If a write fails or is not committed, we rollback to the state at `begin()`.
pub struct SplatTransaction<'a> {
    pub geom_file: &'a mut File,
    pub sem_file: &'a mut File,
    pub phoneme_file: &'a mut File,

    // Track start positions to rollback on error
    pub geom_start: u64,
    pub sem_start: u64,
    pub phoneme_start: u64,
}

impl<'a> SplatTransaction<'a> {
    pub fn begin(geom: &'a mut File, sem: &'a mut File, phoneme: &'a mut File) -> Result<Self> {
        let geom_start = geom.metadata()?.len();
        let sem_start = sem.metadata()?.len();
        let phoneme_start = phoneme.metadata()?.len();

        // Ensure we are at the end of the files before starting
        geom.seek(SeekFrom::End(0))?;
        sem.seek(SeekFrom::End(0))?;
        phoneme.seek(SeekFrom::End(0))?;

        Ok(Self {
            geom_file: geom,
            sem_file: sem,
            phoneme_file: phoneme,
            geom_start,
            sem_start,
            phoneme_start,
        })
    }

    pub fn commit(self) -> Result<()> {
        self.geom_file.flush()?;
        self.sem_file.flush()?;
        self.phoneme_file.flush()?;
        // For raw append-only, flush is our "commit".
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<()> {
        // Truncate files back to their original length
        self.geom_file.set_len(self.geom_start)?;
        self.sem_file.set_len(self.sem_start)?;
        self.phoneme_file.set_len(self.phoneme_start)?;

        // Seek back to the original positions (though set_len might not move the cursor, it's safe to reset)
        self.geom_file.seek(SeekFrom::Start(self.geom_start))?;
        self.sem_file.seek(SeekFrom::Start(self.sem_start))?;
        self.phoneme_file
            .seek(SeekFrom::Start(self.phoneme_start))?;

        Ok(())
    }

    pub fn begin_phoneme_len(&self) -> Result<u64> {
        Ok(self.phoneme_start)
    }
}
