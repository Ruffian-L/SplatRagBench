use anyhow::{Context, Result};
use std::io::{Write, BufReader, BufRead};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

pub enum EmbeddingUsage {
    Query,
    Document,
    Tokens,
}

#[derive(Debug, Deserialize)]
struct DaemonResponseItem {
    pooled: Vec<f32>,
    #[serde(default)]
    token_embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    tokens: Vec<String>,
}

struct DaemonProcess {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

pub struct EmbeddingModel {
    daemon: Arc<Mutex<DaemonProcess>>,
}

impl EmbeddingModel {
    pub fn new(_model_repo: &str, use_gpu: bool) -> Result<Self> {
        println!("🔌 Spawning Nomic Python Daemon...");
        
        let mut cmd = Command::new("./venv/bin/python3");
        cmd.arg("src/nomic_daemon.py");
        
        if !use_gpu {
            cmd.arg("--cpu");
        }

        // Use absolute path or relative to CWD. CWD is workspace root.
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Let Python logs show in terminal
            .spawn()
            .context("Failed to spawn nomic_daemon.py")?;

        let stdin = child.stdin.take().context("Failed to open stdin")?;
        let stdout = child.stdout.take().context("Failed to open stdout")?;
        let reader = BufReader::new(stdout);

        Ok(Self {
            daemon: Arc::new(Mutex::new(DaemonProcess {
                child,
                stdin,
                reader
            }))
        })
    }

    fn call_daemon(&self, texts: &[String], usage: EmbeddingUsage) -> Result<Vec<DaemonResponseItem>> {
        let mode = match usage {
            EmbeddingUsage::Query => "search_query",
            EmbeddingUsage::Document => "search_document",
            EmbeddingUsage::Tokens => "embed_tokens",
        };

        let payload = serde_json::json!({
            "texts": texts,
            "mode": mode
        }).to_string();

        let mut daemon = self.daemon.lock().map_err(|_| anyhow::anyhow!("Failed to lock daemon"))?;
        
        writeln!(daemon.stdin, "{}", payload)?;
        daemon.stdin.flush()?;
        
        let mut response = String::new();
        daemon.reader.read_line(&mut response)?;

        if response.trim().is_empty() {
             anyhow::bail!("Empty response from daemon");
        }

        let items: Vec<DaemonResponseItem> = serde_json::from_str(&response)?;
        Ok(items)
    }

    fn normalize(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in v { *x /= norm; }
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Query)?;
        let mut emb = items[0].pooled.clone();
        emb.truncate(64);
        Self::normalize(&mut emb);
        Ok(emb)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Query)?;
        Ok(items.into_iter().map(|i| i.pooled).collect())
    }

    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Document)?;
        let mut emb = items[0].pooled.clone();
        emb.truncate(64);
        Self::normalize(&mut emb);
        Ok(emb)
    }

    pub fn embed_batch_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Document)?;
        let results = items.into_iter().map(|item| {
            let mut emb = item.pooled;
            emb.truncate(64);
            Self::normalize(&mut emb);
            emb
        }).collect();
        Ok(results)
    }

    pub fn embed_tokens(&self, text: &str) -> Result<(Vec<Vec<f32>>, Vec<String>)> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Tokens)?;
        let item = &items[0];
        
        let sliced_tokens: Vec<Vec<f32>> = item.token_embeddings.iter().map(|t| {
            let mut t = t.clone();
            t.truncate(64);
            // Tokens might not need normalization for PCA, but let's keep it raw?
            // Existing code didn't normalize tokens, only pooled.
            t
        }).collect();

        Ok((sliced_tokens, item.tokens.clone()))
    }

    pub fn embed_batch_tokens(&self, texts: &[String]) -> Result<Vec<(Vec<f32>, Vec<Vec<f32>>, Vec<String>)>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Tokens)?;
        
        let results = items.into_iter().map(|item| {
            let mut pooled = item.pooled;
            pooled.truncate(64);
            Self::normalize(&mut pooled);
            
            let tokens = item.token_embeddings.into_iter().map(|t| {
                let mut t = t;
                t.truncate(64);
                t
            }).collect();

            (pooled, tokens, item.tokens)
        }).collect();
        
        Ok(results)
    }
}
