mod nn;
mod tensor;

use nn::{Adam, GPT, GPTConfig};
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

struct CharTokenizer {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
}

impl CharTokenizer {
    fn build(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort();
        let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { stoi, itos: chars }
    }
    fn encode(&self, s: &str) -> Vec<usize> {
        s.chars().map(|c| self.stoi[&c]).collect()
    }
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.itos[i]).collect()
    }
    fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // ---- Load data ----
    let path = "data/input.txt";
    let text = fs::read_to_string(path).unwrap_or_else(|_| {
        panic!("Couldn't read {}. Put some training text there first.", path)
    });
    // Cap corpus size so training stays fast on pure-Rust hand-rolled matmul.
    let cap = 100_000.min(text.len());
    let text = &text[..cap];
    println!("Loaded {} characters of training text.", text.len());

    let tok = CharTokenizer::build(text);
    println!("Vocab size: {}", tok.vocab_size());
    let data = tok.encode(text);

    // ---- Config ----
    let block_size = 64;
    let cfg = GPTConfig {
        vocab_size: tok.vocab_size(),
        block_size,
        n_embd: 64,
        n_head: 4,
        n_layer: 3,
    };
    println!(
        "Model: {} layers, {} heads, {} dim, block_size {}",
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size
    );

    let mut model = GPT::new(cfg, &mut rng);
    let base_lr = 3e-3;
    let mut opt = Adam::new(base_lr);

    // ---- Train ----
    // `STEPS` = number of optimizer *updates*. Each update accumulates
    // gradients over `ACCUM` sequences before applying them, which averages
    // out per-sequence noise (equivalent to a bigger batch size).
    let updates: i32 = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let accum_steps: i32 = std::env::var("ACCUM").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let warmup_updates = (updates / 20).max(1); // first 5% of training

    fn lr_schedule(step: i32, total: i32, warmup: i32, base_lr: f32) -> f32 {
        if step < warmup {
            // linear warmup from ~0 to base_lr
            base_lr * (step as f32 + 1.0) / warmup as f32
        } else {
            // cosine decay from base_lr down to 10% of base_lr
            let min_lr = base_lr * 0.1;
            let progress = (step - warmup) as f32 / (total - warmup).max(1) as f32;
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    println!(
        "Training: {} updates x {} accumulated sequences each ({} total forward/backward passes)",
        updates,
        accum_steps,
        updates * accum_steps
    );

    let start = Instant::now();
    for update in 0..updates {
        opt.lr = lr_schedule(update, updates, warmup_updates, base_lr);

        // Accumulate gradients over `accum_steps` random sequences.
        let mut acc_grads: HashMap<String, Vec<f32>> = HashMap::new();
        let mut acc_loss = 0.0f32;
        for _ in 0..accum_steps {
            let i = rng.gen_range(0..data.len() - block_size - 1);
            let ids = &data[i..i + block_size];
            let targets = &data[i + 1..i + block_size + 1];
            let (loss, grads) = model.forward_backward(ids, targets);
            acc_loss += loss;
            for (name, g) in grads {
                let entry = acc_grads.entry(name).or_insert_with(|| vec![0.0; g.len()]);
                for (e, v) in entry.iter_mut().zip(g.iter()) {
                    *e += v;
                }
            }
        }
        for g in acc_grads.values_mut() {
            for v in g.iter_mut() {
                *v /= accum_steps as f32;
            }
        }
        acc_loss /= accum_steps as f32;
        model.apply_grads(&acc_grads, &mut opt);

        if update % 50 == 0 || update == updates - 1 {
            println!(
                "update {:4}  loss {:.4}  lr {:.5}  ({:.1}s elapsed)",
                update,
                acc_loss,
                opt.lr,
                start.elapsed().as_secs_f32()
            );
        }
    }

    // ---- Generate ----
    println!("\n--- Sample generation ---");
    let prompt = "\n";
    let prompt_ids = tok.encode(prompt);
    let out_ids = model.generate(&prompt_ids, 300, 0.8, &mut rng);
    println!("{}", tok.decode(&out_ids));
}
