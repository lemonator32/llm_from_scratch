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
    let mut opt = Adam::new(3e-3);

    // ---- Train ----
    // Override with `STEPS=5000 ./llm_from_scratch` for a longer run.
    let steps: i32 = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let start = Instant::now();
    for step in 0..steps {
        let i = rng.gen_range(0..data.len() - block_size - 1);
        let ids = &data[i..i + block_size];
        let targets = &data[i + 1..i + block_size + 1];
        let loss = model.train_step(ids, targets, &mut opt);

        if step % 50 == 0 || step == steps - 1 {
            println!(
                "step {:4}  loss {:.4}  ({:.1}s elapsed)",
                step,
                loss,
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
