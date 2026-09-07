use crate::tensor::{backward as tensor_backward, concat_cols, embedding_lookup, new_tape, Tensor};
use rand::Rng;
use std::collections::HashMap;

/// Persistent parameter storage. Lives *outside* the tape, because the tape
/// is rebuilt fresh on every forward pass (like PyTorch's dynamic graph).
/// Each Param also carries its own Adam moment buffers.
pub struct Param {
    pub data: Vec<f32>,
    pub shape: (usize, usize),
    pub m: Vec<f32>, // Adam first moment
    pub v: Vec<f32>, // Adam second moment
}

impl Param {
    fn zeros(shape: (usize, usize)) -> Self {
        let n = shape.0 * shape.1;
        Param { data: vec![0.0; n], shape, m: vec![0.0; n], v: vec![0.0; n] }
    }

    fn randn(shape: (usize, usize), std: f32, rng: &mut impl Rng) -> Self {
        let n = shape.0 * shape.1;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                // Box-Muller
                let u1: f32 = rng.gen_range(1e-6..1.0);
                let u2: f32 = rng.gen_range(0.0..1.0);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
            })
            .collect();
        Param { data, shape, m: vec![0.0; n], v: vec![0.0; n] }
    }

    fn ones(shape: (usize, usize)) -> Self {
        let n = shape.0 * shape.1;
        Param { data: vec![1.0; n], shape, m: vec![0.0; n], v: vec![0.0; n] }
    }
}

/// Named collection of parameters, in a fixed insertion order so we can
/// deterministically re-materialize them onto a fresh tape each step.
pub struct ParamStore {
    order: Vec<String>,
    map: HashMap<String, Param>,
}

impl ParamStore {
    fn new() -> Self {
        ParamStore { order: vec![], map: HashMap::new() }
    }
    fn insert(&mut self, name: &str, p: Param) {
        self.order.push(name.to_string());
        self.map.insert(name.to_string(), p);
    }
    fn get(&self, name: &str) -> &Param {
        self.map.get(name).unwrap()
    }
    fn get_mut(&mut self, name: &str) -> &mut Param {
        self.map.get_mut(name).unwrap()
    }
}

pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

pub struct GPT {
    pub cfg: GPTConfig,
    pub params: ParamStore,
}

fn linear_name(prefix: &str, suffix: &str) -> String {
    format!("{}.{}", prefix, suffix)
}

impl GPT {
    pub fn new(cfg: GPTConfig, rng: &mut impl Rng) -> Self {
        let mut params = ParamStore::new();
        let d = cfg.n_embd;
        let std = 0.02;

        params.insert("wte", Param::randn((cfg.vocab_size, d), std, rng));
        params.insert("wpe", Param::randn((cfg.block_size, d), std, rng));

        for l in 0..cfg.n_layer {
            let p = format!("h{}", l);
            params.insert(&linear_name(&p, "ln1.g"), Param::ones((1, d)));
            params.insert(&linear_name(&p, "ln1.b"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "wq"), Param::randn((d, d), std, rng));
            params.insert(&linear_name(&p, "bq"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "wk"), Param::randn((d, d), std, rng));
            params.insert(&linear_name(&p, "bk"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "wv"), Param::randn((d, d), std, rng));
            params.insert(&linear_name(&p, "bv"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "wo"), Param::randn((d, d), std, rng));
            params.insert(&linear_name(&p, "bo"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "ln2.g"), Param::ones((1, d)));
            params.insert(&linear_name(&p, "ln2.b"), Param::zeros((1, d)));
            params.insert(&linear_name(&p, "w1"), Param::randn((d, 4 * d), std, rng));
            params.insert(&linear_name(&p, "b1"), Param::zeros((1, 4 * d)));
            params.insert(&linear_name(&p, "w2"), Param::randn((4 * d, d), std, rng));
            params.insert(&linear_name(&p, "b2"), Param::zeros((1, d)));
        }
        params.insert("lnf.g", Param::ones((1, d)));
        params.insert("lnf.b", Param::zeros((1, d)));

        GPT { cfg, params }
    }

    /// Forward pass. Builds a brand new tape, materializes every parameter
    /// as a leaf tensor on it, runs the transformer, and (if targets given)
    /// returns the cross-entropy loss tensor; otherwise returns logits.
    /// Also returns a map of param-name -> leaf tensor so the caller can
    /// read `.grad()` after backward() and update `ParamStore`.
    pub fn forward(
        &self,
        ids: &[usize],
        targets: Option<&[usize]>,
    ) -> (Tensor, HashMap<String, Tensor>) {
        let tape = new_tape();
        let mut leaves: HashMap<String, Tensor> = HashMap::new();
        let leaf = |name: &str, params: &ParamStore, leaves: &mut HashMap<String, Tensor>| {
            let p = params.get(name);
            let t = Tensor::leaf(&tape, p.shape, p.data.clone());
            leaves.insert(name.to_string(), t.clone());
            t
        };

        let seq = ids.len();
        let d = self.cfg.n_embd;
        let n_head = self.cfg.n_head;
        let dk = d / n_head;

        let wte = leaf("wte", &self.params, &mut leaves);
        let wpe = leaf("wpe", &self.params, &mut leaves);

        let tok_emb = embedding_lookup(&wte, ids);
        let positions: Vec<usize> = (0..seq).collect();
        let pos_emb = embedding_lookup(&wpe, &positions);
        let mut x = tok_emb.add(&pos_emb);

        for l in 0..self.cfg.n_layer {
            let p = format!("h{}", l);
            let ln1_g = leaf(&linear_name(&p, "ln1.g"), &self.params, &mut leaves);
            let ln1_b = leaf(&linear_name(&p, "ln1.b"), &self.params, &mut leaves);
            let wq = leaf(&linear_name(&p, "wq"), &self.params, &mut leaves);
            let bq = leaf(&linear_name(&p, "bq"), &self.params, &mut leaves);
            let wk = leaf(&linear_name(&p, "wk"), &self.params, &mut leaves);
            let bk = leaf(&linear_name(&p, "bk"), &self.params, &mut leaves);
            let wv = leaf(&linear_name(&p, "wv"), &self.params, &mut leaves);
            let bv = leaf(&linear_name(&p, "bv"), &self.params, &mut leaves);
            let wo = leaf(&linear_name(&p, "wo"), &self.params, &mut leaves);
            let bo = leaf(&linear_name(&p, "bo"), &self.params, &mut leaves);
            let ln2_g = leaf(&linear_name(&p, "ln2.g"), &self.params, &mut leaves);
            let ln2_b = leaf(&linear_name(&p, "ln2.b"), &self.params, &mut leaves);
            let w1 = leaf(&linear_name(&p, "w1"), &self.params, &mut leaves);
            let b1 = leaf(&linear_name(&p, "b1"), &self.params, &mut leaves);
            let w2 = leaf(&linear_name(&p, "w2"), &self.params, &mut leaves);
            let b2 = leaf(&linear_name(&p, "b2"), &self.params, &mut leaves);

            let ln1 = x.layer_norm(&ln1_g, &ln1_b, 1e-5);
            let q = ln1.matmul(&wq).add_bias(&bq);
            let k = ln1.matmul(&wk).add_bias(&bk);
            let v = ln1.matmul(&wv).add_bias(&bv);

            let mut heads = Vec::with_capacity(n_head);
            let scale = 1.0 / (dk as f32).sqrt();
            for h in 0..n_head {
                let qh = q.slice_cols(h * dk, dk);
                let kh = k.slice_cols(h * dk, dk);
                let vh = v.slice_cols(h * dk, dk);
                let scores = qh.matmul(&kh.transpose()).scale(scale);
                let masked = scores.causal_mask();
                let attn = masked.softmax();
                let out_h = attn.matmul(&vh);
                heads.push(out_h);
            }
            let concat = concat_cols(&heads);
            let proj = concat.matmul(&wo).add_bias(&bo);
            x = x.add(&proj);

            let ln2 = x.layer_norm(&ln2_g, &ln2_b, 1e-5);
            let h1 = ln2.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2);
            x = x.add(&h2);
        }

        let lnf_g = leaf("lnf.g", &self.params, &mut leaves);
        let lnf_b = leaf("lnf.b", &self.params, &mut leaves);
        let xf = x.layer_norm(&lnf_g, &lnf_b, 1e-5);
        // weight tying: reuse token embedding table as the output projection
        let logits = xf.matmul(&wte.transpose());

        match targets {
            Some(t) => (logits.cross_entropy(t), leaves),
            None => (logits, leaves),
        }
    }

    /// One Adam step: forward, backward, update every parameter, return loss.
    pub fn train_step(&mut self, ids: &[usize], targets: &[usize], opt: &mut Adam) -> f32 {
        let (loss, leaves) = self.forward(ids, Some(targets));
        tensor_backward(&loss);
        let loss_val = loss.data()[0];

        for name in leaves.keys() {
            let grad = leaves[name].grad();
            let param = self.params.get_mut(name);
            opt.step(param, &grad);
        }
        loss_val
    }

    pub fn generate(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let mut ids = prompt_ids.to_vec();
        for _ in 0..max_new_tokens {
            let ctx_start = ids.len().saturating_sub(self.cfg.block_size);
            let ctx = &ids[ctx_start..];
            let (logits, _) = self.forward(ctx, None);
            let (seq, vocab) = logits.shape();
            let last_row = &logits.data()[(seq - 1) * vocab..seq * vocab];
            let scaled: Vec<f32> = last_row.iter().map(|x| x / temperature).collect();
            let maxv = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|x| (x - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

            let r: f32 = rng.gen_range(0.0..1.0);
            let mut acc = 0.0;
            let mut chosen = probs.len() - 1;
            for (i, p) in probs.iter().enumerate() {
                acc += p;
                if r <= acc {
                    chosen = i;
                    break;
                }
            }
            ids.push(chosen);
        }
        ids
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: i32,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0 }
    }

    pub fn step(&mut self, param: &mut Param, grad: &[f32]) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for i in 0..param.data.len() {
            let g = grad[i];
            param.m[i] = self.beta1 * param.m[i] + (1.0 - self.beta1) * g;
            param.v[i] = self.beta2 * param.v[i] + (1.0 - self.beta2) * g * g;
            let mhat = param.m[i] / bc1;
            let vhat = param.v[i] / bc2;
            param.data[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}
