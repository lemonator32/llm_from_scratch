// A minimal reverse-mode autodiff engine over 2D f32 matrices.
//
// Design: every op eagerly computes its forward result and pushes a `Node`
// onto a shared `Tape`. Because inputs to a node always have a smaller index
// than the node itself (the graph is built in causal/topological order as we
// go), backward() can simply walk the tape in reverse and, for each node,
// push gradient contributions into its inputs' `grad` buffers. No separate
// topological sort is needed.

use std::cell::RefCell;
use std::rc::Rc;

pub type Shape = (usize, usize); // (rows, cols)

#[derive(Clone, Debug)]
pub enum Op {
    Leaf,
    Add(usize, usize),
    Scale(usize, f32),
    MatMul(usize, usize),
    Transpose(usize),
    Relu(usize),
    Softmax(usize),                    // row-wise softmax
    AddBias(usize, usize),             // (m,n) + broadcast (1,n)
    SliceCols(usize, usize, usize),    // (src, start_col, num_cols)
    Concat(Vec<usize>),                // concat along columns
    CausalMask(usize),                 // set upper-triangular (j>i) to -1e9
    LayerNorm(usize, usize, usize, f32), // (x, gamma, beta, eps) gamma/beta shape (1,d)
    Embedding(usize, Vec<usize>),      // (weight_id, row_indices) -> gathered rows
    CrossEntropy(usize, Vec<usize>),   // (logits, targets) -> scalar (1,1) mean loss
}

pub struct Node {
    pub op: Op,
    pub shape: Shape,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    // cached row-wise softmax outputs / layernorm stats used by backward,
    // stored so backward doesn't need to recompute forward math.
    pub cache: Vec<f32>,
}

pub struct Tape {
    pub nodes: Vec<Node>,
}

pub type TapeRef = Rc<RefCell<Tape>>;

pub fn new_tape() -> TapeRef {
    Rc::new(RefCell::new(Tape { nodes: Vec::new() }))
}

#[derive(Clone)]
pub struct Tensor {
    pub id: usize,
    pub tape: TapeRef,
}

fn push(tape: &TapeRef, op: Op, shape: Shape, data: Vec<f32>, cache: Vec<f32>) -> Tensor {
    let mut t = tape.borrow_mut();
    let id = t.nodes.len();
    let n = shape.0 * shape.1;
    t.nodes.push(Node {
        op,
        shape,
        data,
        grad: vec![0.0; n],
        cache,
    });
    Tensor { id, tape: tape.clone() }
}

impl Tensor {
    pub fn leaf(tape: &TapeRef, shape: Shape, data: Vec<f32>) -> Tensor {
        assert_eq!(data.len(), shape.0 * shape.1);
        push(tape, Op::Leaf, shape, data, vec![])
    }

    pub fn shape(&self) -> Shape {
        self.tape.borrow().nodes[self.id].shape
    }

    pub fn data(&self) -> Vec<f32> {
        self.tape.borrow().nodes[self.id].data.clone()
    }

    pub fn grad(&self) -> Vec<f32> {
        self.tape.borrow().nodes[self.id].grad.clone()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let (m, n) = self.shape();
        assert_eq!(other.shape(), (m, n), "add: shape mismatch");
        let a = self.data();
        let b = other.data();
        let data: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        push(&self.tape, Op::Add(self.id, other.id), (m, n), data, vec![])
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let shape = self.shape();
        let data: Vec<f32> = self.data().iter().map(|x| x * s).collect();
        push(&self.tape, Op::Scale(self.id, s), shape, data, vec![])
    }

    pub fn neg(&self) -> Tensor {
        self.scale(-1.0)
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k1) = self.shape();
        let (k2, n) = other.shape();
        assert_eq!(k1, k2, "matmul: inner dims mismatch ({},{})x({},{})", m, k1, k2, n);
        let a = self.data();
        let b = other.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k1 {
                let av = a[i * k1 + p];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] += av * b[p * n + j];
                }
            }
        }
        push(&self.tape, Op::MatMul(self.id, other.id), (m, n), out, vec![])
    }

    pub fn transpose(&self) -> Tensor {
        let (m, n) = self.shape();
        let a = self.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a[i * n + j];
            }
        }
        push(&self.tape, Op::Transpose(self.id), (n, m), out, vec![])
    }

    pub fn relu(&self) -> Tensor {
        let shape = self.shape();
        let data: Vec<f32> = self.data().iter().map(|x| x.max(0.0)).collect();
        push(&self.tape, Op::Relu(self.id), shape, data, vec![])
    }

    pub fn softmax(&self) -> Tensor {
        let (m, n) = self.shape();
        let a = self.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|x| (x - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
            for j in 0..n {
                out[i * n + j] = exps[j] / sum;
            }
        }
        let cache = out.clone(); // softmax output reused in backward
        push(&self.tape, Op::Softmax(self.id), (m, n), out, cache)
    }

    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        let (m, n) = self.shape();
        let (bm, bn) = bias.shape();
        assert_eq!((bm, bn), (1, n), "add_bias: bias must be (1,n)");
        let a = self.data();
        let b = bias.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = a[i * n + j] + b[j];
            }
        }
        push(&self.tape, Op::AddBias(self.id, bias.id), (m, n), out, vec![])
    }

    pub fn slice_cols(&self, start: usize, len: usize) -> Tensor {
        let (m, n) = self.shape();
        assert!(start + len <= n, "slice_cols out of range");
        let a = self.data();
        let mut out = vec![0.0f32; m * len];
        for i in 0..m {
            for j in 0..len {
                out[i * len + j] = a[i * n + start + j];
            }
        }
        push(&self.tape, Op::SliceCols(self.id, start, len), (m, len), out, vec![])
    }

    pub fn causal_mask(&self) -> Tensor {
        // assumes square (seq, seq) attention score matrix; masks j > i
        let (m, n) = self.shape();
        assert_eq!(m, n, "causal_mask expects a square matrix");
        let a = self.data();
        let mut out = a.clone();
        for i in 0..m {
            for j in 0..n {
                if j > i {
                    out[i * n + j] = -1e9;
                }
            }
        }
        push(&self.tape, Op::CausalMask(self.id), (m, n), out, vec![])
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let (m, n) = self.shape();
        assert_eq!(gamma.shape(), (1, n));
        assert_eq!(beta.shape(), (1, n));
        let x = self.data();
        let g = gamma.data();
        let b = beta.data();
        let mut out = vec![0.0f32; m * n];
        // cache layout per row: [mean, inv_std, xhat_0..xhat_{n-1}]
        let mut cache = vec![0.0f32; m * (2 + n)];
        for i in 0..m {
            let row = &x[i * n..i * n + n];
            let mean: f32 = row.iter().sum::<f32>() / n as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let cbase = i * (2 + n);
            cache[cbase] = mean;
            cache[cbase + 1] = inv_std;
            for j in 0..n {
                let xhat = (row[j] - mean) * inv_std;
                cache[cbase + 2 + j] = xhat;
                out[i * n + j] = xhat * g[j] + b[j];
            }
        }
        push(
            &self.tape,
            Op::LayerNorm(self.id, gamma.id, beta.id, eps),
            (m, n),
            out,
            cache,
        )
    }

    pub fn cross_entropy(&self, targets: &[usize]) -> Tensor {
        let (m, n) = self.shape();
        assert_eq!(targets.len(), m);
        let x = self.data();
        let mut loss = 0.0f32;
        let mut cache = vec![0.0f32; m * n]; // store per-row softmax probs
        for i in 0..m {
            let row = &x[i * n..i * n + n];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
            for j in 0..n {
                cache[i * n + j] = exps[j] / sum;
            }
            let p = cache[i * n + targets[i]].max(1e-12);
            loss += -p.ln();
        }
        loss /= m as f32;
        push(
            &self.tape,
            Op::CrossEntropy(self.id, targets.to_vec()),
            (1, 1),
            vec![loss],
            cache,
        )
    }
}

pub fn concat_cols(parts: &[Tensor]) -> Tensor {
    assert!(!parts.is_empty());
    let tape = parts[0].tape.clone();
    let m = parts[0].shape().0;
    let mut total_n = 0;
    for p in parts {
        assert_eq!(p.shape().0, m, "concat_cols: row mismatch");
        total_n += p.shape().1;
    }
    let mut out = vec![0.0f32; m * total_n];
    let mut ids = Vec::with_capacity(parts.len());
    let mut col_off = 0;
    for p in parts {
        let (_, n) = p.shape();
        let d = p.data();
        for i in 0..m {
            for j in 0..n {
                out[i * total_n + col_off + j] = d[i * n + j];
            }
        }
        ids.push(p.id);
        col_off += n;
    }
    push(&tape, Op::Concat(ids), (m, total_n), out, vec![])
}

pub fn embedding_lookup(weight: &Tensor, indices: &[usize]) -> Tensor {
    let (_vocab, d) = weight.shape();
    let w = weight.data();
    let mut out = vec![0.0f32; indices.len() * d];
    for (row, &idx) in indices.iter().enumerate() {
        out[row * d..row * d + d].copy_from_slice(&w[idx * d..idx * d + d]);
    }
    push(
        &weight.tape,
        Op::Embedding(weight.id, indices.to_vec()),
        (indices.len(), d),
        out,
        vec![],
    )
}

/// Run backward pass from a scalar (1,1) loss tensor, populating .grad on
/// every node reachable from it.
pub fn backward(loss: &Tensor) {
    let mut tape = loss.tape.borrow_mut();
    let n_nodes = tape.nodes.len();
    tape.nodes[loss.id].grad = vec![1.0];

    for i in (0..n_nodes).rev() {
        // Split so we can read node i immutably-ish while writing to earlier nodes.
        let (earlier, current_and_after) = tape.nodes.split_at_mut(i);
        let node = &current_and_after[0];
        let (m, n) = node.shape;
        let dout = node.grad.clone();
        if dout.iter().all(|&g| g == 0.0) {
            continue;
        }
        match &node.op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                for k in 0..dout.len() {
                    earlier[*a].grad[k] += dout[k];
                    earlier[*b].grad[k] += dout[k];
                }
            }
            Op::Scale(a, s) => {
                for k in 0..dout.len() {
                    earlier[*a].grad[k] += dout[k] * s;
                }
            }
            Op::MatMul(a, b) => {
                let (am, ak) = earlier[*a].shape;
                let (bk, bn) = earlier[*b].shape;
                debug_assert_eq!(ak, bk);
                debug_assert_eq!((am, bn), (m, n));
                let a_data = earlier[*a].data.clone();
                let b_data = earlier[*b].data.clone();
                // dA (am,ak) += dOut (am,bn) @ B^T (bn,ak)
                for i2 in 0..am {
                    for p in 0..ak {
                        let mut s = 0.0f32;
                        for j in 0..bn {
                            s += dout[i2 * bn + j] * b_data[p * bn + j];
                        }
                        earlier[*a].grad[i2 * ak + p] += s;
                    }
                }
                // dB (bk,bn) += A^T (bk,am) @ dOut (am,bn)
                for p in 0..bk {
                    for j in 0..bn {
                        let mut s = 0.0f32;
                        for i2 in 0..am {
                            s += a_data[i2 * ak + p] * dout[i2 * bn + j];
                        }
                        earlier[*b].grad[p * bn + j] += s;
                    }
                }
            }
            Op::Transpose(a) => {
                let (an, am) = (n, m); // node shape is (n_a_cols, n_a_rows) = transposed
                let _ = am;
                let (arows, acols) = earlier[*a].shape;
                for i2 in 0..arows {
                    for j in 0..acols {
                        earlier[*a].grad[i2 * acols + j] += dout[j * an + i2];
                    }
                }
            }
            Op::Relu(a) => {
                let a_data = &earlier[*a].data;
                for k in 0..dout.len() {
                    if a_data[k] > 0.0 {
                        earlier[*a].grad[k] += dout[k];
                    }
                }
            }
            Op::Softmax(a) => {
                let y = &node.cache; // softmax output, row-major (m,n)
                for i2 in 0..m {
                    let row = &y[i2 * n..i2 * n + n];
                    let drow = &dout[i2 * n..i2 * n + n];
                    let dot: f32 = row.iter().zip(drow.iter()).map(|(yv, dv)| yv * dv).sum();
                    for j in 0..n {
                        earlier[*a].grad[i2 * n + j] += row[j] * (drow[j] - dot);
                    }
                }
            }
            Op::AddBias(a, b) => {
                for k in 0..dout.len() {
                    earlier[*a].grad[k] += dout[k];
                }
                for i2 in 0..m {
                    for j in 0..n {
                        earlier[*b].grad[j] += dout[i2 * n + j];
                    }
                }
            }
            Op::SliceCols(a, start, len) => {
                let (_arows, acols) = earlier[*a].shape;
                for i2 in 0..m {
                    for j in 0..*len {
                        earlier[*a].grad[i2 * acols + start + j] += dout[i2 * len + j];
                    }
                }
            }
            Op::Concat(ids) => {
                let mut col_off = 0;
                for id in ids {
                    let (_, cn) = earlier[*id].shape;
                    for i2 in 0..m {
                        for j in 0..cn {
                            earlier[*id].grad[i2 * cn + j] += dout[i2 * n + col_off + j];
                        }
                    }
                    col_off += cn;
                }
            }
            Op::CausalMask(a) => {
                for i2 in 0..m {
                    for j in 0..n {
                        if j <= i2 {
                            earlier[*a].grad[i2 * n + j] += dout[i2 * n + j];
                        }
                    }
                }
            }
            Op::LayerNorm(a, g, b, eps) => {
                let _ = eps;
                let d = n as f32;
                let g_data = earlier[*g].data.clone();
                for i2 in 0..m {
                    let cbase = i2 * (2 + n);
                    let inv_std = node.cache[cbase + 1];
                    let xhat = &node.cache[cbase + 2..cbase + 2 + n];
                    let drow = &dout[i2 * n..i2 * n + n];
                    // dbeta, dgamma
                    for j in 0..n {
                        earlier[*b].grad[j] += drow[j];
                        earlier[*g].grad[j] += drow[j] * xhat[j];
                    }
                    // dxhat = drow * gamma
                    let dxhat: Vec<f32> = (0..n).map(|j| drow[j] * g_data[j]).collect();
                    let mean_dxhat: f32 = dxhat.iter().sum::<f32>() / d;
                    let mean_dxhat_xhat: f32 =
                        dxhat.iter().zip(xhat.iter()).map(|(dv, xv)| dv * xv).sum::<f32>() / d;
                    for j in 0..n {
                        let dx = inv_std * (dxhat[j] - mean_dxhat - xhat[j] * mean_dxhat_xhat);
                        earlier[*a].grad[i2 * n + j] += dx;
                    }
                }
            }
            Op::Embedding(w, indices) => {
                let (_vocab, d) = earlier[*w].shape;
                for (row, &idx) in indices.iter().enumerate() {
                    for j in 0..d {
                        earlier[*w].grad[idx * d + j] += dout[row * d + j];
                    }
                }
            }
            Op::CrossEntropy(logits, targets) => {
                let m2 = targets.len();
                let n2 = earlier[*logits].shape.1;
                let g0 = dout[0]; // usually 1.0
                for i2 in 0..m2 {
                    for j in 0..n2 {
                        let mut d = node.cache[i2 * n2 + j];
                        if j == targets[i2] {
                            d -= 1.0;
                        }
                        earlier[*logits].grad[i2 * n2 + j] += (d / m2 as f32) * g0;
                    }
                }
            }
        }
    }
}

pub fn zero_grad(tape: &TapeRef) {
    for node in tape.borrow_mut().nodes.iter_mut() {
        for g in node.grad.iter_mut() {
            *g = 0.0;
        }
    }
}
