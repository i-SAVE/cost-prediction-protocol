# Block B: GATv2 vs MLP ablation on controlled synthetic WBS graph (300 nodes)
# JAX implementation of GATv2 (Brody et al., 2022), matching PyG GATv2Conv semantics
# (separate source/target linear maps, attention after LeakyReLU, self-loops added).
# Fixes vs original notebook: (1) no target leakage; (2) target has genuine graph
# dependence by construction; (3) 10 repeated splits -> mean+/-std + Wilcoxon.
import numpy as np, pandas as pd, os, json, sys, time, pickle, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import networkx as nx
import jax, jax.numpy as jnp
from sklearn.metrics import r2_score
from scipy import stats as sp_stats

OUT = 'results_rerun'; CK = OUT
os.makedirs(f'{OUT}/gnn', exist_ok=True); os.makedirs(CK, exist_ok=True)
SEED = 42
T0 = time.perf_counter()
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9

N_NODES, N_PHASES = 300, 6
rng_g = np.random.default_rng(SEED)
ctypes = ['A','B','C','D','E']
wtypes = ['earth','foundation','structure','envelope','mep','finish']
node_attrs = []
for i in range(N_NODES):
    ph = min(i // (N_NODES//N_PHASES), N_PHASES-1)
    node_attrs.append({'node_id':i,'phase':ph,'work_type':wtypes[ph%len(wtypes)],
        'contractor':rng_g.choice(ctypes),'cost_plan':int(rng_g.integers(500,50000)),
        'duration_days':int(rng_g.integers(3,90)),'complexity':int(rng_g.integers(1,6)),
        'n_stakeholders':int(rng_g.integers(2,10)),'risk_level':int(rng_g.integers(1,5))})
nodes_df = pd.DataFrame(node_attrs)
G = nx.DiGraph(); G.add_nodes_from((i, a) for i, a in enumerate(node_attrs))
el = []
for ph in range(N_PHASES):
    pn = [i for i in range(N_NODES) if node_attrs[i]['phase']==ph]
    for j in range(len(pn)-1):
        if rng_g.random() > 0.3: el.append((pn[j], pn[j+1]))
    if ph < N_PHASES-1:
        npn = [i for i in range(N_NODES) if node_attrs[i]['phase']==ph+1]
        for _ in range(rng_g.integers(2,5)):
            el.append((int(rng_g.choice(pn[-5:])), int(rng_g.choice(npn[:5]))))
el = list(set((u,v) for u,v in el if u != v))
G.add_edges_from(el)
while not nx.is_directed_acyclic_graph(G):
    c = next(nx.simple_cycles(G)); G.remove_edge(c[0], c[1])
    el = [(u,v) for u,v in el if not (u==c[0] and v==c[1])]

comp = nodes_df['complexity'].values.astype(float)
risk = nodes_df['risk_level'].values.astype(float)
stak = nodes_df['n_stakeholders'].values.astype(float)
dur  = nodes_df['duration_days'].values.astype(float)
cost = nodes_df['cost_plan'].values.astype(float)
own_signal = 0.5*comp + 0.4*risk + 0.15*stak + 0.2*np.log1p(dur)
upstream = np.zeros(N_NODES)
for i in range(N_NODES):
    preds = list(G.predecessors(i))
    if preds: upstream[i] = np.mean([0.5*comp[j] + 0.4*risk[j] for j in preds])
GAMMA = 0.9
rng_y = np.random.default_rng(SEED + 7)
y_all = own_signal + GAMMA*upstream + rng_y.normal(0, 0.5, N_NODES)
up_share = float(np.var(GAMMA*upstream)/np.var(y_all))

cmap = {c: float(k) for k, c in enumerate(ctypes)}
cont = nodes_df['contractor'].map(cmap).values.astype(float)
ph_arr = nodes_df['phase'].values.astype(float)
feat_raw = np.stack([cost, dur, comp, stak, risk, cont, ph_arr], axis=1)

# adjacency mask with self-loops: M[i, j] = 1 if message j -> i allowed
M = np.zeros((N_NODES, N_NODES), dtype=bool)
for u, v in G.edges(): M[v, u] = True
np.fill_diagonal(M, True)
Mj = jnp.array(M)
NEG = -1e9

HIDDEN, HEADS, DROPOUT = 32, 2, 0.3
EPOCHS, LR0, PATIENCE, LR_PAT = 500, 0.005, 40, 15
IN_CH = feat_raw.shape[1]

def init_gat(key):
    ks = jax.random.split(key, 8)
    def gl(k, a, b): return jax.random.normal(k, (a, b)) * jnp.sqrt(2.0/(a+b))
    return {'Wl1': gl(ks[0], IN_CH, HIDDEN*HEADS), 'Wr1': gl(ks[1], IN_CH, HIDDEN*HEADS),
            'a1': gl(ks[2], HEADS, HIDDEN), 'b1': jnp.zeros(HIDDEN*HEADS),
            'Wl2': gl(ks[3], HIDDEN*HEADS, HIDDEN), 'Wr2': gl(ks[4], HIDDEN*HEADS, HIDDEN),
            'a2': gl(ks[5], 1, HIDDEN), 'b2': jnp.zeros(HIDDEN),
            'Wo': gl(ks[6], HIDDEN, 1), 'bo': jnp.zeros(1)}

def gat_layer(x, Wl, Wr, a, b, heads, hid, key, training):
    hl = (x @ Wl).reshape(-1, heads, hid)   # source transform (messages)
    hr = (x @ Wr).reshape(-1, heads, hid)   # target transform
    # e[i,j,h] = a_h . leaky_relu(hl[j,h] + hr[i,h])
    s = hl[None, :, :, :] + hr[:, None, :, :]
    s = jax.nn.leaky_relu(s, 0.2)
    e = jnp.einsum('ijhd,hd->ijh', s, a)
    e = jnp.where(Mj[:, :, None], e, NEG)
    alpha = jax.nn.softmax(e, axis=1)
    if training:
        key, k2 = jax.random.split(key)
        alpha = alpha * jax.random.bernoulli(k2, 1-DROPOUT, alpha.shape) / (1-DROPOUT)
    out = jnp.einsum('ijh,jhd->ihd', alpha, hl)
    return out.reshape(x.shape[0], heads*hid) + b, key

def gat_fwd(p, x, key, training):
    if training:
        key, k = jax.random.split(key)
        x = x * jax.random.bernoulli(k, 1-DROPOUT, x.shape) / (1-DROPOUT)
    h, key = gat_layer(x, p['Wl1'], p['Wr1'], p['a1'], p['b1'], HEADS, HIDDEN, key, training)
    h = jax.nn.elu(h)
    if training:
        key, k = jax.random.split(key)
        h = h * jax.random.bernoulli(k, 1-DROPOUT, h.shape) / (1-DROPOUT)
    h, key = gat_layer(h, p['Wl2'], p['Wr2'], p['a2'], p['b2'], 1, HIDDEN, key, training)
    h = jax.nn.elu(h)
    return (h @ p['Wo'] + p['bo']).squeeze(-1)

def init_mlp(key):
    ks = jax.random.split(key, 3)
    def gl(k, a, b): return jax.random.normal(k, (a, b)) * jnp.sqrt(2.0/(a+b))
    return {'W1': gl(ks[0], IN_CH, HIDDEN*2), 'b1': jnp.zeros(HIDDEN*2),
            'W2': gl(ks[1], HIDDEN*2, HIDDEN), 'b2': jnp.zeros(HIDDEN),
            'W3': gl(ks[2], HIDDEN, 1), 'b3': jnp.zeros(1)}

def mlp_fwd(p, x, key, training):
    if training:
        key, k = jax.random.split(key); x = x * jax.random.bernoulli(k, 1-DROPOUT, x.shape)/(1-DROPOUT)
    h = jax.nn.elu(x @ p['W1'] + p['b1'])
    if training:
        key, k = jax.random.split(key); h = h * jax.random.bernoulli(k, 1-DROPOUT, h.shape)/(1-DROPOUT)
    h = jax.nn.elu(h @ p['W2'] + p['b2'])
    return (h @ p['W3'] + p['b3']).squeeze(-1)

def make_train(fwd):
    def loss_fn(p, x, y, mask, key, training):
        o = fwd(p, x, key, training)
        return jnp.sum(jnp.where(mask, (o-y)**2, 0.0)) / jnp.sum(mask)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnames=('training',))
    eval_fn = jax.jit(lambda p, x, y, mask: jnp.sum(jnp.where(mask, (fwd(p, x, jax.random.PRNGKey(0), False)-y)**2, 0.0))/jnp.sum(mask))
    def train(params, x, y, trm, vlm, key):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        lr, best_vl, ni, nlr, t = LR0, np.inf, 0, 0, 0
        best = params
        hist = {'tr': [], 'vl': []}
        for ep in range(1, EPOCHS+1):
            key, k = jax.random.split(key)
            l, g = grad_fn(params, x, y, trm, k, training=True)
            # weight decay (L2, torch-Adam style: added to grads)
            g = jax.tree.map(lambda gi, pi: gi + 1e-4*pi, g, params)
            gn = jnp.sqrt(sum(jnp.sum(gi**2) for gi in jax.tree.leaves(g)))
            scale = jnp.minimum(1.0, 1.0/(gn+1e-12))
            g = jax.tree.map(lambda gi: gi*scale, g)
            t += 1
            m = jax.tree.map(lambda mi, gi: 0.9*mi + 0.1*gi, m, g)
            v = jax.tree.map(lambda vi, gi: 0.999*vi + 0.001*gi**2, v, g)
            mh = jax.tree.map(lambda mi: mi/(1-0.9**t), m)
            vh = jax.tree.map(lambda vi: vi/(1-0.999**t), v)
            params = jax.tree.map(lambda pi, mi, vi: pi - lr*mi/(jnp.sqrt(vi)+1e-8), params, mh, vh)
            vl = float(eval_fn(params, x, y, vlm))
            hist['tr'].append(float(l)); hist['vl'].append(vl)
            if vl < best_vl - 1e-5:
                best_vl, best, ni, nlr = vl, params, 0, 0
            else:
                ni += 1; nlr += 1
                if nlr >= LR_PAT: lr = max(lr*0.5, 1e-5); nlr = 0
                if ni >= PATIENCE: break
        return best, hist
    return train, (lambda p, x: np.array(fwd(p, x, jax.random.PRNGKey(0), False)))

train_gat, pred_gat = make_train(gat_fwd)
train_mlp, pred_mlp = make_train(mlp_fwd)

N_SEEDS = 10
phases = nodes_df['phase'].values
ckf = f'{CK}/blockb_state.pkl'
state = pickle.load(open(ckf,'rb')) if os.path.exists(ckf) else {'rows': [], 'curves': None}
s = len(state['rows'])
while s < N_SEEDS and time.perf_counter()-T0 < BUDGET:
    rng_s = np.random.default_rng(1000 + s)
    tr_i, vl_i, te_i = [], [], []
    for pid in range(N_PHASES):
        pn = np.where(phases==pid)[0].copy(); rng_s.shuffle(pn)
        ntr, nvl = int(len(pn)*0.6), int(len(pn)*0.2)
        tr_i += list(pn[:ntr]); vl_i += list(pn[ntr:ntr+nvl]); te_i += list(pn[ntr+nvl:])
    trm = np.zeros(N_NODES, bool); trm[tr_i] = True
    vlm = np.zeros(N_NODES, bool); vlm[vl_i] = True
    tem = np.zeros(N_NODES, bool); tem[te_i] = True
    fm, fs = feat_raw[trm].mean(0), feat_raw[trm].std(0)+1e-8
    x = jnp.array((feat_raw-fm)/fs, dtype=jnp.float32)
    ym, ys = y_all[trm].mean(), y_all[trm].std()+1e-8
    y = jnp.array((y_all-ym)/ys, dtype=jnp.float32)
    trmj, vlmj = jnp.array(trm), jnp.array(vlm)
    key = jax.random.PRNGKey(1000+s)
    kg, km, kt = jax.random.split(key, 3)
    pg, hg = train_gat(init_gat(kg), x, y, trmj, vlmj, kt)
    pm, hm = train_mlp(init_mlp(km), x, y, trmj, vlmj, kt)
    r_g = float(r2_score(np.array(y)[tem], pred_gat(pg, x)[tem]))
    r_m = float(r2_score(np.array(y)[tem], pred_mlp(pm, x)[tem]))
    state['rows'].append({'seed':1000+s,'GATv2_R2':r_g,'MLP_R2':r_m,'dR2':r_g-r_m,'n_test':int(tem.sum())})
    if s == 0: state['curves'] = (hg, hm)
    print(f'seed {1000+s}: GATv2 {r_g:.4f} | MLP {r_m:.4f} | d {r_g-r_m:+.4f}')
    s += 1
    pickle.dump(state, open(ckf,'wb'))

if s < N_SEEDS:
    print(f'progress {s}/{N_SEEDS}'); sys.exit(0)

res = pd.DataFrame(state['rows'])
res.to_csv(f'{OUT}/gnn/ablation_seeds.csv', index=False)
w, p = sp_stats.wilcoxon(res['GATv2_R2'], res['MLP_R2'], alternative='greater')
n_par_g = sum(int(np.prod(v.shape)) for v in init_gat(jax.random.PRNGKey(0)).values())
n_par_m = sum(int(np.prod(v.shape)) for v in init_mlp(jax.random.PRNGKey(0)).values())
summary = {'n_seeds': N_SEEDS,
    'GATv2_R2_mean': float(res['GATv2_R2'].mean()), 'GATv2_R2_std': float(res['GATv2_R2'].std()),
    'MLP_R2_mean': float(res['MLP_R2'].mean()), 'MLP_R2_std': float(res['MLP_R2'].std()),
    'dR2_mean': float(res['dR2'].mean()), 'dR2_std': float(res['dR2'].std()),
    'wilcoxon_W': float(w), 'wilcoxon_p': float(p),
    'graph': {'n_nodes': N_NODES, 'n_edges': G.number_of_edges(), 'n_phases': N_PHASES,
              'is_dag': bool(nx.is_directed_acyclic_graph(G))},
    'target_upstream_variance_share': up_share, 'gamma': GAMMA,
    'gat_params': n_par_g, 'mlp_params': n_par_m,
    'hyperparams': {'hidden': HIDDEN, 'heads': HEADS, 'dropout': DROPOUT,
                    'epochs_max': EPOCHS, 'lr': LR0, 'patience': PATIENCE,
                    'weight_decay': 1e-4, 'grad_clip': 1.0},
    'env': {'jax': jax.__version__}}
json.dump(summary, open(f'{OUT}/gnn/gnn_results.json','w'), indent=2)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))

hg, hm = state['curves']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hg['tr'], color='#8E44AD', lw=1.2, label='GATv2 train')
axes[0].plot(hg['vl'], color='#8E44AD', lw=1.2, ls='--', label='GATv2 val')
axes[0].plot(hm['tr'], color='#E67E22', lw=1.2, label='MLP train')
axes[0].plot(hm['vl'], color='#E67E22', lw=1.2, ls='--', label='MLP val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE (normalised target)')
axes[0].set_title('(a) Training dynamics (split 1)'); axes[0].legend(); axes[0].grid(alpha=0.3)
bp = axes[1].boxplot([res['MLP_R2'], res['GATv2_R2']], labels=['MLP (no graph)', 'GATv2 (graph)'],
                     patch_artist=True, widths=0.5)
for patch, c in zip(bp['boxes'], ['#E67E22', '#8E44AD']): patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1].set_ylabel('Test R²')
axes[1].set_title(f'(b) {N_SEEDS} repeated splits: ΔR² = {res["dR2"].mean():+.3f} ± {res["dR2"].std():.3f}')
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/gnn/ablation_results.png', dpi=300, bbox_inches='tight'); plt.close()

# risk-weighted critical path demo (networkx)
risk_w = {1:0.1, 2:0.2, 3:0.35, 4:0.5, 5:0.65}
H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H2 = nx.DiGraph(); H2.add_nodes_from(G.nodes())
for u, v in G.edges():
    H.add_edge(u, v, w=dur[u]*(1+risk_w.get(int(risk[u]), 0.3)))
    H2.add_edge(u, v, w=dur[u])
cp = {'critical_path_nodes': len(nx.dag_longest_path(H, weight='w')),
      'risk_weighted_length_days': round(float(nx.dag_longest_path_length(H, weight='w')), 1),
      'plain_length_days': round(float(nx.dag_longest_path_length(H2, weight='w')), 1)}
json.dump(cp, open(f'{OUT}/gnn/critical_path.json','w'), indent=2)
print('Critical path:', cp)
print('BLOCK B COMPLETE')
