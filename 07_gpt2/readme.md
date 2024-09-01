# Pre-train GPT-2 with KGI

### 1. Modify `modeling_gpt2.py`

In `transformers/models/gpt2/modeling_gpt2.py`, change all `Conv1D` to `nn.Linear` and swap the order of the two
arguments.

For example, change

```python
self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
```

to

```python
self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
```

After this, reinstall your local version of `transformers`.

### 2. Copy `kgi.py`

Copy `kgi.py` to the same direction as `run_clm.py`.

### 3. Run training

```shell
python run_clm.py configs/0_True.json
```

Here `0` is the seed, and `True` means KGI is turned on.