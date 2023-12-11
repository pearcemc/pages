## Mistral releases

How to understand the mistral releases? `mixtral-8x7b-32kseqlen` just dropped and it does not come with instructions. You can try it out at [fireworks.ai](https://app.fireworks.ai/models/fireworks/mixtral-8x7b). I am coming from a TensorFlow background and using DL libraries for non nlp tasks. So I am approaching current year LLMs in pytorch afresh.

To recap, this is what we get in the new release:
```bash
mixtral-8x7b-32kseqlen$ ls
RELEASE  consolidated.00.pth  params.json  tokenizer.model
```
Which is pretty much identically structured to the earlier release:
```bash
mistral-7B-v0.1$ ls
RELEASE  consolidated.00.pth  params.json  tokenizer.model
```

This minimalism is an opportunity too. Gives us a chance to look at the essentials without masking by magic or extraneous code. I'm trying to start understanding the *how* here. Not just rely on a script to run it. With LLMs being so hot right now, search is pretty noisy with seo spam and surface level videos. So, if we want to know more, we'll have to dig in👷.

(⏩ For a tl&dr, skip to the Conclusion below.)

### Initial peek

The question is, what do we do with these files?

* `RELEASE` - credits & checksum for validating file integrity.
* `consolidated.00.pth` - serialized py torch state
* `params.json` - high level description of model architecture
* `tokenizer.model` - some serialized tokenizer for the model

### `params.json`

Let's start with the 7b's `params.json` as it looks like a human readable config. What is it for and can we create a model with it? 

```bash
cat params.json
{
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "sliding_window": 4096,
    "vocab_size": 32000
}
```

OK, looks interesting, but it's not enough to get us a model. What are other people doing with it?

In the 7B reference implementation repo [`mistral-src`](https://github.com/mistralai/mistral-src) the contents are used to construct a `ModelArgs` object like this:
```python
@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int
    max_batch_size: int = 0

[...]
model_args = ModelArgs(**json.loads(params_json))
```
This llama library being in the self same repo. The `Transformer` class implements various layers and such. Presumably this is shared code from elsewhere given the speed with which it was released.

This goes on to parameterise a `class Transformer(nn.Module)` instance:
```python
model = Transformer(model_args).to(device=device, dtype=dtype)
```
However it's notable that not *all* the necessary args are in the model_args. In the `Transformer.from_folder` function of the reference implementation `max_batch_size` is set on the `model_args`. Without setting it, the code won't work. It makes sense they'd leave this out as batch size will be a key serving parameter and not part of the model per se. It does mean though that `params.json` is necessary but not sufficient for the model.
#### Over in `8x7b` land

Similarly in `mixtral-inference/mixtral/model.py` for the `8x7b` model we see:
```python
with open(folder / 'params.json', 'r') as f:
  model_args = ModelArgs(**json.loads(f.read()))
```
Where again `ModelArgs` is defined in the same repository and goes on to parameterise a `class Transformer(nn.Module)`.

Conclusion: we have some level of convention/standards. However, `params.json` is not passed to some 3rd party reference library which can give us a pytorch model. We have to supply an implementation ourselves. In the case of `7b` we have a reference implementation. For the `8x7b` we have to make educated guesses, or rely on someone else who has.
### `tokenizer.model`
Same story again. We need to implement the tokenizer, at least partially. Internally the `SentencePieceProcessor` from `sentencepiece` is used to do what looks like most of the heavy lifting. Indeed the docstring refers to this as a `SentencePiece` model file.

Usage (in `llama-mistral`) ends up like:
```python
t = Tokenizer('path/to/mixtral-8x7b-32kseqlen/tokenizer.model')

t.encode('hello', True, True)
# Out[19]: [1, 6312, 28709, 2]
t.encode('hello', False, False)
# Out[20]: [6312, 28709]

t.decode([6312])
# Out[23]: 'hell'
t.decode([1,2])
# Out[24]: ''
```
OK, mystery solved on rehydrating the tokenizer 💦

### `consolidated.00.pth`

This is the 87GB gorilla. What do we do with it? Back to grep.
```bash
grep -r "consolidated.00.pth"
mixtral-inference/mixtral/model.py:      
	loaded = torch.load(folder / 'consolidated.00.pth')
```
So we just do `torch.load`? Sounds like a recipe for an oom error on my poxy little 64GB ram machine.  

Let's try with the 7B model first. Also, `torch.load` offers an `mmap` flag which is off by default. This sounds promising as it suggests we won't deserialize all the tensors off disk.

```python
import torch
consolidated_path="/path/to/mistral-7B-v0.1/consolidated.00.pth"
loaded = torch.load(consolidated_path, mmap=True)

type(loaded)
# Out[4]: dict   

sample=list(loaded.keys())[::50]   # a subset of the keys...
sample
# Out[9]: 
# ['tok_embeddings.weight',
# 'layers.5.attention.wv.weight',
# 'layers.10.attention_norm.weight',
# 'layers.16.attention.wo.weight',
# 'layers.21.ffn_norm.weight',
# 'layers.27.feed_forward.w1.weight']
```
So, we have a bunch of named tensors but not how they fit together.
What can we tell about how they should fit though?

Why not just look at the reference implementation? Because we don't have one for `8x7b` yet, so we won't be able to rely on that there. Also noting the `mmap` works a charm, so may allow us to play with 8x7b even when we can't load it all into RAM.

```python
import collections
pieces = [s.split('.') for s in loaded.keys()]
collections.Counter(p[0] for p in pieces)
# Counter({'tok_embeddings': 1, 'norm': 1, 'output': 1, 'layers': 288})
```
Ok so this looks like some namespacing thing with most of the action in `layers`.
```python
collections.Counter(p[1] for p in pieces if p[0]=='layers')
# Counter({'0': 9, '1': 9, '2': 9, '3': 9, '4': 9, '5': 9, ...
# ...yup, there are 32 layers with 9 tensors each.
```
This lines up with what we learned from `params.json`: there should be 32 layers.

The third and forth parts of the names look like they belong together and say something interesting about the role of the tensor.

```python
shapes = [v.shape for v in iter(loaded.values())]
pieces = [k.split('.') for k in loaded.keys()]
roles=[".".join(p[2:]) for p in pieces]
sorted(collections.Counter(zip(roles, shapes)).items())
[(('', torch.Size([4096])), 1),
 (('', torch.Size([32000, 4096])), 2),
 (('attention.wk.weight', torch.Size([1024, 4096])), 32),
 (('attention.wo.weight', torch.Size([4096, 4096])), 32),
 (('attention.wq.weight', torch.Size([4096, 4096])), 32),
 (('attention.wv.weight', torch.Size([1024, 4096])), 32),
 (('attention_norm.weight', torch.Size([4096])), 32),
 (('feed_forward.w1.weight', torch.Size([14336, 4096])), 32),
 (('feed_forward.w2.weight', torch.Size([4096, 14336])), 32),
 (('feed_forward.w3.weight', torch.Size([14336, 4096])), 32),
 (('ffn_norm.weight', torch.Size([4096])), 32)]
```
So it looks like the internal blocks will be identical in kind and shape.
But the input / output layers have some different shapes. Recalling that 32k is the `vocab_size`.

```python
[(k,s) for k,s in zip(loaded.keys(), shapes) if 32000 in s]
# [('tok_embeddings.weight', (32000, 4096)), ('output.weight', (32000, 4096))]
```
So we expect `tok_embeddings` is the for input. Output should be self evident, although it kind of looks transposed to what I would expect.

### Detour: pytorch state file naming conventions

The naming is going to be important though. How does that work with pytorch state? Let's do a little experiment.

```python
import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200, bias=True)
        self.activation = torch.nn.ReLU()
        self.foo2 = torch.nn.Linear(200, 10, bias=False)
        self.softmax = torch.nn.Softmax()

tinymodel = TinyModel()
torch.save(tinymodel.state_dict(), 'example_state.pth')
loaded = torch.load('example_state.pth')
loaded.keys()
# odict_keys(['linear1.weight', 'linear1.bias', 'foo2.weight'])
```
OK, so the names of the weights aren't definitely linked to the type of the operation used to create them. Instead it looks like the instance of your op, e.g. `torch.nn.Linear` exposes a list of the weights associated with it. In this case `weight` and, optionally, `bias`. When an `torch.nn.Module` has `.state_dict()` called, it pulls those weight names, concatenating them as dot separated string to identify the attribute they're attached to.

This must happen recursively so that `'layers.5.attention.wv.weight'` means the weight of a `Linear` op (likely, unless many ops use `weight`) attached to the `wv` attribute of an `attention` block in the `5`th layer. 

The other implication of this is that if the authors of a model forget to rename the attributes (or were feeling mischievous) in their layers we would get the wrong impression of how to wire things up. Something to watch out for.

The other thing to check is whether this has any implications for state dict ordering. The appearance of `odict` is suggestive of the dict order being informative. 

```python
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.foo2 = torch.nn.Linear(50, 50, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(50, 50, bias=True)        
        self.softmax = torch.nn.Softmax()

tinymodel = TinyModel()
torch.save(tinymodel.state_dict(), 'example_state.pth')
loaded = torch.load('example_state.pth')
loaded.keys()
# odict_keys(['foo2.weight', 'linear1.weight', 'linear1.bias'])
```
So yes, the order of appearance of the ops the state dict has functional implications for how they were wired up.
### 🧩Back on topic
Armed with our theory about pytorch naming conventions it seems each of the 32 layers will have four ops, `attention`, `attention_norm`, `feed_fordward` and `ffn_norm` defined on the model in that order.

Let's compare to the reference implementation.

```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out
```

What do we learn? Well, the `forward` method is using the ops in a different order than they were attached to the block; e.g. `attention_norm` appears after `attention` in the state file keys but is used before it in `forward()`.

### The attention block

So, take this subset of the weights in the 7b state file.  Given their naming, these basically sound like they're going to be trainable weights from an attention block, 
```python
 (('attention.wk', (1024, 4096)), 32),
 (('attention.wo', (4096, 4096)), 32),
 (('attention.wq', (4096, 4096)), 32),
 (('attention.wv', (1024, 4096)), 32),
```

Now it's tempting to just pull out our nearest copy of [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) . However mistral have their own paper for the [7b model](https://arxiv.org/pdf/2310.06825.pdf). Indeed they say they use 'Sliding Window Attention' and/in which 'FlashAttention [11] and xFormers [18] yield a 2x speed improvement over a vanilla attention baseline.' 

So, with our 7b reference model loaded, we can pick out an attention layer
```
model.layers[1].attention
Attention(
  (wq): Linear(in_features=4096, out_features=4096, bias=False)
  (wk): Linear(in_features=4096, out_features=1024, bias=False)
  (wv): Linear(in_features=4096, out_features=1024, bias=False)
  (wo): Linear(in_features=4096, out_features=4096, bias=False)
)

list(model.layers[1].attention.state_dict().keys())
['wq.weight', 'wk.weight', 'wv.weight', 'wo.weight']
```
Noting the potential for getting confused about transposes. E.g. compare the in out here to the shape of the `attention.wk` tensor above.

When we get to `8x7b` this situation will be more acute due to the absence of reference implementation.

### Conclusion

TL&DR: first re `tokenizer.model` it looks like Mistral use `sentencepiece` which is used to load the serialized model. Next, we can deserialize the `pth` weights in pytorch 'easily' (subject to RAM). But the `.pth` only get the implied tensor names, not how they wire together. For the wiring we can use a reference implementation or educated guesses. In reference mode `params.json` allows us to instantiate a wired up model In educated guess mode (i.e. `8x7b` for now), `params.json` provides clues about wiring as the values link to tensor shapes.

So I think that's enough for now. I did a [little exercise](https://github.com/pearcemc/mistral-7b-reference) to enable running the `7b` model on CPU. Next would be hitting up the `8x7b` model to work without GPU.
