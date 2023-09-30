# Notes

## Attention backbones

ViT:
- patch embed DOES NOT normalize patch features
- use pre-norm
  ```
  x = x + MHA(norm(x))
  x = x + MLP(norm(x))
  ```
- Output features are normalized

Swin: 
- patch embed DOES normalize patch features
- use pre-norm
  ```
  x = x + MHA(norm(x))
  x = x + MLP(norm(x))
  ```
- Output features are NOT normalized

DETR:
- use post-norm
  ```
  x = norm(x + MHA(x, context))
  x = norm(x + MLP(x))
  ```
  
Attention is all:
- uses post-norm
  ```
  x = norm(x + MHA(x, context))
  x = norm(x + MLP(x))
  ```

My implementation:
- patch embed DOES NOT normalize patch features
- ViT use pre-norm
- ViT output features are normalized
- Obj attn use pre-norm on the queries only
  ```
  x = x + MHA(norm(x), ctx)
  x = x + MLP(norm(x))
  ```
  
## Questions

### Patch size

Patch size to split the input image into tokens for the backbone:

- (8, 8) is the default in the experiments
  - for a 128x128 CLEVR image it results in 16x16=256 tokens
  - batch_size=64 fits on one GPU
- (4, 4) might result in finer segmentation maps, but it produces 1024 tokens, batch size must be reduced to 16

### Positional encoding

Where to add positional encoding:

- at the input only
- before every self-attn transformer block
- before every self-attn transformer block and slot-attn iteration
- shared between backbone and slot-attn

### Contrastive loss negatives

Contrastive object loss between `aug0(x)` and `aug1(x)`. For a slot in `aug0(x)`
we select a single slot in `aug1(x)` to be the positive sample, what about the negatives?

1. use K-1 slots from `aug1(x)` as negatives
2. use slots from both `aug0(x)` and `aug1(x)` as negatives, in total `2(K-1)` negatives
3. use slots from all images in the batch as negatives, in total `2BK-2`

Option 2 seems to work best.

### Object query initialization

Fixed size, learned
- could learn slots for "background", "red metallic cube", etc
- if using architecture `backbone(-global_fn-global_proj)-obj_fn-obj_proj` collapses because all embeddings learn to
  ignore the img features
- instead, using architecture `backbone-obj_fn(-global_fn-global_proj)-obj_proj`
  works, but the weight on the object loss has to be quite low otherwise the global loss has no time to improve before
  the embeddings degenerate

Variable size, sampled
- test-time generalization to a different number of slots (ideally)
- however, it doesn't seem to learn well
- also, if sampling is made deterministic per-image during validation, the loss becomes artificially small

Based on clustering of backbone features
- if backbone features are clustered based on color/texture, how can the model distinguish between different instances
  of the same object type?

### In-depth experiments

Loss functions:
- Loss `ctr_all` is sensitive to batch size, due to false negatives in unrelated images.
  Should study the effect of batch size on `ctr_all`.
- Loss `ctr_img` is sensitive to number of slots due to false negatives in the same image. 
  Should study the effect of number of slots on `ctr_img`, but it's hard to separate it from number of objects in an img.
- Cosine similarity should not be affected by batch size and number of slots.

Characterize the collapsing behavior as they do in SimSiam. Do all slots become the same constant value?

Downstream VQA task: for models trained with both local and global representation, we know the object representation
performs well on our simple VQA task, but does the global representation work well too?

### Rollout

Rollout options:

- head reduction: mean or max?
- adjust for residual connections, i.e. +0.5?
- the best combination seems: reduction=mean, adjust=True

### Other papers

Embedding dimension:

- slot attention paper for CLEVR uses embed_dim=64 and hidden_dim=128

Batch size:

- slot attention paper uses 64

### Pretrained backbones

architecture | backbone | batch
-|-|-
global only| tiny 16| 16

Max batch size for `vit_tiny_patch16_224_in21k` global-only is 16:
```./train.py model=bb_global model/backbone=vit_base_patch8_224_in21k losses/l_objects=none training.batch_size=16```

Max batch size for `vit_tiny_patch16_224_in21k` global and objects is 64:
```./train.py model/backbone=vit_base_patch8_224_in21k training.batch_size=64```

## Runs

Headless notebook run

```bash
export CUDA_VISIBLE_DEVICES=0
INPUT_NB='train.ipynb'
OUTPUT_NB="$(date --iso-8601=seconds).train.ipynb"
jupyter nbconvert --to notebook --execute --allow-errors \
  "${INPUT_NB}" --output="${OUTPUT_NB}" && chmod a-w "${OUTPUT_NB}" 
```

```bash
export CUDA_VISIBLE_DEVICES=0

for DECAY in cosine exponential; do
for OBJ_NHEADS in 8 4; do
for OBJ_POS_EMBED in backbone learned; do
for OBJ_REUSE in 'false' 'true'; do
for MLOSS in '' '+losses=more_global' '+losses=more_objects'; do
for OBJ_NTEMPLATES in 11 24; do
for OBJ_NLAYERS in 3 1; do
./train.py \
  lr_scheduler.multiplier=1 \
  lr_scheduler.decay.name=${DECAY} \
  ${MLOSS} \
  model/backbone='vit_tiny_patch16_224_in21k' \
  model/obj_queries='learned' \
  model.obj_queries.num_objects=${OBJ_NTEMPLATES} \
  model/obj_fn='cross_attention' \
  model.obj_fn.pos_embed=${OBJ_POS_EMBED} \
  model.obj_fn.num_heads=${OBJ_NHEADS} \
  model.obj_fn.num_layers=${OBJ_NLAYERS} \
  model.obj_fn.reuse_layers=${OBJ_REUSE}
done
done
done
done
done
done
done

```

Bunch of sweeps
```bash
export CUDA_VISIBLE_DEVICES=2
for NLAYERS in 4 2 6; do
for NHEADS in 8 4 2; do
./train.py \
  lr_scheduler=linear1_cosine4_x5 \
  model=bb_global \
  losses/l_objects=none \
  logging.notes='Global only - 4x4 patch - sweep num_layers num_heads' \
  logging.group='global.p44.d64.sweep_layers_heads' \
  training.batch_size=8 \
  model.backbone.patch_size='[4,4]' \
  model.backbone.embed_dim=64 \
  model.backbone.{proj_drop,attn_drop}=0.2 \
  model.backbone.num_layers=$NLAYERS \
  model.backbone.num_heads=$NHEADS
done
done

export CUDA_VISIBLE_DEVICES=2
for NLAYERS in 4 2; do
for NITERS in 4 2 1; do
./train.py \
  +losses=more_global \
  lr_scheduler=linear1_cosine4_x5 \
  model=bb_obj_global \
  model/obj_fn=cross_attention \
  logging.notes='Cross attn - 8x8 patch - 8 backbone heads - sweep num_layers num_layers' \
  logging.group='p88.h8.d64.cross.sweep_layers_layers' \
  model.obj_queries.name=learned \
  model.backbone.embed_dim=64 \
  model.backbone.num_heads=8 \
  model.backbone.{proj_drop,attn_drop}=0.2 \
  model.backbone.num_layers=$NLAYERS \
  model.obj_fn.num_layers=$NITERS
done
done

export CUDA_VISIBLE_DEVICES=1
for NLAYERS in 4 2; do
for NITERS in 4 2 1; do
./train.py \
  +losses=more_global \
  lr_scheduler=linear1_cosine4 \
  model=bb_obj_global \
  model/obj_fn=cross_attention \
  logging.notes='Cross attn - 4x4 patch - 8 backbone heads - sweep num_layers num_layers' \
  logging.group='p44.h8.d64.cross.sweep_layers_layers' \
  training.batch_size=8 \
  model.obj_queries.name=learned \
  model.backbone.patch_size='[4,4]' \
  model.backbone.embed_dim=64 \
  model.backbone.num_heads=8 \
  model.backbone.{proj_drop,attn_drop}=0.2 \
  model.backbone.num_layers=$NLAYERS \
  model.obj_fn.num_layers=$NITERS
done
done

export CUDA_VISIBLE_DEVICES=3
for NLAYERS in 4 2; do
for NITERS in 4 2 1; do
./train.py \
  +losses=more_global \
  lr_scheduler=linear1_cosine4_x5 \
  model=bb_obj_global \
  model/obj_fn=slot_attention \
  logging.notes='Slot attn - 4x4 patch - 8 backbone heads - sweep num_layers num_iters' \
  logging.group='p44.h8.d64.slot.sweep_layers_iters' \
  training.batch_size=8 \
  model.obj_queries.name=learned \
  model.backbone.patch_size='[4,4]' \
  model.backbone.embed_dim=64 \
  model.backbone.num_heads=8 \
  model.backbone.{proj_drop,attn_drop}=0.2 \
  model.backbone.num_layers=$NLAYERS \
  model.obj_fn.num_iters=$NITERS
done
done

```

Drop-ins:
```bash
./train.py --multirun hydra/launcher=submitit_local

model.backbone.embed_dim=64 \
model.backbone.num_layers=4 \
model.backbone.num_heads=8 \

model.obj_queries.name=learned \
model.obj_queries.name=kmeans_euclidean \
  
model.backbone.{proj_drop,attn_drop}=0.2 \
model.{global_fn,global_proj,obj_proj}.dropout=0.2 \

+data=overfit logging.tags='[overfit]'
```

Finished running in slurm:
```bash
# Sampling obj queries (48)
./train.py --multirun hydra/launcher=submitit_slurm +slurm=slurm \
  +losses=more_objects,more_global \
  model=bb_obj_global \
  model/obj_queries=sample \
  model.obj_queries.num_components=12,24,48 \
  model.backbone.embed_dim=64,128,256 \
  logging.group='slurm_sweep' \
  lr_scheduler=linear1_cosine4_x5 \
  lr_scheduler.decay.end_lr=0.0003 \
  optimizer.start_lr=0.0007 \
  optimizer.weight_decay=0.0001 \
  model.backbone.num_heads=4,8 \
  model.backbone.num_layers=2,4,6 \
  model.obj_fn.num_iters=1,2,4
```

Now running in slurm:
```bash
# Global only
./train.py --multirun hydra/launcher=submitit_slurm +slurm=slurm \
  +losses/l_objects=none \
  model=bb_global \
  model/obj_queries=learned \
  model.backbone.embed_dim=64,128,256 \
  lr_scheduler=linear1_cosine4_x2 \
  model.backbone.num_heads=4 training.batch_size=32 \
  model.backbone.num_layers=2,4 \
  model.obj_fn.num_iters=1,2 \
  model.backbone.patch_size='[8,8]'

./train.py --multirun hydra/launcher=submitit_slurm +slurm=slurm \
  losses/l_objects=sim_all \
  +losses=more_global \
  data.crop_scale='[0.3,1.0]' \
  model=bb_obj_global \
  \
  model/obj_queries=learned \
  model/obj_queries=kmeans_euclidean \
  model/obj_queries=sample model.obj_queries.num_components=1,16,32 \
  \
  model/obj_fn=cross_attention model.obj_fn.num_layers=2 \
  model/obj_fn=slot_attention model.obj_fn.num_iters=1 \
  \
  model.backbone.num_layers=2 \
  model.backbone.embed_dim=256,128 \
  model.backbone.patch_size='[4,4],[8,8]' \
  lr_scheduler=linear1_cosine4_x2 \
  \
  model.backbone.num_heads=4 training.batch_size=16 \
  model.backbone.num_heads=8 training.batch_size=8 \
  
./train.py --multirun hydra/launcher=submitit_slurm +slurm=slurm \
  losses/l_objects=ctr_all \
  +losses=more_global \
  data.crop_scale='[0.3,1.0]' \
  model=bb_obj_global \
  model/obj_queries=learned \
  model/obj_fn=slot_attention model.obj_fn.num_iters=1,2 \
  model.backbone.num_layers=2,4 \
  model.backbone.embed_dim=256,128,64 \
  model.backbone.patch_size='[4,4]' \
  lr_scheduler=linear1_cosine4_x2 \
  model.backbone.num_heads=4 training.batch_size=16

for PSIZE in '[8,8]' '[4,4]'; do
for CRSCALE in '[0.1,0.3]' '[0.1,0.5]' '[0.1,0.8]' '[0.3,1.0]'; do
for DIM in 128 64; do
./train.py \
  losses/l_objects=sim_img \
  +losses=more_global \
  data.crop_scale="$CRSCALE" \
  model=bb_obj_global \
  \
  model/obj_queries=learned \
  \
  model/obj_fn=slot_attention model.obj_fn.num_iters=1 \
  \
  model.backbone.num_layers=2 \
  model.backbone.embed_dim=$DIM \
  model.backbone.patch_size="$PSIZE" \
  lr_scheduler=linear1_cosine4_x2 \
  \
  model.backbone.num_heads=4 training.batch_size=16
done
done
done


for PSIZE in '[8,8]' '[4,4]'; do
for DIM in 128 256; do
./train.py \
  losses/l_objects=sim_img \
  +losses=more_global \
  model=bb_obj_global \
  \
  model/obj_queries=learned \
  \
  model/obj_fn=cross_attention model.obj_fn.num_layers=1 \
  \
  model.backbone.num_layers=2 \
  model.backbone.embed_dim=$DIM \
  model.backbone.patch_size="$PSIZE" \
  lr_scheduler=linear1_cosine4_x2 \
  \
  model.backbone.num_heads=4 training.batch_size=16
done
done

for PSIZE in '[8,8]' '[4,4]'; do
for DIM in 128 256; do
./train.py \
  losses/l_objects=sim_img \
  +losses=more_global \
  model=bb_obj_global \
  \
  model/obj_queries=learned \
  \
  model/obj_fn=slot_attention model.obj_fn.num_iters=1 \
  \
  model.backbone.num_layers=2 \
  model.backbone.embed_dim=$DIM \
  model.backbone.patch_size="$PSIZE" \
  lr_scheduler=linear1_cosine4_x2 \
  \
  model.backbone.num_heads=4 training.batch_size=16
done
done
```

## TODOs

Next:

- take data augm pipeline from SimCLR v2 (?)
- linear evaluation on held-out test set:
  - count objects
  - object properties
  
Sweep everything
```
model=bb_obj_global,bb_global_obj
model/obj_queries=learned,sample,kmeans_euclidean model.obj_queries.num_components=1,16,32
model/obj_fn=slot_attention,cross_attention
model.backbone.embed_dim=64,128,256
model.backbone.num_layers=2,4,6
model.backbone.num_heads=4,8,16
model.backbone.patch_size='[4,4]' training.batch_size=16
model.backbone.patch_size='[8,8]' training.batch_size=64
lr_scheduler=linear1_cosine4_x2
+losses=more_global,more_objects,zero_objects
model.backbone.{proj_drop,attn_drop}=0.2
model.{global_fn,global_proj,obj_proj}.dropout=0.2
```

## Rename wandb group
```python
import wandb
api = wandb.Api()
runz = [r for r in api.runs('baldassarrefe/iclr-osc-22') if r.group=="slurm_sweep_learned"]
print(len(runz))
for r in runz:
    r.group = "p88.slot.sweep_loss_dim_layers_heads_iters"
    r.update()
```
