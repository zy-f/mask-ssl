# setup
- create mamba/conda env from `mask_env.yml`
- setup a2mim based on repo

# systems interface
basic idea:
for each dataset (in this case only imagenet), map dataset to standard format

for each model, need interface from:
- jank model weight loading -> standard weight loading format
- jank model forward function/output -> standard output format
- wrapper to restructure model to desired architecture
- wrapper around default dataset format if needed
- wrap a standard "system" around each pytorch model that takes in model config and deals with optimizers/schedulers/etc
    - each system should have its own step function
    - (aka steal from pytorch-lightning's way of doing things)

that should give us enough for a standard train/eval loop
- interfacing with imagenet-x just requires saving off outputs -- we can have a custom dataloader/eval loop for that