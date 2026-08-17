"""Debug HDF5 loader - check dataset validity"""
import sys
sys.path.insert(0, '/home/zhengkai/r2dreamer')
from hdf5_loader import load_hdf5_dataset

ds = load_hdf5_dataset(
    "/home/zhengkai/.stable_worldmodel/datasets/pusht_random_train.h5",
    num_steps=4, frameskip=5
)
print(f"Dataset len: {len(ds)}")
print(f"Valid starts: {len(ds._valid_starts)} / {ds._n}")
if len(ds) > 0:
    item = ds[0]
    for k, v in item.items():
        print(f"  {k}: {type(v).__name__}, shape={v.shape}")
else:
    print("EMPTY DATASET!")
    # Check why
    print(f"Total rows: {ds._n}")
    print(f"num_steps={ds.num_steps}, frameskip={ds.frameskip}")
    if ds._n > 0:
        print(f"First ep_idx range: {ds._file['episode_idx'][:10]}")
        print(f"First step_idx range: {ds._file['step_idx'][:10]}")
