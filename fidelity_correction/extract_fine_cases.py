"""
Extract from the fine-grid pool the cases listed in `test_indices` of the FTNO export,
so that the coarse (sr_coarse_20x20.h5) and fine (sr_fine_100x100.h5) SR files share
the same cases in the same order.
"""
import argparse

import h5py


def main(args):
    with h5py.File(args.coarse, "r") as f:
        idx = f["test_indices"][:]          # ascending case ids
    with h5py.File(args.fine_pool, "r") as f_src, h5py.File(args.out, "w") as f_dst:
        n_pool = f_src["sw_grid"].shape[0]
        for key in f_src.keys():
            src = f_src[key]
            data = src[idx] if (len(src.shape) > 0 and src.shape[0] == n_pool) else src[()]
            f_dst.create_dataset(key, data=data, compression="gzip", compression_opts=4)
    print(f"Saved {len(idx)} fine-grid cases -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", default="../data/sr_coarse_20x20.h5")
    p.add_argument("--fine_pool", default="../data/fine_pool_100x100.h5")
    p.add_argument("--out", default="../data/sr_fine_100x100.h5")
    main(p.parse_args())
