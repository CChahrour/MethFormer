import os

import anndata as ad
import crested
import numpy as np
import pandas as pd
import pyranges as pr
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from methformer import MethformerDataset

# setup logging
logger.remove()
logger.add("logs/methformer_data.log", format="{time} {level} {message}", level="INFO")


# setup logging
logger.remove()
logger.add("logs/methformer_data.log", format="{time} {level} {message}", level="INFO")


@logger.catch
def tile_regions(regions_df, tile_size=32):
    """
    Tiles the regions in the DataFrame into fixed-size bins.
    Each region is split into tiles of the specified size, and the resulting
    tiled regions are returned as a new PyRanges object.
    """
    logger.info(f"Tiling regions into {tile_size}bp bins...")
    tiled_rows = []
    for _, row in regions_df.iterrows():
        for i in range(0, 1024, tile_size):
            tiled_rows.append(
                {
                    "Chromosome": row["Chromosome"],
                    "Start": row["Start"] + i,
                    "End": row["Start"] + i + tile_size,
                    "RegionName": row["Name"],
                    "Bin": i // tile_size,
                    "Score": row.get("Score", "."),
                    "Strand": row.get("Strand", "."),
                }
            )
    return pr.PyRanges(pd.DataFrame(tiled_rows))


@logger.catch
def prepare_methylation_tensor(binned_regions, zarr_path, tile_size=32):
    """
    Prepares the methylation tensor from the Zarr dataset and binned regions.
    This function reads the methylation data from the Zarr file, assigns fractions,
    and reduces the data by the specified binned regions.
    The resulting DataFrame is saved as a Parquet file.
    """
    logger.info("Preparing methylation tensor from Zarr dataset...")
    from modality.contig_dataset import ContigDataset

    meth_ds = ContigDataset.from_zarrz(zarr_path)
    meth_ds.assign_fractions(
        numerators=["num_mc", "num_hmc", "num_modc"],
        denominator="num_total_c",
        min_coverage=5,
        inplace=True,
    )
    meth_panel_ds = meth_ds.reduce_byranges(
        ranges=binned_regions, ranges_are_1based=False, var=["frac_mc", "frac_hmc"]
    )
    df = meth_panel_ds.to_dataframe().reset_index()
    df = df[~df["sample_id"].str.contains("xeno")]
    df = df[
        [
            "RegionName",
            "Bin",
            "sample_id",
            "frac_mc_mean",
            "frac_hmc_mean",
            "contig",
            "start",
        ]
    ]
    df[["frac_mc_mean", "frac_hmc_mean"]] = (
        df[["frac_mc_mean", "frac_hmc_mean"]].fillna(-1).astype(np.float32)
    )
    df.to_parquet("data/meth_panel_binned.parquet", index=False)
    return df


@logger.catch
def build_tensor(df, n_bins=32):
    """
    Builds the methylation tensor from the DataFrame.
    The DataFrame is pivoted to create a tensor of shape (n_samples, n_regions, 2, n_bins),
    where the last dimension contains the mean fractions of methylated and hydroxymethylated cytosines.
    The function also filters out regions that are fully missing in both methylation types.
    """
    logger.info("Building methylation tensor from DataFrame...")
    mc_df = df.pivot(
        index=["RegionName", "Bin"], columns="sample_id", values="frac_mc_mean"
    )
    hmc_df = df.pivot(
        index=["RegionName", "Bin"], columns="sample_id", values="frac_hmc_mean"
    )

    # Drop fully missing regions
    region_mask = (mc_df != -1).groupby("RegionName").any().any(axis=1) | (
        hmc_df != -1
    ).groupby("RegionName").any().any(axis=1)
    valid_regions = region_mask[region_mask].index

    mc_df = mc_df.loc[mc_df.index.get_level_values("RegionName").isin(valid_regions)]
    hmc_df = hmc_df.loc[hmc_df.index.get_level_values("RegionName").isin(valid_regions)]

    mc_df = mc_df.sort_index(level=["RegionName", "Bin"])
    hmc_df = hmc_df.sort_index(level=["RegionName", "Bin"])

    mc_arr = mc_df.values
    hmc_arr = hmc_df.values
    meth_arr = np.stack([mc_arr, hmc_arr], axis=-1)
    meth_arr[np.isnan(meth_arr)] = -1.0

    n_total_rows = mc_df.shape[0]
    n_samples = mc_df.shape[1]
    n_regions = n_total_rows // n_bins
    assert n_total_rows == n_regions * n_bins, "Incomplete regions"

    meth_tensor = meth_arr.reshape(n_regions, n_bins, n_samples, 2).transpose(
        2, 0, 3, 1
    )
    return (
        meth_tensor,
        mc_df.columns.tolist(),
        mc_df.index.get_level_values("RegionName").unique(),
        valid_regions,
    )


@logger.catch
def get_labels(region_file, bigwigs_folder, sample_columns):
    """
    Imports bigwig files from the specified folder and regions file,
    and returns a DataFrame with the mean values for each sample.
    The DataFrame is renamed to match the expected sample columns.
    """
    logger.info("Importing bigwig files and extracting labels...")
    mll_adata = crested.import_bigwigs(
        bigwigs_folder=bigwigs_folder,
        regions_file=region_file,
        target="mean",
        chromsizes_file="/home/ubuntu/project/data/reference/hg38.chrom.sizes",
    )
    mll_df = mll_adata.T.to_df()
    colname_dict = {
        "CAT-RCHACV-1_MLL-N": "cell-RCHACV-1",
        "CAT-RCHACV-2_MLL-N": "cell-RCHACV-2",
        "CAT-RS411-1_MLL-N": "cell-RS411",
        "CAT-SEM-1_MLL-N": "cell-SEM-1",
        "CAT-SEM-2_MLL-N": "cell-SEM-2",
        "CAT-22620-1_MLL-N": "patient-22620",
        "CAT-23003-1_MLL-N": "patient-23003",
        "ChIP-26754-1_MLL-N": "patient-26754",
        "CAT-863388-1_MLL-N": "patient-863388",
        "CAT-9422-1_MLL-N": "patient-9422",
    }
    mll_df.rename(columns=colname_dict, inplace=True)
    return mll_df[sample_columns]


@logger.catch
def make_anndata(meth_df, valid_regions_list, mll_df, meth_tensor):
    """
    Creates an AnnData object from the methylation tensor and labels.
    The AnnData object contains the methylation tensor in the obsm field
    and the labels in the var field.
    """

    logger.info("Creating AnnData object...")
    obs_meta = pd.read_csv("data/meth_metadata.csv")
    obs_meta = obs_meta[~obs_meta["sample_id"].str.contains("xeno")]
    obs_meta["sample_id"] = "METH-" + obs_meta["sample_id"]
    obs_meta = obs_meta.set_index("sample_id")
    region_meta = meth_df.drop_duplicates("RegionName")[
        ["RegionName", "contig", "start"]
    ].set_index("RegionName")
    region_meta = region_meta.loc[valid_regions_list]
    mll_df_filtered = mll_df.loc[valid_regions_list]
    mll_df_filtered = mll_df_filtered.T
    logger.debug("MLL shape:", mll_df_filtered.shape)
    logger.debug("Region meta shape:", region_meta.shape)
    logger.debug("Obs meta shape:", obs_meta.shape)
    logger.debug("Meth tensor shape:", meth_tensor.shape)
    adata = ad.AnnData(X=mll_df_filtered.values, obs=obs_meta, var=region_meta)
    adata.obsm["methylation"] = meth_tensor
    adata.write("data/methformer_pretrain_MLL.h5ad")
    return adata


@logger.catch
def convert_dataset(adata, mask):
    """
    Converts the AnnData object to a MethformerDataset.
    The dataset is filtered based on the provided mask,
    which selects specific regions based on the contig.
    The methylation tensor is reshaped to match the expected input shape for Methformer.
    """
    logger.info("Converting AnnData to MethformerDataset...")
    tensor = adata.obsm["methylation"][:, mask.values, :, :]
    return MethformerDataset(tensor.reshape(-1, 32, 2))


@logger.catch
def to_numpy(torch_dataset):
    """
    Converts a PyTorch dataset to a NumPy dictionary.
    This function iterates over the dataset using a DataLoader,
    collects the inputs, labels, and attention masks,
    and concatenates them into NumPy arrays.
    """
    logger.info("Converting PyTorch dataset to NumPy arrays...")
    loader = DataLoader(torch_dataset, batch_size=1024, num_workers=8)
    inputs, labels, masks = [], [], []
    for batch in tqdm(loader, desc="Batching"):
        inputs.append(batch["inputs"].numpy())
        labels.append(batch["labels"].numpy())
        masks.append(batch["attention_mask"].numpy())
    return {
        "inputs": np.concatenate(inputs),
        "labels": np.concatenate(labels),
        "attention_mask": np.concatenate(masks),
    }


@logger.catch
def main():
    """
    Main function to prepare the pretraining dataset for Methformer.
    It reads the regions from a BED file, tiles them into fixed-size bins,
    prepares the methylation tensor from a Zarr dataset, and creates an AnnData object.
    The AnnData object is then split into training, evaluation, and test sets based on contig,
    and saved as a Hugging Face dataset.
    """
    logger.info("Starting Methformer pretraining data preparation...")
    regions = pr.read_bed(
        "/home/ubuntu/project/data/reference/methylome_1024bp.bed", as_df=True
    )
    binned = tile_regions(regions)
    meth_df_file = "data/meth_panel_binned.parquet"
    if os.path.exists(meth_df_file):
        meth_df = pd.read_parquet(meth_df_file)
    else:
        meth_df = prepare_methylation_tensor(
            binned, "data/meth_evoc.targeted.GRCh38Decoy.markdup.CG.zarrz"
        )

    meth_tensor, sample_ids, _, valid_regions = build_tensor(meth_df)

    mll_df = get_labels(
        "/home/ubuntu/project/data/reference/methylome_1024bp.bed",
        "data/mll_bw",
        sample_ids,
    )
    adata = make_anndata(meth_df, valid_regions, mll_df, meth_tensor)

    adata.write("data/methformer_pretrain_binned.h5ad")

    train_mask = ~(adata.var["contig"].isin(["chr8", "chr9"]))
    eval_mask = adata.var["contig"] == "chr8"
    test_mask = adata.var["contig"] == "chr9"

    train_set = convert_dataset(adata, train_mask)
    eval_set = convert_dataset(adata, eval_mask)
    test_set = convert_dataset(adata, test_mask)

    hf_dset = DatasetDict(
        {
            "train": Dataset.from_dict(to_numpy(train_set)),
            "validation": Dataset.from_dict(to_numpy(eval_set)),
            "test": Dataset.from_dict(to_numpy(test_set)),
        }
    )

    hf_dset.save_to_disk("data/methformer_pretrain_dataset")
    logger.info("✅ Pretraining data ready for Methformer!")


if __name__ == "__main__":
    main()
