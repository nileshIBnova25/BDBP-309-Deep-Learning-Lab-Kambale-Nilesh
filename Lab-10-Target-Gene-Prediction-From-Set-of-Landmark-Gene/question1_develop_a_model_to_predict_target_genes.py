#------------Imports-------------------#
from pathlib import Path
import gzip
import h5py
from cmapPy.pandasGEXpress.parse import parse
#--------------------------------------#

path = Path("/home/ibab/DL-Lab/Data/GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx.gz")
print(path.exists())
print("Compressed size (MB)",round(path.stat().st_size / (1024 * 1024),2))

gctx_path = path.with_suffix("")  # removes only the final .gz suffix

if not gctx_path.exists():
    with gzip.open(path, "rb") as src,open(gctx_path, "wb") as dst:
        while True:
            chunk = src.read(8*1024*1024)
            if not chunk:
                break
            dst.write(chunk)

with h5py.File(gctx_path, "r") as f:
    def show(name,obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(show)

gctoo = parse(str(gctx_path))
print("Matrix shape", gctoo.data_df.shape)
print("First row identifiers", gctoo.data_df.index[:5].tolist())
print("First column identifiers", gctoo.data_df.columns[:5].tolist())
print("Row metadata columns", gctoo.col_metadata_df.columns.tolist())
print("Column metadata columns", gctoo.row_metadata_df.columns.tolist())
print("Row metadata head",gctoo.row_metadata_df.head())
print("Column metadata head",gctoo.col_metadata_df.head())

