import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# =============================
# CONFIG
# =============================
BASE_DIR = Path(r"C:\projects\resneur\projectcrops")

CDL_FILES = [
    BASE_DIR / "CDL_North_2021.tif",
    BASE_DIR / "CDL_Sentinel_Area_2021.tif"
]

SENTINEL_TILES = [
    # North
    BASE_DIR / "MCTNet_400km2_Arkansas_North-0000000000-0000000000.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_North-0000000000-0000001792.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_North-0000001792-0000000000.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_North-0000001792-0000001792(1).tif",
    # South
    BASE_DIR / "MCTNet_400km2_Arkansas_South-0000000000-0000000000-001.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_South-0000000000-0000001792.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_South-0000001792-0000000000.tif",
    BASE_DIR / "MCTNet_400km2_Arkansas_South-0000001792-0000001792.tif"
]

# Actual Day-of-Year for each tile
SENTINEL_DOY = [10, 60, 110, 160, 210, 260, 310, 360]

NUM_SAMPLE = 2000

# =============================
# CDL → Crop classes
# =============================
def map_crop(code):
    if code == 1:
        return "Corn"
    elif code == 3:
        return "Soybeans"
    elif code == 5:
        return "Cotton"
    elif code == 2 or code == 4:
        return "Rice"
    else:
        return "Others"

# =============================
# LOAD & COMBINE CDL
# =============================
def load_cdl():
    cdls = []
    max_width = 0
    for f in CDL_FILES:
        with rasterio.open(f) as src:
            arr = src.read(1)
            cdls.append(arr)
            max_width = max(max_width, arr.shape[1])
            print(f"📂 Loaded CDL: {f} → shape {arr.shape}")
    
    padded_cdls = []
    for arr in cdls:
        h, w = arr.shape
        if w < max_width:
            arr = np.pad(arr, ((0,0),(0,max_width-w)), constant_values=0)
        padded_cdls.append(arr)
    
    combined_cdl = np.vstack(padded_cdls)
    print(f"✅ Combined CDL shape: {combined_cdl.shape}")
    return combined_cdl

# =============================
# SAMPLE PIXELS
# =============================
def sample_pixels(cdl, num_samples=NUM_SAMPLE):
    rows, cols = np.where(cdl > 0)
    idx = np.random.choice(len(rows), num_samples, replace=False)
    return rows[idx], cols[idx]

# =============================
# EXTRACT NDVI TIME SERIES
# =============================
def extract_ndvi(rows, cols):
    ndvi_all = []
    valid_rows, valid_cols = rows.copy(), cols.copy()
    
    for tile_path in SENTINEL_TILES:
        print(f"\n📂 Processing {tile_path.name}")
        with rasterio.open(tile_path) as src:
            data = src.read(1)
            h, w = data.shape
            mask = (valid_rows < h) & (valid_cols < w)
            r = valid_rows[mask]
            c = valid_cols[mask]
            values = np.full((len(valid_rows),), np.nan)
            values[mask] = data[r, c]
            ndvi_all.append(values)
    
    ndvi_stack = np.column_stack(ndvi_all)
    print(f"✅ NDVI time series shape: {ndvi_stack.shape}")
    return ndvi_stack

# =============================
# MAP CDL → CROPS
# =============================
def get_crop_names(cdl, rows, cols):
    codes = cdl[rows, cols]
    return np.array([map_crop(c) for c in codes])

# =============================
# PLOT DISTRIBUTION
# =============================
def plot_distribution(crops):
    unique, counts = np.unique(crops, return_counts=True)
    total = counts.sum()
    
    plt.figure(figsize=(8,5))
    plt.bar(unique, counts, color="skyblue")
    plt.title("Crop Distribution (Arkansas)")
    plt.ylabel("Pixel Count")
    plt.xticks(rotation=30)
    plt.show()
    
    print("Crop Distribution (% of pixels):")
    for u, c in zip(unique, counts):
        print(f"{u}: {c/total*100:.2f}%")

# =============================
# PLOT NDVI TIME SERIES WITH DOY
# =============================
def plot_ndvi(ndvi_ts, crops, doy):
    colors = {"Corn":"gold", "Rice":"green", "Soybeans":"red","Cotton":"orange","Others":"purple"}
    
    plt.figure(figsize=(12,7))
    for crop in ["Corn","Rice","Soybeans","Cotton","Others"]:
        idx = np.where(crops==crop)[0]
        if len(idx)==0: continue
        mean_ndvi = np.nanmean(ndvi_ts[idx], axis=0)
        plt.plot(doy, mean_ndvi, color=colors[crop], label=crop, linewidth=2, marker='o')
    
    plt.xlabel("Day of Year")
    plt.ylabel("Mean NDVI")
    plt.title("NDVI Time-Series Profiles of Crops in Arkansas")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# =============================
# PLOT RAW SAMPLES + INTERPOLATION WITH DOY
# =============================
def plot_raw_interpolation(ndvi_ts, crops, doy):
    plt.figure(figsize=(12,7))
    n_samples = min(20, ndvi_ts.shape[0])
    
    for i in range(n_samples):
        plt.plot(doy, ndvi_ts[i], color="grey", alpha=0.5, marker='o')
        # linear interpolation
        valid_mask = ~np.isnan(ndvi_ts[i])
        f = interp1d(np.array(doy)[valid_mask], ndvi_ts[i][valid_mask], kind='linear', fill_value="extrapolate")
        plt.plot(doy, f(doy), color="red", alpha=0.7)
    
    plt.title("Raw NDVI Samples with Linear Interpolation")
    plt.xlabel("Day of Year")
    plt.ylabel("NDVI")
    plt.show()

# =============================
# SPATIAL CDL MAP
# =============================
def plot_cdl_map(cdl):
    plt.figure(figsize=(10,10))
    plt.imshow(cdl, cmap='tab20')
    plt.colorbar(label="CDL Crop Class")
    plt.title("Spatial CDL Map of Arkansas")
    plt.show()

# =============================
# MAIN
# =============================
def main():
    print("\n🚀 Arkansas Combined Pipeline Start")
    
    cdl = load_cdl()
    plot_cdl_map(cdl)
    
    rows, cols = sample_pixels(cdl)
    print(f"🎯 Sampled {len(rows)} pixels")
    
    crops = get_crop_names(cdl, rows, cols)
    
    ndvi_ts = extract_ndvi(rows, cols)
    
    plot_distribution(crops)
    plot_ndvi(ndvi_ts, crops, SENTINEL_DOY)
    plot_raw_interpolation(ndvi_ts, crops, SENTINEL_DOY)
    
    print("\n🎉 DONE — Arkansas NDVI & Crop Analysis Complete!")

if __name__=="__main__":
    main()