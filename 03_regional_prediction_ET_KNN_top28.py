# Supplementary Material 1 - Part 3
# Regional forest-type prediction using the ET + KNN hierarchical model

import os
import gc
import json
import math
import time
import warnings
from glob import glob

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds, from_origin, xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

MODEL_BUNDLE_PATH = r"<PATH_TO_MODEL_BUNDLE>/ET_KNN_top28_models.joblib"

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
TILE_ZONES = ["48N", "49N"]

AEF_TILE_ROOT_TEMPLATE = r"<PATH_TO_AEF_TILE_ROOT>/{year}/download_tiles"
INITIAL_OUTPUT_TEMPLATE = r"<PATH_TO_INITIAL_OUTPUT_ROOT>/{year}/initial_results"
FINAL_OUTPUT_DIR = r"<PATH_TO_FINAL_OUTPUT_ROOT>"

BLOCK_SIZE = 512
MAX_FILES_PER_ZONE = None

RESUME_MODE = True
FORCE_REBUILD_TILE = False
FORCE_REBUILD_INTERMEDIATE = False
FORCE_REBUILD_MOSAIC = False

RAW_NODATA = -128
OUT_NODATA = 255
TARGET_CRS = "EPSG:4326"
MERGE_METHOD = "first"


# =============================================================================
# Model loading
# =============================================================================

def load_et_knn_bundle(model_path):
    bundle = joblib.load(model_path)

    required_keys = [
        "step1_model",
        "step2_model",
        "step2_scaler",
        "step1_features",
        "step2_features",
    ]
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise KeyError(f"Missing keys in model bundle: {missing}")

    return bundle


# =============================================================================
# Raster prediction
# =============================================================================

def dequantize_alphaearth(values, nodata=RAW_NODATA):
    values = values.astype(np.float32)
    nodata_mask = values == nodata
    out = ((values / 127.5) ** 2) * np.sign(values)
    out[nodata_mask] = np.nan
    return out


def truncate_decimal(value, decimals=2):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor if value >= 0 else math.ceil(value * factor) / factor


def get_lonlat_bounds(src):
    corners_proj = [
        xy(src.transform, 0, 0, offset="center"),
        xy(src.transform, 0, src.width - 1, offset="center"),
        xy(src.transform, src.height - 1, 0, offset="center"),
        xy(src.transform, src.height - 1, src.width - 1, offset="center"),
    ]

    transformer = Transformer.from_crs(src.crs, TARGET_CRS, always_xy=True)
    corners_ll = [transformer.transform(x, y) for x, y in corners_proj]

    lons = [lon for lon, _ in corners_ll]
    lats = [lat for _, lat in corners_ll]

    return {
        "west": min(lons),
        "east": max(lons),
        "south": min(lats),
        "north": max(lats),
    }


def make_output_name(src):
    bounds = get_lonlat_bounds(src)
    south = truncate_decimal(bounds["south"], 2)
    west = truncate_decimal(bounds["west"], 2)
    return f"{south:.2f}_{west:.2f}.tif", bounds


def get_northup_transform(src_transform, src_height):
    if src_transform.e < 0:
        return src_transform

    return Affine(
        src_transform.a,
        src_transform.b,
        src_transform.c,
        src_transform.d,
        -abs(src_transform.e),
        src_transform.f + src_transform.e * src_height,
    )


def get_band_mapping(src):
    if src.descriptions and src.descriptions[0] is not None:
        return {name: i + 1 for i, name in enumerate(src.descriptions)}

    return {f"A{i:02d}": i + 1 for i in range(src.count)}


def update_progress(progress_df, record, key_columns):
    if progress_df is None or len(progress_df) == 0:
        return pd.DataFrame([record])

    mask = np.ones(len(progress_df), dtype=bool)
    for col in key_columns:
        mask &= progress_df[col].fillna("").astype(str).values == str(record.get(col, ""))

    if mask.any():
        for key, value in record.items():
            progress_df.loc[mask, key] = value
    else:
        progress_df = pd.concat([progress_df, pd.DataFrame([record])], ignore_index=True)

    return progress_df


def load_progress(progress_path, columns):
    if os.path.exists(progress_path):
        try:
            return pd.read_csv(progress_path, encoding="utf-8-sig")
        except Exception:
            pass

    return pd.DataFrame(columns=columns)


def save_dataframe(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def classify_one_tile(
    tif_path,
    year,
    zone,
    output_dir,
    bundle,
    block_size=BLOCK_SIZE,
):
    step1_model = bundle["step1_model"]
    step2_model = bundle["step2_model"]
    step2_scaler = bundle["step2_scaler"]
    step1_features = bundle["step1_features"]
    step2_features = bundle["step2_features"]

    with rasterio.open(tif_path) as src:
        base_name, bounds = make_output_name(src)
        final_output = os.path.join(output_dir, base_name)
        temp_output = final_output + ".part.tif"

        if os.path.exists(final_output):
            if FORCE_REBUILD_TILE:
                os.remove(final_output)
            elif RESUME_MODE:
                return {
                    "year": year,
                    "zone": zone,
                    "source_tif": tif_path,
                    "output_tif": final_output,
                    "west": bounds["west"],
                    "east": bounds["east"],
                    "south": bounds["south"],
                    "north": bounds["north"],
                    "status": "skipped_exists",
                    "message": "",
                    "time_seconds": 0.0,
                }

        if os.path.exists(temp_output):
            os.remove(temp_output)

        band_mapping = get_band_mapping(src)
        needed_features = sorted(set(step1_features + step2_features))

        missing_features = [feature for feature in needed_features if feature not in band_mapping]
        if missing_features:
            raise KeyError(f"Missing AEF bands in {tif_path}: {missing_features}")

        needed_indexes = [band_mapping[feature] for feature in needed_features]
        feature_position = {feature: i for i, feature in enumerate(needed_features)}
        step1_positions = [feature_position[feature] for feature in step1_features]
        step2_positions = [feature_position[feature] for feature in step2_features]

        northup_transform = get_northup_transform(src.transform, src.height)
        need_vertical_flip = src.transform.e > 0

        output_profile = src.profile.copy()
        output_profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=OUT_NODATA,
            compress="lzw",
            transform=northup_transform,
        )

        t0 = time.perf_counter()

        with rasterio.open(temp_output, "w", **output_profile) as dst:
            n_rows = src.height
            n_cols = src.width

            for row_off in range(0, n_rows, block_size):
                for col_off in range(0, n_cols, block_size):
                    height = min(block_size, n_rows - row_off)
                    width = min(block_size, n_cols - col_off)
                    src_window = Window(col_off, row_off, width, height)

                    raw_block = src.read(indexes=needed_indexes, window=src_window)
                    block = dequantize_alphaearth(raw_block, nodata=RAW_NODATA)

                    step1_block = block[step1_positions, :, :]
                    h, w = step1_block.shape[1], step1_block.shape[2]
                    step1_flat = np.transpose(step1_block, (1, 2, 0)).reshape(
                        -1, len(step1_features)
                    )
                    valid_step1 = np.all(np.isfinite(step1_flat), axis=1)

                    output_flat = np.full(h * w, OUT_NODATA, dtype=np.uint8)

                    if valid_step1.any():
                        step1_pred = np.zeros(h * w, dtype=np.uint8)
                        step1_pred[valid_step1] = step1_model.predict(
                            step1_flat[valid_step1]
                        ).astype(np.uint8)

                        other_mask = valid_step1 & (step1_pred == 0)
                        forest_mask = valid_step1 & (step1_pred == 1)

                        output_flat[other_mask] = 5

                        if forest_mask.any():
                            step2_block = block[step2_positions, :, :]
                            step2_flat = np.transpose(step2_block, (1, 2, 0)).reshape(
                                -1, len(step2_features)
                            )
                            valid_step2 = np.all(np.isfinite(step2_flat), axis=1)
                            final_forest_mask = forest_mask & valid_step2

                            if final_forest_mask.any():
                                X_step2 = step2_flat[final_forest_mask]
                                X_step2 = step2_scaler.transform(X_step2)
                                step2_pred = step2_model.predict(X_step2).astype(np.uint8)
                                output_flat[final_forest_mask] = step2_pred

                    output_block = output_flat.reshape(h, w)

                    if need_vertical_flip:
                        output_block = np.flipud(output_block)
                        dst_window = Window(col_off, n_rows - (row_off + h), w, h)
                    else:
                        dst_window = src_window

                    dst.write(output_block, 1, window=dst_window)

                    del raw_block, block, step1_block, step1_flat, valid_step1
                    del output_flat, output_block
                    gc.collect()

        os.replace(temp_output, final_output)
        elapsed = time.perf_counter() - t0

        return {
            "year": year,
            "zone": zone,
            "source_tif": tif_path,
            "output_tif": final_output,
            "west": bounds["west"],
            "east": bounds["east"],
            "south": bounds["south"],
            "north": bounds["north"],
            "status": "success",
            "message": "",
            "time_seconds": elapsed,
        }


def collect_year_tasks(year):
    tasks = []
    root = AEF_TILE_ROOT_TEMPLATE.format(year=year)

    for zone in TILE_ZONES:
        zone_dir = os.path.join(root, zone)
        tif_list = sorted(glob(os.path.join(zone_dir, "*.tif"))) + sorted(
            glob(os.path.join(zone_dir, "*.tiff"))
        )

        if MAX_FILES_PER_ZONE is not None:
            tif_list = tif_list[:MAX_FILES_PER_ZONE]

        for tif_path in tif_list:
            tasks.append((year, zone, tif_path))

    return tasks


def run_tile_prediction(bundle):
    all_summary = []

    for year in YEARS:
        output_dir = INITIAL_OUTPUT_TEMPLATE.format(year=year)
        os.makedirs(output_dir, exist_ok=True)

        progress_path = os.path.join(output_dir, f"ET_KNN_top28_tile_prediction_progress_{year}.csv")
        summary_path = os.path.join(output_dir, f"ET_KNN_top28_tile_prediction_summary_{year}.json")

        progress_df = load_progress(
            progress_path,
            [
                "year",
                "zone",
                "source_tif",
                "output_tif",
                "west",
                "east",
                "south",
                "north",
                "status",
                "message",
                "time_seconds",
            ],
        )

        tasks = collect_year_tasks(year)
        t_start = time.perf_counter()

        for idx, (task_year, zone, tif_path) in enumerate(tasks, 1):
            print(f"[{task_year}] {idx}/{len(tasks)} {zone}: {os.path.basename(tif_path)}")

            try:
                record = classify_one_tile(
                    tif_path=tif_path,
                    year=task_year,
                    zone=zone,
                    output_dir=output_dir,
                    bundle=bundle,
                    block_size=BLOCK_SIZE,
                )
            except Exception as exc:
                record = {
                    "year": task_year,
                    "zone": zone,
                    "source_tif": tif_path,
                    "output_tif": "",
                    "west": np.nan,
                    "east": np.nan,
                    "south": np.nan,
                    "north": np.nan,
                    "status": "failed",
                    "message": str(exc),
                    "time_seconds": np.nan,
                }

            progress_df = update_progress(
                progress_df,
                record,
                key_columns=["year", "zone", "source_tif"],
            )
            save_dataframe(progress_df, progress_path)

        elapsed = time.perf_counter() - t_start

        summary = {
            "year": year,
            "tile_count": len(tasks),
            "success_count": int((progress_df["status"] == "success").sum()),
            "skipped_count": int((progress_df["status"] == "skipped_exists").sum()),
            "failed_count": int((progress_df["status"] == "failed").sum()),
            "output_dir": output_dir,
            "time_seconds": elapsed,
        }

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        all_summary.append(summary)

    return pd.DataFrame(all_summary)


# =============================================================================
# Annual mosaic
# =============================================================================

def find_classified_tiles(input_dir):
    tif_list = sorted(glob(os.path.join(input_dir, "*.tif"))) + sorted(
        glob(os.path.join(input_dir, "*.tiff"))
    )
    excluded = ["mosaic", "_fixed4326", "_northup", ".part"]

    return [
        path for path in sorted(set(tif_list))
        if not any(token in os.path.basename(path).lower() for token in excluded)
    ]


def make_intermediate_path(year, tif_path, intermediate_root):
    year_dir = os.path.join(intermediate_root, str(year))
    os.makedirs(year_dir, exist_ok=True)
    return os.path.join(year_dir, os.path.basename(tif_path))


def fix_orientation_and_reproject(src_path, dst_path):
    temp_dst = dst_path + ".part.tif"

    if os.path.exists(temp_dst):
        os.remove(temp_dst)

    with rasterio.open(src_path) as src:
        arr = src.read()
        src_crs = src.crs
        src_profile = src.profile.copy()

        if src_crs is None:
            raise ValueError(f"Missing CRS: {src_path}")

        if src.transform.e > 0:
            arr_fixed = arr[:, ::-1, :]

            left = min(src.bounds.left, src.bounds.right)
            right = max(src.bounds.left, src.bounds.right)
            bottom = min(src.bounds.bottom, src.bounds.top)
            top = max(src.bounds.bottom, src.bounds.top)

            xres = abs(src.transform.a)
            yres = abs(src.transform.e)
            transform_fixed = from_origin(left, top, xres, yres)
        else:
            arr_fixed = arr
            transform_fixed = src.transform

        height_fixed = arr_fixed.shape[1]
        width_fixed = arr_fixed.shape[2]
        left, bottom, right, top = array_bounds(height_fixed, width_fixed, transform_fixed)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            TARGET_CRS,
            width_fixed,
            height_fixed,
            left,
            bottom,
            right,
            top,
        )

        dst_profile = src_profile.copy()
        dst_profile.update(
            crs=TARGET_CRS,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            nodata=OUT_NODATA,
            compress="lzw",
        )

        with rasterio.open(temp_dst, "w", **dst_profile) as dst:
            for band_id in range(arr_fixed.shape[0]):
                reproject(
                    source=arr_fixed[band_id],
                    destination=rasterio.band(dst, band_id + 1),
                    src_transform=transform_fixed,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=TARGET_CRS,
                    src_nodata=src.nodata if src.nodata is not None else OUT_NODATA,
                    dst_nodata=OUT_NODATA,
                    resampling=Resampling.nearest,
                )

    os.replace(temp_dst, dst_path)


def mosaic_year(fixed_files, output_tif):
    temp_output = output_tif + ".part.tif"

    if os.path.exists(temp_output):
        os.remove(temp_output)

    src_files = [rasterio.open(path) for path in fixed_files]

    try:
        mosaic, out_transform = merge(src_files, method=MERGE_METHOD, nodata=OUT_NODATA)

        out_meta = src_files[0].meta.copy()
        out_meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=mosaic.shape[0],
            compress="lzw",
            nodata=OUT_NODATA,
            crs=TARGET_CRS,
        )

        with rasterio.open(temp_output, "w", **out_meta) as dst:
            dst.write(mosaic)

        os.replace(temp_output, output_tif)

    finally:
        for src in src_files:
            src.close()


def run_annual_mosaic():
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    intermediate_root = os.path.join(FINAL_OUTPUT_DIR, "_intermediate_fixed4326")
    os.makedirs(intermediate_root, exist_ok=True)

    progress_path = os.path.join(FINAL_OUTPUT_DIR, "ET_KNN_top28_mosaic_progress_2019_2025.csv")
    summary_path = os.path.join(FINAL_OUTPUT_DIR, "ET_KNN_top28_yearly_mosaic_summary_2019_2025.csv")

    progress_df = load_progress(
        progress_path,
        [
            "year",
            "task_type",
            "source_tif",
            "intermediate_tif",
            "mosaic_output",
            "status",
            "message",
        ],
    )

    summary_rows = []

    for year in YEARS:
        input_dir = INITIAL_OUTPUT_TEMPLATE.format(year=year)
        tile_list = find_classified_tiles(input_dir)

        fixed_files = []

        for idx, tif_path in enumerate(tile_list, 1):
            dst_path = make_intermediate_path(year, tif_path, intermediate_root)

            if os.path.exists(dst_path) and not FORCE_REBUILD_INTERMEDIATE:
                fixed_files.append(dst_path)
                record = {
                    "year": year,
                    "task_type": "intermediate",
                    "source_tif": tif_path,
                    "intermediate_tif": dst_path,
                    "mosaic_output": "",
                    "status": "skipped_exists",
                    "message": "",
                }
                progress_df = update_progress(
                    progress_df,
                    record,
                    key_columns=["year", "task_type", "source_tif"],
                )
                save_dataframe(progress_df, progress_path)
                continue

            try:
                fix_orientation_and_reproject(tif_path, dst_path)
                fixed_files.append(dst_path)
                status = "success"
                message = ""
            except Exception as exc:
                status = "failed"
                message = str(exc)

            record = {
                "year": year,
                "task_type": "intermediate",
                "source_tif": tif_path,
                "intermediate_tif": dst_path if status == "success" else "",
                "mosaic_output": "",
                "status": status,
                "message": message,
            }
            progress_df = update_progress(
                progress_df,
                record,
                key_columns=["year", "task_type", "source_tif"],
            )
            save_dataframe(progress_df, progress_path)

        mosaic_output = os.path.join(FINAL_OUTPUT_DIR, f"{year}.tif")

        if len(fixed_files) == 0:
            summary_rows.append({
                "year": year,
                "input_dir": input_dir,
                "input_file_count": len(tile_list),
                "processed_file_count": 0,
                "mosaic_output": "",
                "status": "no_valid_tile",
            })
            continue

        if os.path.exists(mosaic_output) and not FORCE_REBUILD_MOSAIC:
            summary_rows.append({
                "year": year,
                "input_dir": input_dir,
                "input_file_count": len(tile_list),
                "processed_file_count": len(fixed_files),
                "mosaic_output": mosaic_output,
                "status": "mosaic_exists_skipped",
            })
            continue

        try:
            mosaic_year(fixed_files, mosaic_output)
            mosaic_status = "success"
            message = ""
        except Exception as exc:
            mosaic_status = "failed"
            message = str(exc)
            mosaic_output = ""

        record = {
            "year": year,
            "task_type": "mosaic",
            "source_tif": "",
            "intermediate_tif": "",
            "mosaic_output": mosaic_output,
            "status": mosaic_status,
            "message": message,
        }
        progress_df = update_progress(
            progress_df,
            record,
            key_columns=["year", "task_type", "source_tif"],
        )
        save_dataframe(progress_df, progress_path)

        summary_rows.append({
            "year": year,
            "input_dir": input_dir,
            "input_file_count": len(tile_list),
            "processed_file_count": len(fixed_files),
            "mosaic_output": mosaic_output,
            "status": mosaic_status,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return summary_df


def main():
    bundle = load_et_knn_bundle(MODEL_BUNDLE_PATH)

    print("Loaded ET + KNN top-28 model bundle.")
    print("Step 1 model: ET forest/non-forest classifier")
    print("Step 2 model: KNN forest-type classifier")
    print("Step 1 features:", bundle["step1_features"])
    print("Step 2 features:", bundle["step2_features"])

    prediction_summary = run_tile_prediction(bundle)
    mosaic_summary = run_annual_mosaic()

    print("\nTile prediction summary:")
    print(prediction_summary)
    print("\nAnnual mosaic summary:")
    print(mosaic_summary)
    print("\nClass codes:")
    print("0 = Deciduous broadleaved forest")
    print("1 = Deciduous needleleaved forest")
    print("2 = Evergreen broadleaved forest")
    print("3 = Evergreen needleleaved forest")
    print("4 = Mixed forest")
    print("5 = Other")
    print("255 = NoData")


if __name__ == "__main__":
    main()
