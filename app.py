import glob
import os
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

# Путь к папке с сырыми данными .h5
data_dir = r"D:\sar_data_Almaty"

# Путь для сохранения готовых .tif файлов
output_dir = r"C:\InSAR.Almaty"
os.makedirs(output_dir, exist_ok=True)

# Ищем все .h5 файлы и сортируем их
h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

if not h5_files:
  print(f"❌ Файлы .h5 не найдены в директории: {data_dir}")
else:
  print(f"🎯 Найдено сырых файлов NISAR: {len(h5_files)}\n")

  cumulative_array = None
  dst_transform_cumulative = None
  dst_crs_cumulative = None
  dst_height_cum = None
  dst_width_cum = None

  for i, input_path in enumerate(h5_files):
    filename = os.path.basename(input_path)
    output_tif = os.path.join(output_dir, f"displacement_pair_{i + 1}.tif")

    print(
        f"📁 [{i + 1}/{len(h5_files)}] Конвертация пары {i + 1}"
        f" ({filename[:35]}...) в EPSG:4326..."
    )

    with h5py.File(input_path, "r") as f:
      base_path = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"

      if base_path in f:
        displacement = f[f"{base_path}/unwrappedPhase"][()].astype(np.float32)
        x_coords = f[f"{base_path}/xCoordinates"][()]
        y_coords = f[f"{base_path}/yCoordinates"][()]

        left, right = x_coords.min(), x_coords.max()
        bottom, top = y_coords.min(), y_coords.max()
        height, width = displacement.shape

        src_crs = "EPSG:32643"
        src_transform = from_bounds(left, bottom, right, top, width, height)

        dst_crs = "EPSG:4326"
        transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, left, bottom, right, top
        )

        dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=displacement,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        # Сохраняем месячный/периодичный GeoTIFF
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=rasterio.float32,
            crs=dst_crs,
            transform=transform,
            nodata=0,
        ) as dst:
          dst.write(dst_array, 1)

        print(f"  ✅ Сохранено: displacement_pair_{i + 1}.tif")

        # Накапливаем фазу для кумулятивного файла
        valid_mask = (dst_array != 0) & (~np.isnan(dst_array))
        if cumulative_array is None:
          cumulative_array = np.zeros_like(dst_array)
          dst_transform_cumulative = transform
          dst_crs_cumulative = dst_crs
          dst_height_cum = dst_height
          dst_width_cum = dst_width

        cumulative_array = np.where(
            valid_mask, cumulative_array + dst_array, cumulative_array
        )

      else:
        print(f"⚠️ Путь {base_path} не найден в файле {filename}")

  # Создаем кумулятивный файл за весь сезон
  if cumulative_array is not None:
    cumulative_tif = os.path.join(output_dir, "displacement_pair_cumulative.tif")
    with rasterio.open(
        cumulative_tif,
        "w",
        driver="GTiff",
        height=dst_height_cum,
        width=dst_width_cum,
        count=1,
        dtype=rasterio.float32,
        crs=dst_crs_cumulative,
        transform=dst_transform_cumulative,
        nodata=0,
    ) as dst:
      dst.write(cumulative_array, 1)
    print(
        "\n🎉 Успешно сгенерирован кумулятивный сезонный файл:"
        f" {cumulative_tif}"
    )

  print(
      "\n🚀 Конвертация завершена! Все файлы (включая кумулятивный) лежат в"
      r" C:\InSAR.Almaty"
  )
