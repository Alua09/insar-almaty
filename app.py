from datetime import datetime
import math
import os
import folium
import matplotlib.cm as cm
import numpy as np
import rasterio
import requests
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, label, median_filter
from scipy.ndimage import sum as nd_sum
import streamlit as st
from streamlit_folium import st_folium

try:
  import asf_search as asf

  ASF_AVAILABLE = True
except ImportError:
  ASF_AVAILABLE = False

# Streamlit Page Configuration
st.set_page_config(
    page_title="InSAR.Almaty — Control Center", page_icon="🛰️", layout="wide"
)

# UI Styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 0rem; padding-left: 1.5rem; padding-right: 1.5rem; }
    .metric-card {
        background-color: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-header { display: flex; justify-content: space-between; align-items: center; }
    .metric-label { font-size: 0.78rem; color: #64748b; font-weight: 600; }
    .metric-value { font-size: 1.05rem; color: #0f172a; font-weight: 700; margin-top: 4px; margin-bottom: 2px; }
    .metric-delta { font-size: 0.72rem; color: #10b981; font-weight: 600; line-height: 1.25; }
    .tooltip-container { position: relative; display: inline-block; cursor: help; }
    .tooltip-container .tooltip-text {
        visibility: hidden; width: 250px; background-color: #ffffff; color: #334155;
        text-align: left; border-radius: 6px; padding: 9px 11px; position: absolute;
        z-index: 100; bottom: 130%; right: 0; opacity: 0; transition: opacity 0.2s ease-in-out;
        border: 1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-size: 0.73rem; font-weight: normal; line-height: 1.4;
    }
    .tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
    .alert-safe {
        background-color: #f0fdf4; border: 1.5px solid #bbf7d0; color: #166534;
        padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .alert-danger {
        background-color: #fef2f2; border: 1.5px solid #fecaca; color: #991b1b;
        padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- PROJECT HEADER ---
st.markdown(
    """
    <h1 title="💡 About: The platform combines NASA radar spectra (NISAR L+S Band) and real-time USGS seismic feeds for deformation and seismic activity monitoring in Almaty." 
        style="color: #0284c7; font-size: 1.7rem; font-weight: 700; margin-bottom: 0px; line-height: 1.3; cursor: help; display: inline-block;">
        🛰️ InSAR.Almaty: Seismo-Space Monitoring of the Megapolis
    </h1>
    <p style="color: #475569; font-size: 0.9rem; margin-top: 2px; margin-bottom: 10px;">
        NASA Space Apps Challenge | Operational Deformation and Seismic Activity Control Center
    </p>
    """,
    unsafe_allow_html=True,
)

# --- COMPACT BANNER ---
st.markdown(
    """
    <div style="width: 100%; height: 120px; overflow: hidden; border-radius: 8px; margin-bottom: 8px;">
        <img src="https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1200&q=80" 
             style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    """,
    unsafe_allow_html=True,
)

# Metrics
m1, m2, m3, m4 = st.columns(4)

with m1:
  st.markdown(
      """
        <div class="metric-card">
            <div>
                <div class="metric-header">
                    <span class="metric-label">📍 Analysis Zone</span>
                    <div class="tooltip-container">ℹ️
                        <div class="tooltip-text"><b>Monitoring Region (450 km):</b> Almaty agglomeration zone and Northern Tien Shan. Covers the city and mountain ranges.</div>
                    </div>
                </div>
                <div class="metric-value">Almaty & Semirechye</div>
            </div>
            <div class="metric-delta">↑ Radius 450 km</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with m2:
  st.markdown(
      """
        <div class="metric-card">
            <div>
                <div class="metric-header">
                    <span class="metric-label">⚠️ Active Faults</span>
                    <div class="tooltip-container">ℹ️
                        <div class="tooltip-text"><b>Seismogenic Structures:</b><br>• Almaty Fault (Urban)<br>• Zailiyskiy Fault (~280–350 km)<br>• Kemin-Chilik Zone (~500 km)</div>
                    </div>
                </div>
                <div class="metric-value">3 Key Zones</div>
            </div>
            <div class="metric-delta">↑ Almaty, Zailiyskiy, Kemin-Chilik</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with m3:
  st.markdown(
      """
        <div class="metric-card">
            <div>
                <div class="metric-header">
                    <span class="metric-label">🛰️ NASA Radars</span>
                    <div class="tooltip-container">ℹ️
                        <div class="tooltip-text"><b>NISAR Spectra (L+S):</b><br>• L-Band — slopes & foothills (Medeu).<br>• S-Band — urban & city center.</div>
                    </div>
                </div>
                <div class="metric-value">NISAR L-Band (GUNW)</div>
            </div>
            <div class="metric-delta">↑ Real GeoTIFF Data</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with m4:
  st.markdown(
      """
        <div class="metric-card">
            <div>
                <div class="metric-header">
                    <span class="metric-label">⚡ API Status</span>
                    <div class="tooltip-container">ℹ️
                        <div class="tooltip-text"><b>Data Integration:</b> Automated live stream and synchronization with USGS catalogs 24/7.</div>
                    </div>
                </div>
                <div class="metric-value">ONLINE 24/7</div>
            </div>
            <div class="metric-delta">↑ Synchronized</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.markdown("---")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("### 🎛️ GIS & NISAR Control Panel")

selected_pair = st.sidebar.selectbox(
    "📅 NISAR InSAR Timeframe (2026):",
    [
        "June — July (Jun-Jul)",
        "July — August (Jul-Aug)",
        "August — September (Aug-Sep)",
        "September — October (Sep-Oct)",
        "October — June / Extra Pair",
        "June — October (Seasonal Summary)",
    ],
)

raster_files = {
    "June — July (Jun-Jul)": "displacement_pair_1_small.tif",
    "July — August (Jul-Aug)": "displacement_pair_2_small.tif",
    "August — September (Aug-Sep)": "displacement_pair_3_small.tif",
    "September — October (Sep-Oct)": "displacement_pair_4_small.tif",
    "October — June / Extra Pair": "displacement_pair_5_small.tif",
    "June — October (Seasonal Summary)": (
        "displacement_pair_cumulative_small.tif"
    ),
}

# --- MAP LEGEND ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Map Legend")
st.sidebar.markdown(
    """
    <div style="background: #f8fafc; border: 1.5px solid #e2e8f0; padding: 12px; border-radius: 8px; font-size: 0.8rem;">
        <div style="font-weight: 600; color: #1e293b; margin-bottom: 6px;">🛰️ InSAR Deformation Spectrum</div>
        <div style="background: linear-gradient(to right, #2563eb, #ffffff, #dc2626); height: 12px; border-radius: 4px; margin-bottom: 4px; border: 1px solid #cbd5e1;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #64748b; margin-bottom: 10px; font-weight: 500;">
            <span>🔵 Subsidence</span>
            <span>⚪ Stable</span>
            <span>🔴 Uplift</span>
        </div>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 8px; display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; background-color: #f97316; border-radius: 50%; display: inline-block; border: 1px solid #b45309;"></span>
                <span style="color: #334155;"><b>Earthquakes (USGS):</b> Shocks</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; background-color: #dc2626; border-radius: 50%; display: inline-block; border: 2px solid white; box-shadow: 0 0 2px rgba(0,0,0,0.5);"></span>
                <span style="color: #334155;"><b>Red markers:</b> InSAR Uplift Center</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; background-color: #2563eb; border-radius: 50%; display: inline-block; border: 2px solid white; box-shadow: 0 0 2px rgba(0,0,0,0.5);"></span>
                <span style="color: #334155;"><b>Blue markers:</b> InSAR Subsidence Center</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background-color: #10b981; color: white; border-radius: 4px; padding: 0px 4px; font-size: 11px; font-weight: bold;">🛡️</span>
                <span style="color: #334155;"><b>Green markers:</b> Evacuation Points</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 2px;">
                <span style="width: 16px; height: 4px; background-color: #dc2626; display: inline-block; border-radius: 2px;"></span>
                <span style="color: #334155;"><b>Almaty Fault</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 16px; height: 3px; background-color: #f97316; display: inline-block; border-radius: 2px;"></span>
                <span style="color: #334155;"><b>Zailiyskiy Fault</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 16px; height: 3px; background-color: #eab308; display: inline-block; border-radius: 2px;"></span>
                <span style="color: #334155;"><b>Kemin-Chilik Zone</b></span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Telegram PUSH Integration")
tg_token = st.sidebar.text_input(
    "Bot Token",
    value="9116406484:AAH8ZKEbT4F7b6z0zwpkuDBW-tlktKqb-50",
    type="password",
)
tg_chat_id = st.sidebar.text_input("Your Chat ID", value="2146204975")

ALMATY_LAT = 43.2389
ALMATY_LON = 76.8897


def calculate_distance(lat1, lon1, lat2, lon2):
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon_real = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
      math.radians(lat2)
  ) * math.sin(dlon_real / 2) ** 2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  return R * c


def send_telegram_alert(token, chat_id, message_text):
  if not token or not chat_id:
    return False, "Specify Token and Chat ID!"
  url = f"https://api.telegram.org/bot{token}/sendMessage"
  payload = {"chat_id": chat_id, "text": message_text, "parse_mode": "Markdown"}
  try:
    res = requests.post(url, json=payload, timeout=5)
    return (
        (True, "PUSH notification sent successfully!")
        if res.status_code == 200
        else (False, f"API Error: {res.text}")
    )
  except Exception as e:
    return False, f"Network Error: {str(e)}"


# USGS data gathering
filtered_events = []
high_magnitude_alerts = []
usgs_feed_url = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.geojson"
)

try:
  response = requests.get(usgs_feed_url, timeout=5)
  response.raise_for_status()
  data = response.json()
  now = datetime.now()
  for event in data.get("features", []):
    geom = event["geometry"]
    lon, lat = geom["coordinates"][0], geom["coordinates"][1]
    dist = calculate_distance(ALMATY_LAT, ALMATY_LON, lat, lon)
    if dist <= 450:
      event["distance_to_almaty"] = round(dist, 1)
      filtered_events.append(event)
      event_time = datetime.fromtimestamp(event["properties"]["time"] / 1000)
      hours_ago = (now - event_time).total_seconds() / 3600
      if event["properties"]["mag"] >= 4.5 and hours_ago <= 48:
        high_magnitude_alerts.append(event)
except Exception:
  pass

shelters = [
    {
        "name": "Central Stadium",
        "coords": [43.2383, 76.9242],
        "desc": "Evacuation Point #1 (Almaly District)",
    },
    {
        "name": "First President Park",
        "coords": [43.2044, 76.8831],
        "desc": "Evacuation Point #2 (Bostandyk District)",
    },
    {
        "name": "Astana Square / KBTU",
        "coords": [43.2572, 76.9458],
        "desc": "Evacuation Point #3 (Medeu District)",
    },
]

if high_magnitude_alerts:
  nearest_shelter = min(
      shelters,
      key=lambda s: calculate_distance(
          ALMATY_LAT, ALMATY_LON, s["coords"][0], s["coords"][1]
      ),
  )
  min_dist = round(
      calculate_distance(
          ALMATY_LAT,
          ALMATY_LON,
          nearest_shelter["coords"][0],
          nearest_shelter["coords"][1],
      ),
      2,
  )

# MAIN SCREEN
col_left, col_right = st.columns([1, 2], gap="medium")

with col_left:
  if high_magnitude_alerts:
    st.markdown(
        """<div class="alert-danger">🚨 AUTOMATIC ALERT: Seismic events M ≥ 4.5 detected in the last 48 hours!</div>""",
        unsafe_allow_html=True,
    )
    if "auto_alert_sent" not in st.session_state:
      auto_msg = (
          "🚨 **AUTOMATIC EMERGENCY ALERT (InSAR.Almaty)**\n\n"
          "⚠️ Fresh seismic event M ≥ 4.5 detected.\n\n"
          "📍 **Nearest Evacuation Point:**\n"
          f"👉 *{nearest_shelter['name']}* ({nearest_shelter['desc']})\n"
          f"📏 Distance: ~{min_dist} km\n\n"
          f"🗺 [Google Maps Navigation](https://www.google.com/maps/dir/?api=1&destination={nearest_shelter['coords'][0]},{nearest_shelter['coords'][1]})"
      )
      success, _ = send_telegram_alert(tg_token, tg_chat_id, auto_msg)
      if success:
        st.session_state["auto_alert_sent"] = True
        st.toast("⚡ Automatic Telegram PUSH successfully sent!", icon="🚀")
  else:
    st.markdown(
        """<div class="alert-safe">🟢 Alert System: No critical shock threats detected in the last 48 hours. Monitoring is normal.</div>""",
        unsafe_allow_html=True,
    )

  with st.expander("📡 Real-time Seismicity Monitoring (USGS)", expanded=True):
    if filtered_events:
      for event in filtered_events[:5]:
        prop = event["properties"]
        geom = event["geometry"]
        eq_time = datetime.fromtimestamp(prop["time"] / 1000).strftime(
            "%m-%d %H:%M"
        )
        mag, place, dist = prop["mag"], prop["place"], event["distance_to_almaty"]
        st.warning(
            f"**M:{mag}** | {place}\n📍 **{dist} km** from Almaty | ⏰ {eq_time}"
        )
    else:
      st.info("No recent earthquakes found within a 450 km radius.")

  with st.expander("📲 Manual PUSH Alert Testing (Demo)", expanded=False):
    if st.button("🚨 Simulate Emergency & Send Telegram PUSH"):
      demo_msg = (
          "🚨 **DEMO ALERT: SEISMIC SIMULATION**\n\n"
          "📍 **Your Nearest Evacuation Point:**\n"
          f"👉 *{shelters[0]['name']}* ({shelters[0]['desc']})\n"
          "📏 Distance: ~2.4 km\n\n"
          f"🗺 [Google Maps Navigation](https://www.google.com/maps/dir/?api=1&destination={shelters[0]['coords'][0]},{shelters[0]['coords'][1]})"
      )
      success, res_msg = send_telegram_alert(tg_token, tg_chat_id, demo_msg)
      if success:
        st.success(res_msg)
      else:
        st.error(res_msg)

  with st.expander("🛰️ NASA ASF Cloud Integration (asf-search)", expanded=True):
    st.markdown("Query NASA Earthdata catalog for NISAR GUNW interferograms.")
    if st.button("🔍 Search NASA ASF Archive"):
      if ASF_AVAILABLE:
        try:
          almaty_bbox = [76.5, 43.0, 77.5, 43.5]
          lon_min, lat_min, lon_max, lat_max = almaty_bbox
          wkt_polygon = (
              f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max}"
              f" {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
          )
          results = asf.geo_search(
              intersectsWith=wkt_polygon,
              start="2026-06-01",
              end="2026-07-01",
              processingLevel=["GUNW"],
          )
          st.success(
              f"Successfully connected to NASA ASF! Found {len(results)} NISAR"
              " GUNW scenes."
          )
        except Exception as e:
          st.error(f"ASF Search Error: {str(e)}")
      else:
        st.warning("`asf-search` library is not installed in the environment.")

with col_right:
  st.markdown(
      "#### 🗺️ Interactive Map of Almaty Region (NASA NISAR InSAR Layer)"
  )
  m = folium.Map(
      location=[ALMATY_LAT, ALMATY_LON], zoom_start=11, tiles="OpenStreetMap"
  )

  major_faults = [
      {
          "name": "Almaty Fault",
          "coords": [[43.250, 76.780], [43.235, 76.850], [43.220, 76.950]],
          "color": "#dc2626",
          "weight": 5,
      },
      {
          "name": "Zailiyskiy Fault",
          "coords": [[43.310, 76.720], [43.285, 76.840], [43.260, 76.980]],
          "color": "#f97316",
          "weight": 4,
      },
      {
          "name": "Kemin-Chilik Zone",
          "coords": [[43.180, 77.020], [43.130, 77.200], [43.070, 77.400]],
          "color": "#eab308",
          "weight": 4,
      },
  ]

  for fault in major_faults:
    folium.PolyLine(
        fault["coords"],
        color=fault["color"],
        weight=fault["weight"],
        opacity=0.9,
        tooltip=f"⚠️ Tectonic Fault: {fault['name']}",
    ).add_to(m)

  # Evacuation points
  for s in shelters:
    shelter_icon_html = """
            <div style="background-color: #10b981; width: 28px; height: 28px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; font-size: 14px;">
                🛡️
            </div>
        """
    shelter_div_icon = folium.DivIcon(
        html=shelter_icon_html, icon_size=(28, 28), icon_anchor=(14, 14)
    )
    folium.Marker(
        location=s["coords"],
        icon=shelter_div_icon,
        popup=folium.Popup(f"<b>{s['name']}</b><br>{s['desc']}", max_width=250),
        tooltip=f"🏠 Assembly Point: {s['name']}",
    ).add_to(m)

  # USGS Earthquakes
  for event in filtered_events:
    geom = event["geometry"]
    prop = event["properties"]
    lon, lat = geom["coordinates"][0], geom["coordinates"][1]
    mag = prop["mag"]
    place = prop["place"]
    folium.CircleMarker(
        location=[lat, lon],
        radius=max(float(mag) * 2.5, 5),
        color="#b45309",
        weight=2,
        fill=True,
        fill_color="#f97316",
        fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>📡 USGS Epicenter</b><br>{place}<br>Magnitude: {mag}",
            max_width=250,
        ),
        tooltip=f"USGS Earthquake (M: {mag})",
    ).add_to(m)

  # --- STANDARD RASTER RENDERING FROM NATIVE METADATA ---
  target_tif = raster_files.get(selected_pair)
  if target_tif and os.path.exists(target_tif):
    try:
      with rasterio.open(target_tif) as src:
        downsample_factor = 2
        h_down = src.height // downsample_factor
        w_down = src.width // downsample_factor

        img_data = src.read(1, out_shape=(h_down, w_down)).astype(float)
        bounds = src.bounds

        img_data[np.isnan(img_data)] = np.nan
        img_data[(img_data == 0) | (img_data > 10000) | (img_data < -10000)] = (
            np.nan
        )

        valid_mask = ~np.isnan(img_data)
        if np.any(valid_mask):
          filled_data = np.where(valid_mask, img_data, 0.0)

          trend = gaussian_filter(filled_data, sigma=30)
          detrended_data = img_data - trend
          detrended_data[~valid_mask] = np.nan

          smoothed_detrended = median_filter(
              np.nan_to_num(detrended_data, nan=0.0), size=3
          )
          smoothed_detrended[~valid_mask] = np.nan

          valid_vals = smoothed_detrended[~np.isnan(smoothed_detrended)]
          if valid_vals.size > 0:
            vmin, vmax = valid_vals.min(), valid_vals.max()
            v_abs_max = max(abs(vmin), abs(vmax))
            if v_abs_max == 0:
              v_abs_max = 1.0

            norm_data = (smoothed_detrended + v_abs_max) / (2 * v_abs_max)
            norm_data = np.clip(norm_data, 0, 1)

            colormap = cm.coolwarm
            rgba_float = colormap(norm_data)
            rgba_img = (rgba_float * 255).astype(np.uint8)

            noise_mask = (np.abs(smoothed_detrended) < (v_abs_max * 0.15)) | (
                ~valid_mask
            )

            def remove_small_clusters(mask, min_size=40):
              labeled, num_features = label(mask)
              if num_features == 0:
                return mask
              component_sizes = nd_sum(mask, labeled, range(1, num_features + 1))
              too_small = component_sizes < min_size
              remove_pixel = too_small[labeled - 1]
              labeled[remove_pixel] = 0
              return labeled > 0

            active_mask = ~noise_mask
            active_mask = remove_small_clusters(active_mask, min_size=40)

            rgba_img[~active_mask, 3] = 0
            rgba_img[active_mask, 3] = 210

            folium.raster_layers.ImageOverlay(
                image=rgba_img,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.9,
                name="NASA NISAR Spectral InSAR",
            ).add_to(m)

            # --- LOCAL DEFORMATION CENTERS VIA AFFINE TRANSFORM ---
            clean_data = np.nan_to_num(smoothed_detrended, nan=0.0)
            height_arr, width_arr = clean_data.shape
            max_local = (
                clean_data
                == ndimage.maximum_filter(
                    clean_data, size=max(30, height_arr // 10)
                )
            ) & (clean_data > v_abs_max * 0.4)
            min_local = (
                clean_data
                == ndimage.minimum_filter(
                    clean_data, size=max(30, height_arr // 10)
                )
            ) & (clean_data < -v_abs_max * 0.4)

            y_idxs, x_idxs = np.where(max_local | min_local)

            for y_idx, x_idx in zip(y_idxs[::4], x_idxs[::4]):
              orig_y = y_idx * downsample_factor
              orig_x = x_idx * downsample_factor
              lon, lat = rasterio.transform.xy(
                  src.transform, orig_y, orig_x
              )

              val = smoothed_detrended[y_idx, x_idx]
              color_marker = "#dc2626" if val > 0 else "#2563eb"
              label_type = (
                  "Local Uplift" if val > 0 else "Local Subsidence"
              )

              folium.CircleMarker(
                  location=[lat, lon],
                  radius=5,
                  color="#ffffff",
                  weight=1.5,
                  fill=True,
                  fill_color=color_marker,
                  fill_opacity=0.9,
                  popup=folium.Popup(
                      f"<b>🚨 InSAR Deformation Center</b><br>Type:"
                      f" {label_type}<br>Value: {val:.2f} rad",
                      max_width=250,
                  ),
                  tooltip=f"InSAR Center: {val:.2f} rad",
              ).add_to(m)

    except Exception as e:
      st.error(f"❌ Ошибка рендеринга растра: {str(e)}")
  else:
    st.warning(f"ℹ️ Файл `{target_tif}` не найден в репозитории.")

  st_folium(m, use_container_width=True, height=450, returned_objects=[])

st.markdown("---")
st.caption(
    "NASA Space Apps Challenge 2026 | ALDA Space Team — InSAR.Almaty Platform"
)
