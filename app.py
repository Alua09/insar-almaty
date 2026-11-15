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
    <h1 style="color: #0284c7; font-size: 1.7rem; font-weight: 700; margin-bottom: 0px; line-height: 1.3;">
        🛰️ InSAR.Almaty: Seismo-Space Monitoring of the Megapolis
    </h1>
    <p style="color: #475569; font-size: 0.9rem; margin-top: 2px; margin-bottom: 10px;">
        NASA Space Apps Challenge | Operational Deformation and Seismic Activity Control Center
    </p>
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
                <div class="metric-header"><span class="metric-label">📍 Analysis Zone</span></div>
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
                <div class="metric-header"><span class="metric-label">⚠️ Active Faults</span></div>
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
                <div class="metric-header"><span class="metric-label">🛰️ NASA Radars</span></div>
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
                <div class="metric-header"><span class="metric-label">⚡ API Status</span></div>
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
        "June — October (Seasonal Summary)",
    ],
)

raster_files = {
    "June — July (Jun-Jul)": "displacement_pair_1_small.tif",
    "July — August (Jul-Aug)": "displacement_pair_2_small.tif",
    "August — September (Aug-Sep)": "displacement_pair_3_small.tif",
    "September — October (Sep-Oct)": "displacement_pair_4_small.tif",
    "June — October (Seasonal Summary)": (
        "displacement_pair_cumulative_small.tif"
    ),
}

st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Telegram PUSH Integration")
tg_token = st.sidebar.text_input(
    "Bot Token",
    value="9116406484:AAH8ZKEbT4F7b6z0zwpkuDBW-tlktKqb-50",
    type="password",
)
tg_chat_id = st.sidebar.text_input("Group Chat ID", value="-1004295077552")

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

      if event["properties"]["mag"] >= 4.5 and hours_ago <= 48 and dist <= 100:
        high_magnitude_alerts.append(event)
except Exception:
  pass

shelters = [
    {
        "name": "Central Stadium",
        "coords": [43.2383, 76.9242],
        "desc": "Evacuation Point #1 (Almaly)",
    },
    {
        "name": "First President Park",
        "coords": [43.2044, 76.8831],
        "desc": "Evacuation Point #2 (Bostandyk)",
    },
    {
        "name": "Astana Square / KBTU",
        "coords": [43.2572, 76.9458],
        "desc": "Evacuation Point #3 (Medeu)",
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
        """<div class="alert-danger">🚨 AUTOMATIC ALERT: Local seismic events M ≥ 4.5 detected within 100 km!</div>""",
        unsafe_allow_html=True,
    )
    if "auto_alert_sent" not in st.session_state:
      auto_msg = (
          "🚨 **AUTOMATIC EMERGENCY ALERT (InSAR.Almaty)**\n\n"
          "⚠️ Local seismic event M ≥ 4.5 detected near the city.\n\n"
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
        """<div class="alert-safe">🟢 Alert System: No critical local shocks (≤ 100 km) in the last 48 hours.</div>""",
        unsafe_allow_html=True,
    )

  with st.expander("📡 Real-time Seismicity Monitoring (USGS)", expanded=True):
    if filtered_events:
      for event in filtered_events[:5]:
        prop = event["properties"]
        eq_time = datetime.fromtimestamp(prop["time"] / 1000).strftime(
            "%m-%d %H:%M"
        )
        mag, place, dist = prop["mag"], prop["place"], event["distance_to_almaty"]
        st.warning(
            f"**M:{mag}** | {place}\n📍 **{dist} km** from Almaty | ⏰ {eq_time}"
        )
    else:
      st.info("No recent earthquakes found within a 450 km radius.")

  with st.expander("📲 Manual PUSH Alert Testing", expanded=False):
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

with col_right:
  st.markdown("#### 🗺️ Interactive Map of Almaty Region")
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
  ]

  for fault in major_faults:
    folium.PolyLine(
        fault["coords"],
        color=fault["color"],
        weight=fault["weight"],
        opacity=0.9,
        tooltip=fault["name"],
    ).add_to(m)

  for s in shelters:
    folium.Marker(
        location=s["coords"],
        popup=folium.Popup(f"<b>{s['name']}</b><br>{s['desc']}", max_width=250),
        tooltip=s["name"],
        icon=folium.Icon(color="green", icon="shield", prefix="fa"),
    ).add_to(m)

  for event in filtered_events:
    geom = event["geometry"]
    prop = event["properties"]
    lon, lat = geom["coordinates"][0], geom["coordinates"][1]
    mag = prop["mag"]
    folium.CircleMarker(
        location=[lat, lon],
        radius=max(float(mag) * 2.5, 5),
        color="#b45309",
        weight=2,
        fill=True,
        fill_color="#f97316",
        fill_opacity=0.85,
        tooltip=f"USGS M:{mag}",
    ).add_to(m)

  # Raster Overlay with safety checks
  target_tif = raster_files.get(selected_pair)
  if target_tif and os.path.exists(target_tif):
    try:
      with rasterio.open(target_tif) as src:
        img_data = src.read(1).astype(float)
        bounds = src.bounds
        img_data[np.isnan(img_data)] = np.nan
        valid_mask = ~np.isnan(img_data)
        if np.any(valid_mask):
          filled = np.where(valid_mask, img_data, 0.0)
          trend = gaussian_filter(filled, sigma=50)
          detrended = img_data - trend
          detrended[~valid_mask] = np.nan
          smoothed = median_filter(
              np.nan_to_num(detrended, nan=0.0), size=3
          )
          smoothed[~valid_mask] = np.nan

          valid_vals = smoothed[~np.isnan(smoothed)]
          if valid_vals.size > 0:
            v_max = max(abs(valid_vals.min()), abs(valid_vals.max()))
            if v_max == 0:
              v_max = 1.0
            norm = np.clip(
                (smoothed + v_max) / (2 * v_max), 0, 1
            )
            rgba = (cm.coolwarm(norm) * 255).astype(np.uint8)
            rgba[~valid_mask, 3] = 0
            rgba[valid_mask, 3] = 210

            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.9,
            ).add_to(m)
    except Exception as e:
      st.error(f"InSAR render error: {e}")

  st_folium(m, height=450, returned_objects=[], key=selected_pair)

st.markdown("---")
st.caption("NASA Space Apps Challenge 2026 | InSAR.Almaty Platform")
