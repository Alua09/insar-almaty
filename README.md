🌍 InSAR.Almaty: Seismo-Space Monitoring of the Megapolis

**NASA Space Apps Challenge 2026 | Team: ALDA Space Team**

📌 Overview
**InSAR.Almaty** is an operational seismo-space monitoring dashboard designed to protect over 2 million residents of Almaty, Kazakhstan—a megapolis located in a high-risk seismic zone along active tectonic faults.

⚙️ How It Works
1. **Long-Term Crustal Deformation Monitoring (NASA NISAR InSAR):** Processes L-band GUNW radar interferograms (`asf-search`, `rasterio`, `scipy`) across multiple months. Tracking color progression (from blue to intense red) identifies hidden tectonic strain accumulation before major events occur.
2. **Real-Time Seismic Response (USGS API):** Monitors live seismic activity. When an earthquake with magnitude **M ≥ 4.5** is detected, the system triggers an emergency protocol, calculates safe evacuation routes on an interactive `Folium` map, and sends instant alerts via Telegram.

🛠️ Tech Stack
* **Frontend & Dashboard:** Python, Streamlit, Folium
* **Geospatial & Radar Processing:** `rasterio`, `scipy`, `asf-search` (NASA Earthdata)
* **Real-Time Data & Alerts:** USGS Earthquake API, Telegram Bot API

🚀 Live Demo
* **Interactive Dashboard:** [insar-almaty.streamlit.app](https://insar-almaty.streamlit.app) *(обновите ссылку после запуска)*
