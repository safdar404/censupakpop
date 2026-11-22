import streamlit as st
from streamlit_folium import st_folium

from core import analytics as A
from core.config import INDICATORS
from core.geo import build_map, load_boundaries, load_custom_boundaries, static_map_png
from core.ui import get_df, insight, page_header, register_chart

df = get_df()
page_header("Module 6", "Geospatial Heatmap & District Map",
            "Interactive choropleth of Pakistan's districts. Switch the indicator, hover a district for details, export a static map.")


@st.cache_data(show_spinner=False)
def _geo():
    return load_boundaries()


geo = st.session_state.get("custom_geo") or _geo()

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ind = st.selectbox("Map indicator", list(INDICATORS), format_func=lambda k: INDICATORS[k][0])
with c2:
    st.write("")
    st.write("")
    export = st.button("🖼️ Export static map (PNG)", width="stretch")
with c3:
    with st.popover("🌐 Use HDX boundaries", width="stretch"):
        st.caption("Upload district polygons (GeoJSON) from data.humdata.org - e.g. *pak_admbnda_adm2*.")
        gj = st.file_uploader("GeoJSON", type=["geojson", "json"])
        field = st.text_input("Name field", "ADM2_EN")
        if gj:
            st.session_state["custom_geo"] = load_custom_boundaries(gj, field)
            st.success("Custom boundaries loaded")
        if st.button("Use bundled boundaries"):
            st.session_state.pop("custom_geo", None)
            st.rerun()

label = INDICATORS[ind][0]
mc, tc = st.columns([5, 2])
with mc:
    m = build_map(df, geo, ind)
    st_folium(m, height=640, width=None, returned_objects=[], key=f"map_{ind}")
    st.caption("Polygons: open district boundaries (GADM via pakdata/gisdata). Districts created after that boundary "
               "release (e.g. Keamari, Korangi, Upper/Lower Dir) are shown as circles at their parent district. "
               "Load official HDX boundaries above for exact 2023 shapes.")
with tc:
    fmt = INDICATORS[ind][1]
    top = df.nlargest(10, ind)[["district", ind]]
    bot = df.nsmallest(10, ind)[["district", ind]]
    st.markdown(f"##### ⬆️ Top 10")
    st.dataframe(top.style.format({ind: fmt}), hide_index=True, width="stretch", height=390)
    st.markdown(f"##### ⬇️ Bottom 10")
    st.dataframe(bot.style.format({ind: fmt}), hide_index=True, width="stretch", height=390)

if export:
    with st.spinner("Rendering high-resolution map..."):
        png = static_map_png(df, geo, ind)
    register_chart(f"Map - {label}", png)
    st.image(png, caption=f"{label} - added to your PDF report", width=620)
    st.download_button("⬇️ Download PNG", png, f"pakistan_{ind}_map.png", "image/png")

nat = A.national_summary(df)
insight(f"Spatial pattern: density and urban share peak along the Karachi-Lahore-Islamabad/Peshawar corridor, "
        f"while Balochistan's large, sparsely populated districts drive the national average of {nat['density']:.0f} people/km².")
