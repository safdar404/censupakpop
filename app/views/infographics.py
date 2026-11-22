import streamlit as st
import streamlit.components.v1 as components

from core.ui import get_df, page_header
from core.webviz import render

df = get_df()
page_header("Geo-infographics", "Pakistan in Numbers",
            "Census 2023 at a glance: headline numbers, province profiles with map silhouettes, a district bubble map and "
            "'if Pakistan were 100 people'.")
components.html(render("infographics.html", df), height=1660, scrolling=False)
st.caption("Built with D3.js. Province silhouettes and the bubble map use the district boundary data; bubble size = population.")
