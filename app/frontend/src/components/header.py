import streamlit as st

def render_header():
    st.markdown("""
    <div class="glass-card fade-in">
        <h1 style="text-align: center; margin-bottom: 10px;">🧬 NeuroRestore Pro</h1>
        <p style="text-align: center; font-size: 1.1rem; opacity: 0.8;">
            Advanced Neural Image Restoration System
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Quick Guide", expanded=False):
        st.markdown("""
        ### How to use NeuroRestore Pro
        1. **Select a Model**: Choose between **VAE** (faster) or **GAN** (sharper details) in the sidebar.
        2. **Configure Preprocessing**: Enable augmentation or degradation simulation if needed.
        3. **Upload Image**: Drag and drop your degraded image.
        4. **Run Process**: Click the "Run Restoration Process" button.
        5. **Analyze Results**: Compare the original and restored images using the interactive sliders and metrics.
        """)
