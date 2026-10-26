import streamlit as st
from streamlit_image_comparison import image_comparison
import io
from PIL import Image

def render_metrics(original_image, process_time, model_type):
    st.markdown(f"""
    <div class="glass-card fade-in" style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div class="metric-container">
            <div class="metric-icon">📐</div>
            <div class="metric-value">{original_image.size[0]} x {original_image.size[1]}</div>
            <div class="metric-label">Resolution</div>
        </div>
        <div class="metric-container">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{process_time:.3f}s</div>
            <div class="metric-label">Inference Time</div>
        </div>
        <div class="metric-container">
            <div class="metric-icon">🤖</div>
            <div class="metric-value">{model_type.split()[0]}</div>
            <div class="metric-label">Model Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_comparison(original_image, processed_image, label="Restored"):
    st.markdown("### 🔍 Analysis & Comparison")
    
    tab1, tab2 = st.tabs(["↔️ Slider Comparison", "🖼️ Side-by-Side"])
    
    # Convert images to RGB to ensure compatibility
    img1 = original_image.convert("RGB")
    img2 = processed_image.convert("RGB")
    
    with tab1:
        image_comparison(
            img1=img1,
            img2=img2,
            label1="Original",
            label2=label,
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, use_container_width=True, caption="Original Input")
        with col2:
            st.image(img2, use_container_width=True, caption=f"{label} Output")

def create_side_by_side_image(img1, img2):
    # Resize images to match height
    height = max(img1.height, img2.height)
    img1_ratio = height / img1.height
    img2_ratio = height / img2.height
    
    img1 = img1.resize((int(img1.width * img1_ratio), height))
    img2 = img2.resize((int(img2.width * img2_ratio), height))
    
    # Create new image
    total_width = img1.width + img2.width + 20 # 20px padding
    new_im = Image.new('RGB', (total_width, height), (255, 255, 255))
    
    new_im.paste(img1, (0, 0))
    new_im.paste(img2, (img1.width + 20, 0))
    
    return new_im

def render_download_button(original_image, processed_image, filename, label="Restored"):
    col1, col2 = st.columns(2)
    
    with col1:
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        st.download_button(
            label=f"📥 Download {label} Image",
            data=buf.getvalue(),
            file_name=f"restored_{filename}",
            mime="image/png",
            use_container_width=True
        )
        
    with col2:
        # Create side-by-side comparison
        comparison_img = create_side_by_side_image(original_image, processed_image)
        buf_comp = io.BytesIO()
        comparison_img.save(buf_comp, format="PNG")
        st.download_button(
            label="📑 Download Comparison Report",
            data=buf_comp.getvalue(),
            file_name=f"comparison_{filename}",
            mime="image/png",
            use_container_width=True
        )

def render_empty_state():
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; opacity: 0.5;">
        <h2>Waiting for Input</h2>
        <p>Select an image from the sidebar to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
