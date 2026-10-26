import streamlit as st
from PIL import Image
from src.styles.css import load_css
from src.components.sidebar import render_sidebar
from src.components.header import render_header
from src.components.image_display import render_metrics, render_comparison, render_download_button, render_empty_state
from src.utils.api_client import api_client

# Page Config
st.set_page_config(
    page_title="NeuroRestore Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'processed_image' not in st.session_state:
        st.session_state['processed_image'] = None
    if 'process_time' not in st.session_state:
        st.session_state['process_time'] = 0
    if 'last_uploaded' not in st.session_state:
        st.session_state['last_uploaded'] = None

# Initialize Session State
init_session_state()

# Render Sidebar
model_type, uploaded_file, use_preprocessing, preprocessing_options = render_sidebar()

# Load CSS
load_css()

# Render Header
render_header()

# Main Logic
if uploaded_file is not None:
    # Load Image
    original_image = Image.open(uploaded_file)
    
    # Processing State
    if st.session_state.get('last_uploaded') != uploaded_file.name:
        st.session_state['processed_image'] = None
        st.session_state['process_time'] = 0
        st.session_state['last_uploaded'] = uploaded_file.name

    # Action Bar
    col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
    
    with col_act2:
        if st.button("✨ RUN RESTORATION PROCESS"):
            with st.spinner("Processing image on neural network..."):
                # Use enhanced pipeline with preprocessing
                if use_preprocessing:
                    result = api_client.enhance(
                        uploaded_file.getvalue(), 
                        model_type,
                        preprocess=True,
                        apply_augment=preprocessing_options.get('apply_augment', False),
                        apply_degrade=preprocessing_options.get('apply_degrade', True)
                    )
                else:
                    # Use basic prediction without preprocessing
                    result = api_client.predict(uploaded_file.getvalue(), model_type)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.session_state['processed_image'] = result['image']
                    st.session_state['process_time'] = result['process_time']
                    
                    # Update Session History
                    # Avoid duplicates if re-running same image/settings immediately
                    current_entry = {
                        'filename': uploaded_file.name,
                        'model': model_type,
                        'time': result['process_time']
                    }
                    
                    if not st.session_state['history'] or st.session_state['history'][-1] != current_entry:
                        st.session_state['history'].append(current_entry)

    # Results Area
    if st.session_state['processed_image']:
        st.markdown("---")
        
        # Determine label based on processing type
        output_label = "Preprocessed" if use_preprocessing else "Restored"
        
        render_metrics(original_image, st.session_state['process_time'], model_type)
        render_comparison(original_image, st.session_state['processed_image'], label=output_label)
        render_download_button(original_image, st.session_state['processed_image'], uploaded_file.name, label=output_label)

else:
    # Empty State
    render_empty_state()
