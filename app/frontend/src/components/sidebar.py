import streamlit as st
from src.utils.api_client import api_client

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 NeuroRestore")
        st.markdown("Professional Image Restoration")
        
        st.markdown("---")
        
        # Session History
            
        if st.session_state['history']:
            with st.expander("📜 Session History", expanded=False):
                st.caption(f"Total Restorations: {len(st.session_state['history'])}")
                for i, item in enumerate(reversed(st.session_state['history'][-5:])):
                    st.markdown(f"**{len(st.session_state['history']) - i}.** {item['filename']}")
                    st.caption(f"Model: {item['model']} | Time: {item['time']:.2f}s")
            st.markdown("---")
        
        st.markdown("### ⚙️ Model Settings")
        model_type = st.selectbox(
            "Restoration Engine",
            ("VAE (Probabilistic)", "GAN (Adversarial)"),
            help="VAE is faster but blurrier. GAN produces sharper details."
        )
        
        st.markdown("### 🎨 Preprocessing Options")
        
        use_preprocessing = st.checkbox("Enable Preprocessing", value=True, 
                                       help="Apply augmentation and degradation before model inference")
        
        preprocessing_options = {}
        if use_preprocessing:
            with st.expander("⚡ Augmentation", expanded=False):
                preprocessing_options['apply_augment'] = st.checkbox("Enable Augmentation", value=False)
                if preprocessing_options['apply_augment']:
                    st.caption("Geometric & colorimetric transforms")
            
            with st.expander("💥 Degradation", expanded=True):
                preprocessing_options['apply_degrade'] = st.checkbox("Enable Degradation", value=True)
                if preprocessing_options['apply_degrade']:
                    st.caption("Realistic noise, blur, scratches, rain drops")
        else:
            preprocessing_options['apply_augment'] = False
            preprocessing_options['apply_degrade'] = False
        
        st.markdown("### 📁 Input")
        uploaded_file = st.file_uploader("Drop degraded image", type=['png', 'jpg', 'jpeg'])
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Check Backend Status
        status = api_client.check_status()
        if status:
            st.success(f"Backend Online")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption("VAE Model")
                st.markdown("🟢 Ready" if status['models']['vae'] else "🔴 Offline")
            with col_s2:
                st.caption("GAN Model")
                st.markdown("🟢 Ready" if status['models']['gan'] else "🔴 Offline")
            
            # Show image processing services
            if status.get('services'):
                st.caption("Image Processing")
                services = status['services']
                st.markdown(
                    f"Augment: {'🟢' if services.get('augmentation') else '🔴'} | "
                    f"Degrade: {'🟢' if services.get('degradation') else '🔴'}"
                )
        else:
            st.error("Backend Offline")
            st.caption("Ensure Docker container is running")
            
        return model_type, uploaded_file, use_preprocessing, preprocessing_options

