import streamlit as st

def load_css():
    # Default Dark Theme Colors
    bg_gradient = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    text_color = "#f8fafc"
    card_bg = "rgba(255, 255, 255, 0.05)"
    card_border = "rgba(255, 255, 255, 0.1)"
    sidebar_bg = "rgba(15, 23, 42, 0.95)"
    metric_label = "#94a3b8"

    st.markdown(f"""
<style>
    /* Global Reset & Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        color: {text_color};
    }}

    /* Animated Deep Background */
    .stApp {{
        background: {bg_gradient};
        color: {text_color};
    }}

    /* Glass Cards */
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid {card_border};
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .glass-card:hover {{
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }}

    /* Fade In Animation */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s ease-out forwards;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid {card_border};
    }}

    /* Typography */
    h1, h2, h3 {{
        color: {text_color} !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    p, label, .stMarkdown, .stText {{
        color: {text_color} !important;
        opacity: 0.9;
    }}
    
    .stCaption {{
        color: {metric_label} !important;
    }}

    /* Custom Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, #2563eb 0%, #d946ef 100%);
        color: white !important;
        border: none;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Force all text inside button to be white */
    .stButton > button * {{
        color: white !important;
        fill: white !important;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 8px 30px rgba(217, 70, 239, 0.5);
        transform: translateY(-2px) scale(1.02);
        background: linear-gradient(90deg, #1d4ed8 0%, #c026d3 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 10px rgba(217, 70, 239, 0.3);
    }}

    /* Metrics */
    .metric-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    
    .metric-value {{
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }}
    
    .metric-label {{
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {metric_label};
        font-weight: 500;
    }}
    
    .metric-icon {{
        font-size: 24px;
        margin-bottom: 8px;
        opacity: 0.8;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {card_bg};
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: none;
        color: {metric_label};
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 255, 255, 0.1);
        color: {text_color};
        font-weight: 600;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {card_bg};
        border-radius: 10px;
        border: 1px solid {card_border};
        color: {text_color} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 0 0 10px 10px;
        border: 1px solid {card_border};
        border-top: none;
    }}

</style>
""", unsafe_allow_html=True)
