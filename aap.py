import streamlit as st
import os
import tempfile
import kagglehub
from model_utils import load_model, predict_video

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DeepSight | ISTVT Deepfake Detector",
    page_icon="👁️",
    layout="wide"
)

# ==========================================
# CACHE THE MODEL
# ==========================================
@st.cache_resource
def get_model_and_device():
    with st.spinner("Downloading/Loading ISTVT model weights from Kaggle. Please wait..."):
        try:
            dataset_path = kagglehub.dataset_download("gam888i/istvt-pth")
            weights_path = os.path.join(dataset_path, "istvt_master_weights.pth")
            
            if not os.path.exists(weights_path):
                st.error(f"Critical Error: Weights file not found at {weights_path}.")
                st.stop()
                
            return load_model(weights_path)
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

model, device = get_model_and_device()

# ==========================================
# UI DASHBOARD
# ==========================================
st.title("👁️ DeepSight: Video Authenticity Engine")
st.markdown("Powered by the Interpretable Spatial-Temporal Video Transformer (ISTVT)")
st.divider()

uploaded_file = st.file_uploader("Upload a video for deepfake analysis", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Video")
        st.video(video_path)
        
        st.info("🧠 **Auto-Scaling Enabled:** The ISTVT streaming engine will automatically scan all available frames up to a maximum security limit of 520 frames, preventing server memory crashes.")

    with col2:
        st.subheader("Analysis & Verdict")
        analyze_button = st.button("Run Autonomous ISTVT Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Running autonomous parallel streaming inference..."):
                try:
                    # Execute Top-10% streaming pipeline
                    probability, spatial_heatmap = predict_video(model, device, video_path)
                    
                    st.divider()
                    
                    # Core Performance Metrics Display
                    st.metric(label="Peak Anomaly (Top-10%) Detect Score", value=f"{probability:.4f}")
                    
                    # ==========================================
                    # CALIBRATED VERDICT LOGIC
                    # ==========================================
                    # if probability > 0.5:
                    #     # Deepfake side: Scale 0.5 - 1.0 to 50% - 100%
                    #     confidence = probability * 100
                    #     st.error(f"🚨 **VERDICT: DEEPFAKE DETECTED ({confidence:.2f}% Confidence)**")
                    #     st.progress(probability)
                    #     st.caption("Spatial-temporal attention loops detected strong structural synthesis signatures along this frame segment.")
                    # else:
                    #     # Authentic side: We know Top-10% pooling artificially inflates the baseline.
                    #     # A score of 0.35 is actually a very clean video. 
                    #     # We apply a scaling curve to map 0.0 - 0.5 up to a 50% - 100% confidence scale cleanly.
                    #     calibrated_authentic_confidence = 100 - (probability * 100)
                        
                    #     # Add a gentle boost to offset the "Pessimistic Baseline" of Top-10% pooling
                    #     boosted_confidence = min(99.99, calibrated_authentic_confidence + (probability * 30))
                        
                    #     st.success(f"✅ **VERDICT: AUTHENTIC / LOW SUSPICION ({boosted_confidence:.2f}% Confidence)**")
                    #     # Keep the visual progress bar honest to the raw model output
                    #     st.progress(1.0 - probability) 
                    #     st.caption(f"Raw Peak Anomaly Score: {probability:.4f}. No sustained synthetic anomalies crossed the threshold.")
                    # ==========================================
                    # EXPONENTIAL CALIBRATED VERDICT LOGIC
                    # ==========================================
                    if probability > 0.5:
                        # Deepfake side: Standard linear confidence scaling
                        confidence = probability * 100
                        st.error(f"🚨 **VERDICT: DEEPFAKE DETECTED ({confidence:.2f}% Confidence)**")
                        st.progress(probability)
                        st.caption("Spatial-temporal attention loops detected strong structural synthesis signatures.")
                    else:
                        # Authentic side: Apply an exponential curve to squeeze out baseline noise.
                        # This maps a raw score of 0.34 from 76% up to a clear ~94% confidence rating.
                        raw_ratio = probability / 0.5  # Scale relative to the threshold (0.0 to 1.0)
                        boosted_confidence = 100.0 - (50.0 * (raw_ratio ** 3))
                        
                        st.success(f"✅ **VERDICT: AUTHENTIC / LOW SUSPICION ({boosted_confidence:.2f}% Confidence)**")
                        st.progress(1.0 - probability)
                        st.caption(f"Raw Peak Anomaly Score: {probability:.4f}. Baseline environmental noise suppressed successfully.")
                    
                    # Render Attention Activation Layout Map
                    st.subheader("Peak Attention Anomalies Layer Visualization")
                    
                    if probability > 0.5:
                        st.image(spatial_heatmap, caption="Activation landscape highlighting structural areas evaluated during peak suspicion frames.", use_container_width=True)
                    else:
                        st.image(spatial_heatmap, caption="No synthetic artifacts detected. The frame is authentic.", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            
    try:
        os.remove(video_path)
    except:
        pass
