import streamlit as st
import torch
from audiocraft.models import MusicGen
import os
import base64
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="MelodyAI - Music & Lyric Generator",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    
    .title-text {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .subtitle-text {
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    
    .lyric-box {
        background: rgba(0, 0, 0, 0.3);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ff416c;
        white-space: pre-wrap;
        font-style: italic;
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_model(version='facebook/musicgen-small'):
    """Loads the MusicGen model and caches it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = MusicGen.get_pretrained(version, device=device)
        return model
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

def generate_music_file(model, description, duration):
    """Generates music and returns the generated tensor."""
    model.set_generation_params(duration=duration)
    wav = model.generate([description], progress=True)
    return wav[0]

def save_audio(wav_tensor, filename):
    """Saves the wav tensor as a file."""
    from audiocraft.data.audio import audio_write
    audio_write(filename, wav_tensor.cpu(), 32000, strategy="loudness", loudness_compressor=True)

def generate_lyrics(prompt):
    """Simple rule-based lyric generator (Placeholder for LLM)."""
    # In a real app, you'd use OpenAI/Gemini API here.
    lyrics = f"""[Verse 1]
가슴 속 깊은 곳에서 시작된 작은 떨림
{prompt} 분위기에 취해 시간을 잊어봐
세상은 멈춘 듯 우리만의 멜로디
다신 오지 않을 이 순간을 노래해

[Chorus]
우리의 노래가 하늘에 닿기를
불확실한 내일도 두렵지 않아
{prompt} 비트 위에 꿈을 실어 보내
영원히 기억될 멜로디

[Outro]
Fade away...
Just like a dream..."""
    return lyrics

# App UI
def main():
    st.markdown('<h1 class="title-text">MelodyAI 🎵</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">텍스트 한 줄로 당신만의 명곡(과 가사)을 만들어보세요.</p>', unsafe_allow_html=True)

    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ 설정 (Settings)")
        model_size = st.selectbox(
            "모델 크기 선택",
            ["facebook/musicgen-small", "facebook/musicgen-medium"],
            index=0
        )
        duration = st.slider("음악 길이 (초)", 5, 30, 10)
        
        st.divider()
        st.markdown("### 🛠️ 설치 문제 해결")
        with st.expander("VC++ 오류 또는 설치 실패 시"):
            st.write("""
            1. **Python 버전 확인**: Python 3.10 또는 3.11 사용을 강력히 권장합니다. (3.13은 아직 지원이 미비할 수 있습니다.)
            2. **Build Tools**: [여기](https://visualstudio.microsoft.com/visual-cpp-build-tools/)에서 C++ 빌드 도구를 설치하세요.
            3. **ffmpeg**: 반드시 설치되어 있어야 오디오 저장이 가능합니다.
            """)
        
        st.divider()
        st.info("💡 안내: 이 AI는 '배경 음악'을 생성합니다. 목소리(VOCAL)는 포함되지 않지만, 가사를 함께 생성해드립니다.")

    # Main Interaction Area
    tab1, tab2 = st.tabs(["🎵 음악 생성", "📜 가사 도우미"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        description = st.text_area(
            "어떤 노래를 만들고 싶으신가요?",
            placeholder="예: K-Pop style dance music with upbeat energy, 120 BPM",
            height=80
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_btn = st.button("음악 생성하기")
        st.markdown('</div>', unsafe_allow_html=True)

        if generate_btn:
            if not description:
                st.warning("먼저 노래 설명을 입력해주세요!")
            else:
                try:
                    with st.status("인공지능이 음악을 작곡하고 있습니다...", expanded=True) as status:
                        st.write("모델 로딩 중...")
                        model = load_model(model_size)
                        if model is None: return
                        
                        st.write("멜로디 생성 중...")
                        wav = generate_music_file(model, description, duration)
                        
                        st.write("파일 변환 중...")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file_base = f"music_{timestamp}"
                        save_audio(wav, output_file_base)
                        
                        status.update(label="음악 생성 완료!", state="complete", expanded=False)
                    
                    output_file_full = f"{output_file_base}.wav"
                    if os.path.exists(output_file_full):
                        st.success("노래가 완성되었습니다!")
                        st.audio(output_file_full, format="audio/wav")
                        
                        # Show lyrics below music if created
                        st.markdown("#### 📜 어울리는 가사 (추천)")
                        st.markdown(f'<div class="lyric-box">{generate_lyrics(description)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

    with tab2:
        st.markdown("### 🖋️ 가사 생성기 (Lyrics Generator)")
        lyric_prompt = st.text_input("가사의 주제나 키워드를 입력하세요.", value=description if description else "")
        if st.button("가사만 생성하기"):
            st.markdown(f'<div class="lyric-box">{generate_lyrics(lyric_prompt)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
