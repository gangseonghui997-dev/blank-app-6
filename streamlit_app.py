import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
from datetime import datetime

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="MelodyAI Pro - AI Music Studio",
    page_icon="🎹",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;800&family=Nanum+Gothic:wght@400;700&display=swap');
    
    :root {
        --primary: #8a2be2;
        --secondary: #ff4b2b;
        --bg-color: #0d0d12;
        --card-bg: rgba(255, 255, 255, 0.03);
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Nanum Gothic', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e, #0d0d12);
        color: #efefef;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(13, 13, 18, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .studio-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .studio-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 30px rgba(138, 43, 226, 0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #8a2be2 0%, #ff4b2b 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 75, 43, 0.4);
    }
    
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 1rem !important;
    }
    
    .lyric-container {
        border-left: 3px solid var(--secondary);
        padding-left: 1.5rem;
        color: #ccc;
        line-height: 1.8;
        font-size: 1.1rem;
        white-space: pre-wrap;
    }
    
    .neon-text {
        text-shadow: 0 0 8px rgba(138, 43, 226, 0.8);
    }
    
    .badge {
        background: var(--primary);
        color: white;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL & LYRIC ENGINE
# ==========================================

@st.cache_resource
def get_model(version: str = "facebook/musicgen-small"):
    """
    MusicGen 모델 로더 (GPU 있으면 cuda, 없으면 cpu).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained(version)
    model.to(device)
    return model, device


def lyric_engine(genre, mood, topic):
    templates = {
        "K-Pop": f"""[Verse 1]
잠깐만 Stop, 눈이 마주친 그 순간
심장 소리는 이미 Up to the roof (Yeah)
{topic}의 색깔로 물들어가는 맘
피할 수 없어 This attraction, it's true

[Pre-Chorus]
심장이 뛰는 리듬 속에
너와 나만이 아는 이 느낌 (Feel it)

[Chorus]
Oh oh, {mood}하게 터지는 Melody
우리만의 스테이지 위로 Higher
오늘처럼 {mood}한 밤이면 난
너와 함께 춤을 춰 Tonight!""",

        "Ballad": f"""[Intro]
스미는 바람 끝에 우리 기억이...

[Verse 1]
차가운 공기 속에 섞인 {topic}의 향기
잊으려 할수록 더욱 선명해지네요
{mood}한 그대의 뒷모습을 보며
혼자 남겨진 이 길을 걷고 있죠

[Chorus]
우리의 노래가 이제는 슬픈 소음이 되어
바람결에 흩어지는 이 마음
{mood}했던 그 날의 우리를
다시 한 번만 안아줄 수 있다면...""",

        "Hip-Hop": f"""[Intro]
Yo, check the vibe. It's {mood} season.

[Verse 1]
거리 위에 흩뿌려진 {topic}의 조각들
난 그 위를 걸어가지 Like a king of the world
내 안의 {mood}함을 태우고 더 높이 가
Haters들은 뒤로, 우린 앞만 보고 달려 나 (Skrr)

[Chorus]
Volume을 높여, 내 꿈이 들리게
이 비트 위에 내 혼을 다 쏟아낼게
{topic}이 내 미래를 바꿀 때까지
We never stop, we keep it real!""",

        "Lo-Fi": f"""[Verse]
창밖을 봐, 비 내리는 소리와 함께
{topic} 한 잔에 {mood}한 감정이 섞여
천천히 흘러가는 구름처럼
난 그냥 이 자리에 머물고 싶어

[Chorus]
우우- 정답은 없어도 괜찮아
오늘처럼 {mood}한 분위기에
그냥 몸을 맡겨봐...""",
    }
    return templates.get(genre, "원하는 장르와 어울리는 가사를 준비 중입니다.")


# ==========================================
# APP LAYOUT
# ==========================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="neon-text">🎹 STUDIO SETTINGS</h2>', unsafe_allow_html=True)
        st.divider()

        pro_mode = st.toggle("Pro Mode - Advanced Params", value=False)

        model_name = st.selectbox(
            "Model Version",
            ["facebook/musicgen-small", "facebook/musicgen-medium"],
            index=0
        )

        st.divider()
        st.markdown("### 🛠 Troubleshooting")
        with st.expander("VC++ Error? Click here"):
            st.error("Windows Python 3.13 환경에서는 C++ 빌드 도구가 필수적입니다.")
            st.write("1. Python 3.10/3.11 사용 권장")
            st.write("2. Visual Studio Build Tools 설치")
            st.code("pip install spacy==3.8.0 --no-build-isolation")

        st.divider()
        st.info("💡 Tip: 프롬프트에 '808 bass', 'high quality', 'mastered' 등을 추가해보세요.")

    # Header
    st.markdown('<div class="badge">V2.0 PRO EDITION</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title-text">MelodyAI <span style="color:var(--primary)">Studio</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text" style="margin-bottom:3rem;">AI와 함께 완성하는 나만의 프리미엄 작곡 사이트</p>', unsafe_allow_html=True)

    tab_music, tab_lyrics, tab_about = st.tabs(["🎵 MUSIC PRODUCTION", "📜 LYRIC STUDIO", "⚙️ SYSTEM INFO"])

    # =======================
    # MUSIC TAB
    # =======================
    with tab_music:
        container = st.container()
        with container:
            col_in, col_set = st.columns([2, 1])

            with col_in:
                st.markdown('<div class="studio-card">', unsafe_allow_html=True)
                st.subheader("1. 가제와 컨셉 (Concept)")
                song_topic = st.text_input("노래의 주제를 입력하세요 (예: 첫사랑, 새벽 감성, 성공)", "비 내리는 서울")

                music_description = st.text_area(
                    "사운드 설명 (Music Prompt)",
                    placeholder="예: Cinematic lo-fi hip hop, soulful piano, rain sounds, 80 BPM, high resolution",
                    value="Modern K-Pop style upbeat dance song, catchy synth, 120 BPM",
                    height=100
                )

                duration = st.slider("생성 길이 (초)", 5, 30, 10, step=5)
                generate_music_btn = st.button("🔥 GENERATE MASTER TRACK")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_set:
                st.markdown('<div class="studio-card">', unsafe_allow_html=True)
                st.subheader("2. 스타일 필터 (Style)")
                genre = st.selectbox("장르 선택", ["K-Pop", "Ballad", "Hip-Hop", "Lo-Fi", "Rock"])
                mood = st.select_slider("무드 (Mood)", ["Deep/Dark", "Mellow", "Neutral", "Bright", "Explosive"], value="Neutral")

                if pro_mode:
                    st.divider()
                    st.subheader("3. 정교한 믹싱")
                    st.slider("Loudness Compression", 0.0, 1.0, 0.5)
                    st.checkbox("High Fidelity (32kHz)", value=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if generate_music_btn:
            if not music_description:
                st.warning("먼저 사운드 설명을 입력해주세요!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    with st.status("AI Studio가 가동 중입니다...", expanded=True) as status:
                        status_text.write("🔌 모델 불러오는 중...")
                        model, device = get_model(model_name)
                        progress_bar.progress(25)

                        status_text.write("🎹 악보 그리는 중 (AI Compositing)...")
                        final_prompt = f"{music_description}, {genre} style, {mood} mood, studio quality"

                        # 최신 스타일: descriptions 인자 명시
                        model.set_generation_params(duration=duration)
                        with torch.no_grad():
                            wav = model.generate(
                                descriptions=[final_prompt],
                                progress=True
                            )
                        progress_bar.progress(75)

                        status_text.write("🎚️ 마스터링 및 품질 보정 중...")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_base = f"track_{timestamp}"
                        audio_write(out_base, wav[0].cpu(), 32000, strategy="loudness")
                        progress_bar.progress(100)

                        status.update(label="곡이 완성되었습니다!", state="complete", expanded=False)

                    full_file = f"{out_base}.wav"
                    if os.path.exists(full_file):
                        st.markdown('<div class="studio-card" style="text-align:center;">', unsafe_allow_html=True)
                        st.success(f"Track '{song_topic}' is Ready!")
                        st.audio(full_file, format="audio/wav")

                        st.markdown("---")
                        st.markdown("### 🎙️ AI 추천 가사")
                        st.markdown(
                            f'<div class="lyric-container">{lyric_engine(genre, mood, song_topic)}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("오디오 파일이 생성되지 않았습니다.")
                except Exception as e:
                    st.error(f"생성 중 오류 발생: {e}")

    # =======================
    # LYRIC TAB
    # =======================
    with tab_lyrics:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        col_la, col_lb = st.columns([1, 1])

        with col_la:
            st.subheader("가사 생성 설정")
            lyric_genre = st.selectbox("장르", ["K-Pop", "Ballad", "Hip-Hop", "Lo-Fi"], key="lyric_g")
            lyric_mood = st.text_input("감정 키워드", "신나는, 청량한", key="lyric_m")
            lyric_topic = st.text_input("핵심 단어 (Topic)", "비 내리는 서울", key="lyric_t")
            if st.button("가사만 정교하게 뽑기"):
                st.session_state['last_lyrics'] = lyric_engine(lyric_genre, lyric_mood, lyric_topic)

        with col_lb:
            st.subheader("결과물 (Script)")
            if 'last_lyrics' in st.session_state:
                st.markdown(
                    f'<div class="lyric-container">{st.session_state["last_lyrics"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("왼쪽 설정을 마치고 생성 버튼을 눌러주세요.")
        st.markdown('</div>', unsafe
