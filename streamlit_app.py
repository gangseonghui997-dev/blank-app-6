import streamlit as st
from openai import OpenAI

# 페이지 설정
st.set_page_config(page_title="K-Pop AI 작곡가", page_icon="🎵")

# 사이드바에서 API 키 입력 받기
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[OpenAI API 키 발급받기](https://platform.openai.com/account/api-keys)"

st.title("🎵 AI 한국어 노래 생성기")
st.caption("가사와 멜로디를 한 번에 만들어보세요!")

# 사용자 입력창
with st.form("song_form"):
    genre = st.selectbox("장르를 선택하세요", ["발라드", "K-Pop 댄스", "힙합", "인디", "트로트"])
    mood = st.text_input("어떤 분위기의 노래인가요? (예: 비 오는 날의 그리움, 신나는 여름 밤)")
    topic = st.text_area("노래에 포함하고 싶은 핵심 내용을 적어주세요.")
    
    submitted = st.form_submit_button("노래 만들기")

# 생성 로직
if submitted:
    if not openai_api_key:
        st.info("시작하려면 OpenAI API 키를 입력해주세요.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    with st.spinner("AI가 가사를 쓰고 노래를 부르는 중입니다..."):
        try:
            # 1. 가사 생성 단계
            lyric_prompt = f"장르: {genre}, 분위기: {mood}, 주제: {topic}. 이 조건에 맞는 감성적인 한국어 노래 가사를 써줘. [Verse], [Chorus] 구조를 갖춰서 작성해줘."
            
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": lyric_prompt}]
            )
            lyrics = completion.choices[0].message.content
            
            st.subheader("📝 생성된 가사")
            st.write(lyrics)

            # 2. 오디오 생성 단계 (OpenAI의 TTS 모델 사용)
            # 주의: 전문적인 음악 생성(배경음 포함)은 별도의 전용 API(Suno, Udio 등)가 필요할 수 있습니다.
            # 여기서는 가사를 읽어주는 형태의 음성 생성을 예시로 합니다.
            
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova", # 여성 음성 (alloy, echo, fable, onyx, nova, shimmer 선택 가능)
                input=lyrics[:4000] # TTS 글자수 제한 대응
            )
            
            # 오디오 출력
            st.subheader("🎧 생성된 노래(음성)")
            st.audio(response.content, format="audio/mp3")
            
            st.success("노래 생성이 완료되었습니다!")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# 하단 안내
st.divider()
st.info("Tip: 더 전문적인 음악 생성을 원하시면 Suno AI나 Udio API를 연동하는 것을 추천합니다.")
