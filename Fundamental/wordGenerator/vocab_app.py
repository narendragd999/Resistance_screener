import streamlit as st
import requests
import random

# Fallback dictionary for common words (English to Hindi)
fallback_dictionary = {
    "exemplary": "उत्कृष्ट",
    "forest": "जंगल",
    "river": "नदी",
    "mountain": "पहाड़",
    "ocean": "महासागर",
    "tree": "पेड़",
    "technology": "प्रौद्योगिकी",
    "circuit": "सर्किट",
    "byte": "बाइट",
    "pizza": "पिज़्ज़ा",
    "sushi": "सुशी",
    "tiger": "बाघ",
    "elephant": "हाथी",
    "happy": "खुश",
    "bright": "उज्ज्वल",
    "swift": "तेज",
    "calm": "शांत",
    "vivid": "जीवंत",
    "bold": "नन्हा",
    "quiet": "चुप",
    "strong": "मजबूत",
    "flower": "फूल",
    "sky": "आकाश",
    "software": "सॉफ्टवेयर",
    "bread": "रोटी",
    "wolf": "भेड़िया",
    "dog": "कुत्ता",
    "cat": "बिल्ली",
    "apple": "सेब",
    "book": "किताब",
    "sun": "सूरज",
    "moon": "चाँद",
    "star": "तारा",
    "cloud": "बादल"
}

# Streamlit app configuration
st.set_page_config(page_title="Kids Vocabulary Builder", page_icon="🌟", layout="centered")

# Custom CSS for sophisticated, child-friendly UI
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(to bottom, #e0f7ff, #ffe4e1); 
    }
    .stButton>button {
        background: linear-gradient(to right, #ff6347, #ff4500);
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 12px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border: 2px solid #1e90ff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        background-color: #f9f9f9;
    }
    .stSelectbox>div {
        border: 2px solid #1e90ff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        background-color: #f9f9f9;
    }
    .word-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white;
    }
    .word-table th, .word-table td {
        padding: 16px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    .word-table th {
        background: linear-gradient(to right, #98fb98, #90ee90);
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    .word-table td {
        font-size: 16px;
        color: #444;
    }
    .word-table tr:hover {
        background-color: #f0f8ff;
        transition: background-color 0.3s;
    }
    .word-table tr:last-child td {
        border-bottom: none;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for progress
if 'level_progress' not in st.session_state:
    st.session_state.level_progress = {
        "Beginner": False,
        "Intermediate": False,
        "Advanced": False
    }
if 'current_level' not in st.session_state:
    st.session_state.current_level = "Beginner"

# Functions
def get_random_words(topic, num_words):
    try:
        url = f"https://api.datamuse.com/words?rel_jjb={topic}&max={num_words}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [item['word'] for item in data]
        return []
    except Exception as e:
        st.error(f"Error fetching words: {e}")
        return []

def get_hindi_translation(word):
    if word in fallback_dictionary:
        return fallback_dictionary[word]
    try:
        url = f"https://api.mymemory.translated.net/get?q={word}&langpair=en|hi"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            translation = data['responseData']['translatedText']
            return translation if translation else fallback_dictionary.get(word, "Translation unavailable")
        return fallback_dictionary.get(word, "Translation unavailable")
    except Exception as e:
        st.error(f"Error translating {word}: {e}")
        return fallback_dictionary.get(word, "Translation unavailable")

def get_word_definition(word):
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data:
                for meaning in entry.get('meanings', []):
                    for definition in meaning.get('definitions', []):
                        return definition.get('definition', "Definition unavailable")
            return "Definition unavailable"
        return "Definition unavailable"
    except Exception as e:
        st.error(f"Error fetching definition for {word}: {e}")
        return "Definition unavailable"

# Main app
st.title("🌟 Kids Vocabulary Builder 🌟")
st.markdown("Learn new words with fun topics! Choose a level, pick a topic, and grow your vocabulary! 😊")

# Level selection
st.subheader("Choose Your Level")
level = st.selectbox("Select Level", ["Beginner", "Intermediate", "Advanced"], index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.current_level))

# Update current level
st.session_state.current_level = level

# Topic input
st.subheader("Pick a Topic")
topic = st.text_input("Enter any topic (e.g., animals, food, nature)", value="animals")

# Number of words input
st.subheader("How Many Words?")
num_words = st.number_input("Enter number of words (1–15)", min_value=1, max_value=15, value=3 if level == "Beginner" else 6 if level == "Intermediate" else 11)

# Generate words button
if st.button("Generate Words 🚀"):
    if not topic.strip():
        st.error("Please enter a topic!")
    elif num_words < 1 or num_words > 15:
        st.error("Please enter a number between 1 and 15!")
    else:
        with st.spinner("Finding fun words..."):
            words = get_random_words(topic, num_words)
            if not words:
                st.error("No words found for this topic. Try another one!")
            else:
                word_data = []
                for word in words:
                    hindi = get_hindi_translation(word)
                    definition = get_word_definition(word)
                    word_data.append({
                        "English Word": word,
                        "Hindi Meaning": hindi,
                        "Definition": definition
                    })
                
                # Display words in a table
                st.subheader(f"New Words for {topic.capitalize()} ({level})")
                st.markdown("<table class='word-table'><tr><th>English Word</th><th>Hindi Meaning</th><th>Definition</th></tr>", unsafe_allow_html=True)
                for data in word_data:
                    st.markdown(
                        f"<tr><td>{data['English Word']}</td><td>{data['Hindi Meaning']}</td><td>{data['Definition']}</td></tr>",
                        unsafe_allow_html=True
                    )
                st.markdown("</table>", unsafe_allow_html=True)

                # Update progress
                st.session_state.level_progress[level] = True
                st.success(f"Great job! You've completed the {level} level with {num_words} words! 🎉")

# Display progress
st.subheader("Your Progress")
progress = sum(1 for completed in st.session_state.level_progress.values() if completed)
st.markdown(f"Levels Completed: {progress}/3 ⭐")
st.progress(progress / 3.0)

# Unlock next level
if st.session_state.level_progress["Beginner"] and not st.session_state.level_progress["Intermediate"]:
    st.write("You've unlocked Intermediate! Try it now! 😄")
elif st.session_state.level_progress["Intermediate"] and not st.session_state.level_progress["Advanced"]:
    st.write("Amazing! You've unlocked Advanced! Ready for a challenge? 🚀")
elif st.session_state.level_progress["Advanced"]:
    st.balloons()
    st.markdown("You're a vocabulary superstar! All levels complete! 🌟 Try new topics to learn more!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for kids to learn and grow! Keep exploring new words!")