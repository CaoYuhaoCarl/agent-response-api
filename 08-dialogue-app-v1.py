import streamlit as st
from openai import OpenAI, OpenAIError
import re
import os
from dotenv import load_dotenv # <--- Import load_dotenv

# --- Load Environment Variables ---
# Load variables from .env file into the environment BEFORE initializing OpenAI client
# This will automatically find the .env file in the current directory or parent directories
load_dotenv() # <--- Add this line EARLY in your script

# --- Configuration ---
# Attempt to initialize OpenAI client.
# It will now automatically pick up OPENAI_API_KEY and proxy settings
# from the environment variables loaded by load_dotenv().
try:
    # No need to pass api_key explicitly if OPENAI_API_KEY is set in .env
    # The library also automatically checks HTTP_PROXY/HTTPS_PROXY env vars
    client = OpenAI()
    API_KEY_VALID = True # Assume valid if client initializes without error
    # You might add a simple test call here if needed to be sure key/proxy work
except OpenAIError as e:
    # Handle missing API key or other initialization errors gracefully
    st.error(f"Failed to initialize OpenAI Client. Check API Key and network/proxy settings. Error: {e}")
    API_KEY_VALID = False
except Exception as e:
    # Catch potential general errors during initialization
    st.error(f"An unexpected error occurred during OpenAI client initialization: {e}")
    API_KEY_VALID = False


# --- Default Values & Examples ---
DEFAULT_DIALOGUE = """A: Oh, excuse me! I'm so sorry, I wasn't looking where I was going when I stood up.
B: No problem at all. It happens! Are you okay?
A: Yes, I'm fine, thanks. Hey, is that 'The Hidden City'? I've been looking for that specific version everywhere!
B: Oh, this? Yes, it's a great read. I found it in a small bookshop last week. Are you a fan of the author?
A: Definitely! I love his writing style. Actually, I enjoy reading, travel, and music quite a lot. It seems we might have similar interests.
B: It seems so! I'm Ben, by the way. It's nice talking to you. This cafe is quite busy, but there's a free table over there if you want to sit.
A: I'd love that, thanks! I'm [User's Name]. It's nice to meet you, Ben. (They move to the table) This is such a nice coincidence, meeting like this.
B: It really is. Speaking of books, I'm part of a small book club. We're discussing 'The Hidden City' next week, actually.
A: Really? That sounds fantastic! I've wanted to join a book club for ages.
B: Well, would you like to join us for the meeting? You'd be very welcome.
A: That's so kind of you, thank you! Yes, please. How can I get the details? Maybe we could exchange contact information?
B: Good idea. My number is 555-2626, and the name's Ben Carter. What's yours?
A: It's 111-1234. Thanks! Wait... Ben Carter? The Ben Carter, the writer? I follow your blog! I can't believe it!
B: (Smiling) That's me. Well, this is quite a surprise! It's really great meeting you, [User's Name]."""

DEFAULT_PERSONA_NAME = "Carl"
DEFAULT_PERSONA_TRAITS = "Enthusiastic, Friendly, Informal, Energetic"
DEFAULT_ADDRESS_TERM = "[User's Name], Pal, Mate"
DEFAULT_CATCHPHRASES = "Awesome!, Cool!, Wow!, Like,"

# --- Helper Functions ---

def parse_dialogue(text):
    """Parses dialogue text into a list of (speaker, line) tuples."""
    lines = text.strip().split('\n')
    parsed = []
    # Regex to capture speaker (A or B) and the rest of the line
    pattern = re.compile(r"^\s*(A|B)\s*:\s*(.*)")
    for line in lines:
        match = pattern.match(line)
        if match:
            speaker, content = match.groups()
            parsed.append((speaker.strip(), content.strip()))
        elif line.strip(): # Add non-matching lines if they aren't empty (e.g., stage directions)
             print(f"Skipping line (doesn't match A:/B: format): {line}") # Optional: log skipped lines
    return parsed

def generate_rewritten_line(original_line, persona, cefr_level, model="gpt-4o-mini"):
    """Calls OpenAI API to rewrite a single line based on persona and CEFR level."""
    if not API_KEY_VALID:
        return "[ERROR: OpenAI Client not initialized]"

    system_prompt = """You are an expert dialogue writer adapting scripts for different character personas and language levels.
Your task is to rewrite the following dialogue line spoken by character 'B'.
You must follow all instructions precisely."""

    user_prompt = f"""
Original Line (Character B): "{original_line}"

Character Persona for 'B':
- Name: {persona['name']}
- Personality: {persona['traits']}
- Address Term for User (use naturally): {persona['address_term']}
- Catchphrases (use naturally): {persona['catchphrases']}

Target Language Level: CEFR {cefr_level}

Instructions:
1. Rewrite the original line to perfectly match the specified character persona and target CEFR level ({cefr_level}).
2. Maintain the original core meaning and function of the line within the conversation's flow. DO NOT change the essential information conveyed.
3. Start the rewritten line *immediately* with a concise non-verbal action or expression cue in square brackets (e.g., [Smiling], [Nodding], [Looking surprised]).
4. Use the specified address term and catchphrases naturally where appropriate, fitting the character and context. Do not force them unnaturally.
5. Ensure the vocabulary, grammar, and sentence structure complexity precisely match the target CEFR level:
    - A1: Very basic words, simple short sentences.
    - A2: Basic everyday words, simple compound sentences.
    - B1: Good everyday vocabulary, clear opinions/feelings, common tenses.
    - B2: Wider vocabulary (some idioms), more complex sentences (clauses, conditionals).
    - C1: Broad vocabulary (nuance, idioms), complex/varied sentences, fluent expression.
    - C2: Very wide/precise vocabulary, highly complex/nuanced sentences, near-native fluency.
6. Output *only* the rewritten line including the action cue. Do not add *any* explanations, greetings, speaker labels (like "B:"), or extra text before or after the rewritten line. Just the bracketed action and the spoken text.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # Adjust for creativity vs consistency
            max_tokens=200
        )
        rewritten_line = response.choices[0].message.content.strip()
        if rewritten_line.startswith('[') and ']' in rewritten_line:
             return rewritten_line
        else:
             st.warning(f"Model response format might be incorrect for line: '{original_line}'. Response: '{rewritten_line}'")
             return f"[Action?] {rewritten_line}" # Fallback

    except OpenAIError as e:
        # Catch specific OpenAI errors, including connection errors which might happen here
        st.error(f"OpenAI API call failed: {e}")
        # If it's a connection error, it likely means the proxy/network is still an issue
        if "Connection error" in str(e):
             st.warning("Connection Error persists. Please double-check your proxy settings in the .env file and ensure the proxy server is running and accessible.")
        return f"[ERROR: API Call Failed for line: {original_line}]"
    except Exception as e:
        st.error(f"An unexpected error occurred during generation: {e}")
        return f"[ERROR: Unexpected error for line: {original_line}]"


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎭 Carl's AI Dialogue Personalizer")

st.info("Paste your dialogue script (using 'U:' and 'M:' format), define AI character 'B's persona, select a CEFR level, and generate a rewritten script.")

col1, col2 = st.columns(2)

with col1:
    st.header("Inputs")
    original_dialogue_text = st.text_area(
        "Original Dialogue Script (User=U, AI=M)",
        height=300,
        value=DEFAULT_DIALOGUE,
        help="Use the format 'U: Dialogue...' or 'M: Dialogue...' for each line."
    )

    st.subheader("AI Character 'B' Persona")
    persona_name = st.text_input("Character Name", value=DEFAULT_PERSONA_NAME)
    persona_traits = st.text_area("Personality Traits (comma-separated)", value=DEFAULT_PERSONA_TRAITS)
    address_term = st.text_input("Address Term(s) for User", value=DEFAULT_ADDRESS_TERM)
    catchphrases = st.text_input("Catchphrase(s) (comma-separated)", value=DEFAULT_CATCHPHRASES)

    persona = {
        "name": persona_name,
        "traits": persona_traits,
        "address_term": address_term,
        "catchphrases": catchphrases,
    }

    cefr_level = st.selectbox(
        "Target CEFR Level for AI ('B')",
        options=["A1", "A2", "B1", "B2", "C1", "C2"],
        index=2 # Default to B1
    )

    model_choice = st.selectbox(
        "OpenAI Model",
        options=["gpt-4o-mini", "gpt-4o"], # Add more models if needed
        index=0
    )

    generate_button = st.button("✨ Generate Personalized Dialogue", disabled=(not API_KEY_VALID))

with col2:
    st.header("Generated Dialogue")
    output_area = st.empty() # Placeholder for the output text_area
    output_area.text_area("Rewritten Dialogue", value="Click 'Generate' to see the result.", height=500, key="output")


# --- Processing Logic ---
if generate_button and API_KEY_VALID:
    if not original_dialogue_text:
        st.warning("Please paste the original dialogue script.")
    else:
        parsed_original = parse_dialogue(original_dialogue_text)
        if not parsed_original:
            st.error("Could not parse the dialogue. Please ensure it uses the 'A: ...' / 'B: ...' format.")
        else:
            new_dialogue_lines = []
            total_b_lines = sum(1 for speaker, _ in parsed_original if speaker == 'B')
            progress_bar = st.progress(0)
            processed_b_lines = 0

            with st.spinner(f"Generating {cefr_level} dialogue for '{persona['name']}'..."):
                for i, (speaker, line_content) in enumerate(parsed_original):
                    if speaker == 'A':
                        new_dialogue_lines.append(f"A: {line_content}")
                    elif speaker == 'B':
                        # Add status update within the spinner context if desired
                        # st.write(f"Rewriting B's line {processed_b_lines + 1}/{total_b_lines}...")
                        rewritten_b_line = generate_rewritten_line(line_content, persona, cefr_level, model=model_choice)
                        new_dialogue_lines.append(f"B: {rewritten_b_line}")
                        processed_b_lines += 1
                        progress_bar.progress(processed_b_lines / total_b_lines)
                    # Handle potential non-dialogue lines if parse_dialogue was modified to include them
                    # else:
                    #    new_dialogue_lines.append(line_content) # Append other lines directly

            final_dialogue = "\n".join(new_dialogue_lines)
            # Update the text_area content using its key
            output_area.text_area("Rewritten Dialogue", value=final_dialogue, height=500, key="output_updated") # Use diff key to force update
            st.success("Dialogue generation complete!")
            progress_bar.empty() # Remove progress bar after completion

elif generate_button and not API_KEY_VALID:
    st.error("Cannot generate dialogue. Please ensure your OpenAI API Key is correctly configured.")