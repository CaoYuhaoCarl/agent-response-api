# -*- coding: utf-8 -*- # Ensure UTF-8 encoding for wider character support

import streamlit as st
# Set page config as the very first Streamlit command
st.set_page_config(layout="wide", page_title="AI Dialogue Personalizer")

from openai import OpenAI, OpenAIError
import re
import os
import json
from dotenv import load_dotenv

# --- Load Environment Variables ---
try:
    load_dotenv()
    print("Attempted to load .env file.") # Add print statement for debugging
except Exception as e:
    print(f"Error loading .env file (this might be ignorable if not using .env): {e}")

# --- Configuration ---
# Attempt to initialize OpenAI client using environment variables.
API_KEY_VALID = False # Assume invalid until proven otherwise
OPENAI_CLIENT_ERROR_MESSAGE = None # Store specific error message

try:
    # The OpenAI client automatically uses OPENAI_API_KEY, HTTP_PROXY, HTTPS_PROXY
    # environment variables if they are set (thanks to load_dotenv or manual export).
    client = OpenAI()
    # Perform a simple test call to ensure the client is functional (optional but recommended)
    # Using a very cheap model and minimal request for validation
    client.models.list() # 移除 limit 参数
    API_KEY_VALID = True
    print("OpenAI client initialized and validated successfully.")
except OpenAIError as e:
    OPENAI_CLIENT_ERROR_MESSAGE = f"OpenAI API Error during initialization/validation: {e}. Please check your API Key, network connection, and proxy settings (if applicable) in your environment or .env file."
    print(OPENAI_CLIENT_ERROR_MESSAGE) # Log error to console
except Exception as e:
    OPENAI_CLIENT_ERROR_MESSAGE = f"An unexpected error occurred during OpenAI client initialization: {e}. Check library installation and environment."
    print(OPENAI_CLIENT_ERROR_MESSAGE) # Log error to console

# Display error in Streamlit UI if client initialization failed
if not API_KEY_VALID and OPENAI_CLIENT_ERROR_MESSAGE:
    st.error(OPENAI_CLIENT_ERROR_MESSAGE)


# --- Default Values ---
DEFAULT_DIALOGUE = """
A: Oh, excuse me! I'm so sorry, I wasn't looking where I was going when I stood up.
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
B: [Smiling] That's me. Well, this is quite a surprise! It's really great meeting you, [User's Name].
"""

# Default persona elements
DEFAULT_PERSONA_NAME = "Carl"
DEFAULT_PERSONA_TRAITS = "Enthusiastic, Friendly, Informal, Energetic" # More specific default
DEFAULT_ADDRESS_TERM = "[User's Name]" # Default to placeholder
DEFAULT_CATCHPHRASES = "Awesome!, Cool!, Wow!, Like," # Usually better to leave blank unless specific needed

# --- Trait Suggestions (Broad ideas, avoid strict stereotypes) ---
TRAIT_SUGGESTIONS = {
    "Male": [
        "Confident", "Analytical", "Reserved", "Witty", "Direct", "Calm", "Humorous",
        "Assertive", "Loyal", "Practical", "Adventurous", "Resourceful", "Sarcastic", "Protective"
    ],
    "Female": [
        "Empathetic", "Nurturing", "Expressive", "Intuitive", "Collaborative", "Warm", "Patient",
        "Detail-oriented", "Creative", "Supportive", "Communicative", "Insightful", "Resilient", "Organized"
    ],
    "Other/Prefer not to say": [
        "Curious", "Adaptable", "Logical", "Imaginative", "Independent", "Objective", "Diplomatic",
        "Methodical", "Philosophical", "Spontaneous", "Quiet", "Energetic", "Optimistic", "Pessimistic"
    ]
}

# --- Helper Functions ---

def parse_dialogue(text):
    """
    Parses dialogue text into a list of (speaker, line) tuples.
    Accepts A/U for User and B/M for AI, standardizing to A and B.
    """
    lines = text.strip().split('\n')
    parsed = []
    # Regex to capture speaker (A, B, U, or M, case-insensitive) and the rest of the line
    pattern = re.compile(r"^\s*(A|B|U|M)\s*:\s*(.*)", re.IGNORECASE)
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            speaker, content = match.groups()
            # Standardize speaker: A/U -> 'A' (User), B/M -> 'B' (AI)
            standard_speaker = 'A' if speaker.upper() in ['A', 'U'] else 'B'
            parsed.append((standard_speaker, content.strip()))
        elif line.strip():
             # Log skipped lines with line numbers for easier debugging
             print(f"Info: Skipping line {idx+1} (doesn't match Speaker: format): '{line.strip()}'")
    return parsed

# 定义工具函数：重写AI对话行
def rewrite_dialogue_line(original_line, persona, cefr_level):
    """根据指定的角色和CEFR级别重写对话行"""
    try:
        # 返回重写的对话行
        return {
            "rewritten_line": original_line,  # 这里实际上会被AI根据角色和语言级别重写
            "persona": persona,
            "cefr_level": cefr_level
        }
    except Exception as e:
        raise Exception(f"对话行重写失败: {str(e)}")

def generate_rewritten_line(original_line, persona, cefr_level, model="gpt-4o-mini"):
    """
    使用OpenAI Responses API和工具函数调用重写AI对话行。
    基于角色和CEFR级别，使用agent方式处理。
    """
    # 仅在OpenAI客户端初始化正确时继续
    if not API_KEY_VALID:
        # 返回清晰的错误消息以在对话中显示
        return "[ERROR: OpenAI Client not available - Check configuration]"

    # 定义重写对话行的工具
    tools = [{
        "type": "function",
        "function": {
            "name": "rewrite_dialogue_line",
            "description": "根据指定的AI角色和CEFR语言级别重写对话行",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_line": {
                        "type": "string",
                        "description": "需要重写的原始对话行"
                    },
                    "persona": {
                        "type": "object",
                        "description": "AI角色的特征定义",
                        "properties": {
                            "name": {"type": "string", "description": "AI角色的名字"},
                            "traits": {"type": "string", "description": "AI角色的性格特征"},
                            "address_term": {"type": "string", "description": "AI如何称呼用户"},
                            "catchphrases": {"type": "string", "description": "AI常用的口头禅"}
                        }
                    },
                    "cefr_level": {
                        "type": "string",
                        "description": "目标CEFR语言级别(A1, A2, B1, B2, C1, C2)"
                    }
                },
                "required": ["original_line", "persona", "cefr_level"]
            }
        }
    }]

    # 系统提示，定义AI的角色和任务
    system_prompt = f"""You are an expert dialogue writer specializing in adapting scripts for language learning apps.
Your specific task is to rewrite a single line of dialogue for an AI character (referred to as 'B') interacting with a human user ('A').
The goal is to make the AI character's line fit a specific persona and language (CEFR) level, while maintaining the conversational flow and core meaning.
Follow these instructions for the rewrite:
1. Analyze the core meaning of the original line.
2. Rewrite it to sound like it's spoken by the character with the specified personality traits.
3. Adjust vocabulary and grammar to match the target CEFR level.
4. Maintain the same function in the dialogue as the original.
5. Start the rewritten line with a concise action cue in square brackets (e.g., [Smiling]).
6. Use the address terms and catchphrases naturally if appropriate.
7. Return only the final rewritten line, starting with the action cue.

Interpret CEFR levels as follows:
- A1: Very basic words and simple Subject-Verb-Object sentences.
- A2: Common everyday vocabulary with simple present/past tenses.
- B1: Good vocabulary range for familiar topics with main tenses used correctly.
- B2: Wider vocabulary with some idioms and complex sentence structures.
- C1: Broad vocabulary including nuanced language and complex structures.
- C2: Very wide vocabulary with mastery of idioms and near-native grammar.
"""

    # 用户提示，提供要重写的原始行
    user_prompt = f"Rewrite this dialogue line: \"{original_line}\""

    try:
        # 调用OpenAI Responses API
        response = client.responses.create(
            model=model,
            temperature=0.6,
            system=system_prompt,
            input=user_prompt,
            tools=tools
        )
        
        # 检查是否有工具调用
        if response.output and len(response.output) > 0 and hasattr(response.output[0], 'type') and response.output[0].type == 'tool_call':
            # 处理工具调用
            tool_call = response.output[0]
            
            if tool_call.name == "rewrite_dialogue_line":
                # 解析参数
                args = json.loads(tool_call.arguments)
                
                # 在实际应用中，这里会调用rewrite_dialogue_line函数
                # 这里我们直接处理AI生成的回答，添加动作提示
                
                # 构造工具调用结果
                rewritten_line = f"[AI Action] {args.get('original_line', original_line)}"
                
                # 传递工具调用结果回OpenAI
                dialogue_input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    tool_call,  # 工具调用请求
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps({
                            "rewritten_line": original_line, 
                            "persona": persona,
                            "cefr_level": cefr_level
                        })
                    }
                ]
                
                # 再次调用模型获取最终结果
                final_response = client.responses.create(
                    model=model,
                    temperature=0.7,
                    input=dialogue_input_messages
                )
                
                # 获取最终文本响应
                if final_response.output_text:
                    rewritten_line = final_response.output_text.strip()
                    
                    # 检查是否是合适的格式（以动作提示开头）
                    if not re.match(r"^\[.*?\]", rewritten_line):
                        # 如果没有动作提示，添加一个通用的
                        print(f"Warning: Response doesn't start with action cue: '{rewritten_line}'")
                        rewritten_line = f"[Action?] {rewritten_line}"
                    
                    return rewritten_line
                else:
                    # 如果无法获取有效的最终响应，返回处理过的原始行
                    print(f"Warning: Could not get valid final response for line '{original_line}'.")
                    return f"[Action?] {original_line}"
            else:
                # 未知的工具调用
                print(f"Warning: Unknown tool call '{tool_call.name}' for line '{original_line}'.")
                return f"[ERROR: Unknown Tool Call - {tool_call.name}]"
        else:
            # 没有工具调用，直接使用文本输出
            if response.output_text:
                rewritten_line = response.output_text.strip()
                
                # 检查是否是合适的格式（以动作提示开头）
                if not re.match(r"^\[.*?\]", rewritten_line):
                    # 如果没有动作提示，添加一个通用的
                    print(f"Warning: Response doesn't start with action cue: '{rewritten_line}'")
                    rewritten_line = f"[Action?] {rewritten_line}"
                
                return rewritten_line
            else:
                # 无有效响应
                print(f"Warning: No valid response for line '{original_line}'.")
                return f"[ERROR: No Valid Response]"

    except OpenAIError as e:
        # 记录具体错误
        print(f"Error: OpenAI API call failed for line '{original_line}'. Details: {e}")
        # 在UI中提供具体反馈
        if "authentication" in str(e).lower():
             return "[ERROR: API Authentication Failed - Check Key]"
        elif "rate limit" in str(e).lower():
             return "[ERROR: API Rate Limit Exceeded - Try Again Later]"
        elif "connection" in str(e).lower():
             return "[ERROR: API Connection Failed - Check Network/Proxy]"
        else:
             return f"[ERROR: OpenAI API Error - {type(e).__name__}]" # 一般API错误
    except Exception as e:
        # 记录意外错误
        print(f"Error: Unexpected error during generation for line '{original_line}'. Details: {e}")
        return f"[ERROR: Unexpected Script Error - {type(e).__name__}]"


# --- Streamlit UI ---
st.title("🎭 Carl's AI Dialogue Personalizer")

# Display client initialization error prominently if it occurred
if not API_KEY_VALID and OPENAI_CLIENT_ERROR_MESSAGE:
    # The error is already displayed via st.error during initialization check
    st.warning("OpenAI Client failed to initialize. Generation is disabled. Please check the error message above and your setup.")


st.info("Paste your dialogue script (using 'A:'/'U:' for User and 'B:'/'M:' for AI), define the AI's persona, select a CEFR level, and generate a rewritten script.")

col1, col2 = st.columns(2)

with col1:
    st.header("Inputs")

    # Dialogue Input
    original_dialogue_text = st.text_area(
        "Original Dialogue Script",
        height=250,
        value=DEFAULT_DIALOGUE,
        key="dialogue_input",
        help="Paste the script here. Use 'A:' or 'U:' for the User's lines and 'B:' or 'M:' for the AI's lines. Lines without this format will be skipped."
    )

    # User Gender Input for Suggestions (optional)
    st.subheader("Context (Optional)")
    user_gender = st.radio(
        "User's Gender (for AI trait *ideas*):",
        options=["Male", "Female", "Other/Prefer not to say"],
        index=2, # Default to 'Other'
        key="user_gender",
        horizontal=True,
        help="Selecting a user gender provides AI trait suggestions below. It does NOT directly influence the AI generation itself."
    )

    # AI Character ('B'/'M') Persona Definition
    st.subheader("AI Character Persona ('B'/'M')")

    persona_name = st.text_input(
        "AI Character Name",
        value=DEFAULT_PERSONA_NAME,
        key="persona_name"
        )

    # Display Trait Suggestions based on User Gender selection
    selected_gender_for_suggestions = st.session_state.user_gender
    trait_suggestions_list = TRAIT_SUGGESTIONS.get(selected_gender_for_suggestions, [])

    with st.expander("View AI Trait Suggestions (Ideas Only)"):
        st.caption(f"These are just brainstorm ideas based on a '{selected_gender_for_suggestions}' user context. Define the *actual* AI traits in the text box below.")
        if trait_suggestions_list:
            num_cols = 3
            cols = st.columns(num_cols)
            for i, trait in enumerate(trait_suggestions_list):
                cols[i % num_cols].markdown(f"- {trait}") # Use markdown for bullet points
        else:
            st.write("No specific suggestions provided for this selection.")

    # Actual Trait Input Area (This is what gets used)
    persona_traits = st.text_area(
        "AI Personality Traits (comma-separated - Define the AI here)",
        value=DEFAULT_PERSONA_TRAITS,
        key="persona_traits",
        help="Enter the desired traits for the AI character ('B'/'M'). Separate traits with commas. This field defines the persona used in generation."
        )

    address_term = st.text_input(
        "AI's Way of Addressing User (e.g., [User's Name], buddy)",
        value=DEFAULT_ADDRESS_TERM,
        key="address_term",
        help="How the AI refers to the user. Use '[User's Name]' as a placeholder for the actual user name if needed."
        )

    catchphrases = st.text_input(
        "AI's Catchphrase(s) (comma-separated, optional)",
        value=DEFAULT_CATCHPHRASES,
        key="catchphrases",
        help="Short phrases the AI might use occasionally (e.g., Right!, Indeed, Awesome!). Use sparingly."
        )

    # Store the user-defined persona details in a dictionary
    persona_details = {
        "name": persona_name,
        "traits": persona_traits, # Takes value from the text_area
        "address_term": address_term,
        "catchphrases": catchphrases,
    }

    # CEFR Level and Model Selection
    cefr_level = st.selectbox(
        "Target CEFR Level for AI ('B'/'M')",
        options=["A1", "A2", "B1", "B2", "C1", "C2"],
        index=2, # Default to B1
        key="cefr_level"
    )

    model_choice = st.selectbox(
        "OpenAI Model",
        # Add or remove models based on availability and preference
        options=["gpt-4o-mini", "gpt-4o"],
        index=0, # Default to gpt-4o-mini
        key="model_choice",
        help="gpt-4o generally provides higher quality but is more expensive than gpt-4o-mini."
    )

    # Disable button if OpenAI client isn't valid
    generate_button = st.button(
        "✨ Generate Personalized Dialogue",
        disabled=(not API_KEY_VALID),
        use_container_width=True,
        key="generate_button"
        )

with col2:
    st.header("Generated Dialogue")
    # Use a placeholder for the output area so it can be updated cleanly
    output_area_placeholder = st.empty()
    # Initialize with placeholder text
    output_area_placeholder.text_area(
        "Rewritten Dialogue",
        value="Click 'Generate' to see the result.",
        height=600, # Increased height for better visibility
        key="output_text_area"
        )


# --- Processing Logic ---
if st.session_state.get("generate_button"): # Check if button was clicked via session state
    if not API_KEY_VALID:
        # Error already shown, but prevent processing
        st.warning("Generation disabled because OpenAI Client is not ready.")
    elif not st.session_state.dialogue_input: # Check if input dialogue is empty
        st.warning("Please paste the original dialogue script into the input area.")
    else:
        # Get potentially updated values from session state (though direct variable access often works too)
        current_dialogue = st.session_state.dialogue_input
        current_persona = {
            "name": st.session_state.persona_name,
            "traits": st.session_state.persona_traits,
            "address_term": st.session_state.address_term,
            "catchphrases": st.session_state.catchphrases,
        }
        current_cefr = st.session_state.cefr_level
        current_model = st.session_state.model_choice

        parsed_original = parse_dialogue(current_dialogue)

        if not parsed_original:
            st.error("Could not parse the dialogue. Please ensure lines start with 'A:', 'U:', 'B:', or 'M:' followed by a colon.")
        else:
            ai_lines_exist = any(speaker == 'B' for speaker, _ in parsed_original)

            if not ai_lines_exist:
                 st.warning("The script does not contain any lines marked for the AI ('B:' or 'M:'). The original script is shown.")
                 # Display original script in output if no AI lines to change
                 output_area_placeholder.text_area(
                     "Rewritten Dialogue",
                     value=current_dialogue, # Show original
                     height=600,
                     key="output_text_area_no_change" # Use different key if needed
                     )
            else:
                new_dialogue_lines = []
                total_b_lines = sum(1 for speaker, _ in parsed_original if speaker == 'B')
                # Initialize progress bar only if there are lines to process
                progress_bar = st.progress(0)
                processed_b_lines = 0
                errors_encountered = False # Flag to track if any line failed

                # Show spinner during processing
                with st.spinner(f"Generating {current_cefr} dialogue for AI '{current_persona['name']}' using {current_model}..."):
                    for i, (speaker, line_content) in enumerate(parsed_original):
                        # Standardize output speaker labels to A (User) and B (AI)
                        if speaker == 'A':
                            new_dialogue_lines.append(f"A: {line_content}")
                        elif speaker == 'B':
                            # Generate the rewritten line for the AI
                            rewritten_b_line = generate_rewritten_line(
                                line_content,
                                current_persona,
                                current_cefr,
                                model=current_model
                            )
                            # Check if generation failed for this line
                            if rewritten_b_line.startswith("[ERROR:"):
                                errors_encountered = True
                                print(f"Error recorded for line {i+1}: {rewritten_b_line}") # Log error to console
                            
                            # Append the line (even if error, to maintain dialogue structure)
                            new_dialogue_lines.append(f"B: {rewritten_b_line}")
                            
                            # Update progress
                            processed_b_lines += 1
                            progress_bar.progress(processed_b_lines / total_b_lines)
                    
                    # Combine all lines into the rewritten script
                    rewritten_dialogue = "\n".join(new_dialogue_lines)
                    
                    # Display the result, with an error notice if any line failed
                    if errors_encountered:
                        st.error("⚠️ Some lines encountered errors during generation (marked with [ERROR:...]).")
                    
                    # Update the output area with the result
                    output_area_placeholder.text_area(
                        "Rewritten Dialogue",
                        value=rewritten_dialogue,
                        height=600,
                        key="output_text_area_generated"
                        )
                    
                    # Reset progress bar
                    progress_bar.progress(0.0)
                
                # Success notice if everything worked
                if not errors_encountered:
                    st.success(f"✅ Successfully rewrote {processed_b_lines} lines of dialogue in {current_persona['name']}'s voice at CEFR {current_cefr} level.")