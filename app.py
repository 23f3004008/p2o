import os
import json
import asyncio
import time
import datetime
import re
import base64
import io
import tempfile
import zipfile
import tarfile
import subprocess
import uuid
import httpx
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types
from firecrawl import AsyncFirecrawlApp

# Load API keys from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Helper Functions ---

async def send_discord_webhook(message: str, request_id: str = None):
    """Sends a message to the Discord webhook asynchronously."""
    if not DISCORD_WEBHOOK_URL:
        return

    async with httpx.AsyncClient() as client:
        try:
            log_prefix = f"[`{request_id}`] " if request_id else ""
            payload = {
                "content": f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log_prefix}{message}"
            }
            await client.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            print_with_timestamp(f"Warning: Failed to send Discord webhook: {e}", request_id=request_id)

def print_with_timestamp(message, **kwargs):
    """Prints a message with a timestamp and optional request_id."""
    request_id = kwargs.pop('request_id', None)
    log_prefix = f"[{request_id}] " if request_id else ""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {log_prefix}{message}", **kwargs)

def get_content_type_for_image(filename):
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "gif":
        return "image/gif"
    else:
        return "application/octet-stream"

def extract_code_from_markdown(text):
    # More robust regex to find python code blocks, handling 'py' and varied whitespace.
    code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)

    if code_blocks:
        # Join all non-empty code blocks found.
        return "\n\n".join(block.strip() for block in code_blocks if block.strip())

    # If markdown fences are present but weren't parsed, return empty to avoid errors.
    if '```' in text:
        return ''

    # Otherwise, assume the whole text is code.
    return text

def execute_code(code: str, temp_dir: str, loop=None, request_id=None) -> tuple[str | None, str | None]:
    """
    Executes Python code and returns a tuple of (stdout, stderr).
    """
    def log_message(msg):
        if loop:
            asyncio.run_coroutine_threadsafe(send_discord_webhook(msg, request_id=request_id), loop)

    if not code:
        return None, "No code was generated to execute."
        
    script_name = "generated_script.py"
    script_path = os.path.join(temp_dir, script_name)
    
    print_with_timestamp(f"Creating new script at: {script_path}", request_id=request_id)

    script_content = "# -*- coding: utf-8 -*--\n" + code
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        print_with_timestamp(f"Executing code in: {temp_dir}", request_id=request_id)
        log_message(f"Executing generated code in `{temp_dir}`.")
        result = subprocess.run(
            ['python', script_name],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print_with_timestamp(f"Error: Code execution failed with stderr:\n{result.stderr}", request_id=request_id)
            log_message(f"Error: Code execution failed. Stderr: ```{result.stderr[:1800]}```")
            return None, result.stderr
        
        print_with_timestamp(f"Success: Code execution successful. Output:\n{result.stdout}", request_id=request_id)
        log_message(f"Success: Code execution successful. Output: ```json\n{result.stdout[:1800]}```")
        return result.stdout, None

    except subprocess.TimeoutExpired:
        print_with_timestamp("Error: Code execution timed out.", request_id=request_id)
        log_message("Error: Code execution timed out after 60 seconds.")
        return None, "Execution timed out"
    except Exception as e:
        print_with_timestamp(f"Error: An unexpected error occurred during execution: {e}", request_id=request_id)
        log_message(f"Error: An unexpected error occurred during code execution: {e}")
        return None, str(e)


class FileData:
    def __init__(self, name, content, content_type, is_image=False, is_text=False):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.is_image = is_image
        self.is_text = is_text

# --- Core Gemini Logic ---

def extract_prompt_metadata(user_input: str, loop=None, request_id=None) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.5-flash-lite"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.1,

        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["questions", "url", "data_file", "image"],
            properties={
                "questions": genai.types.Schema(type=genai.types.Type.ARRAY, items=genai.types.Schema(type=genai.types.Type.STRING)),
                "url": genai.types.Schema(type=genai.types.Type.STRING),
                "data_file": genai.types.Schema(type=genai.types.Type.STRING),
                "image": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
        system_instruction = [
    types.Part.from_text(text="""
Extract and return the following information strictly as JSON:

- "questions": A list of all questions explicitly mentioned in the user prompt.
- "url": A list of web URLs to scrape from, only if clearly and explicitly stated in the prompt.
- "data_file": The name(s) of any data file(s) if explicitly mentioned or attached (e.g. .csv, .xlsx, .json, .zip, .pdf, etc).
- "image": The name(s) of any image file(s) if explicitly mentioned or attached (e.g. .png, .jpg, .jpeg, .svg, .gif).

Rules:
1. Do not solve or attempt to answer the questions.
2. Do not infer or assume any information that is not directly and explicitly stated.
3. If a field is not present in the prompt, omit it entirely from the JSON — do not include blank or null fields.
4. Output ONLY valid JSON. Do not include extra commentary or text outside the JSON.
5. Include URLs ONLY if:
   - They are actual web addresses (starting with http://, https://, or www.)
   - AND the user prompt clearly asks to scrape from them.
6. Do NOT treat file names (e.g. .csv, .pdf, .png) as URLs — instead, classify them under "data_file" or "image" as appropriate.
7. Think carefully before including multiple URLs — only include those clearly necessary for scraping.

""")
]
    )
    output = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        output += chunk.text
    print_with_timestamp("\nGemini Flash-Lite Output (Structured):", request_id=request_id)
    print_with_timestamp(output, request_id=request_id)
    if loop:
        asyncio.run_coroutine_threadsafe(send_discord_webhook(f"Gemini Flash message response: ```json\n{output}```", request_id=request_id), loop)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print_with_timestamp("Error: Failed to parse JSON from Gemini response.", request_id=request_id)
        if loop:
            asyncio.run_coroutine_threadsafe(send_discord_webhook("Error: Failed to parse JSON from Gemini flash response.", request_id=request_id), loop)
        return {}


async def scrape_url_to_markdown(url: str, request_id=None) -> str:
    if not url:
        return ""
    print_with_timestamp(f"Scraping URL: {url}", request_id=request_id)
    await send_discord_webhook(f"Firecrawl: Starting Firecrawl scrape for URL: {url}", request_id=request_id)
    app = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
    try:
        response = await app.scrape_url(url=url, formats=['markdown'], only_main_content=True, parse_pdf=True, max_age=14400000)
        markdown = response.markdown
        print_with_timestamp("\nFirecrawl Markdown Output (preview):", request_id=request_id)
        print_with_timestamp(markdown[:1000], request_id=request_id)
        await send_discord_webhook(f"Success: Firecrawl scrape successful for URL: {url}", request_id=request_id)
        return markdown
    except Exception as e:
        print_with_timestamp(f"Error: Firecrawl scraping failed: {e}", request_id=request_id)
        await send_discord_webhook(f"Error: Firecrawl scrape failed for URL: {url}. Error: {e}", request_id=request_id)
        return ""

def generate_answer_with_context(original_prompt: str, context_markdown: str, image_files: list, start_time: float, loop=None, request_id=None):
    def log_message(msg):
        if loop:
            asyncio.run_coroutine_threadsafe(send_discord_webhook(msg, request_id=request_id), loop)

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.5-flash"
    
    contents = [
        f"User Prompt:\n{original_prompt}",
        f"Additional Context (Scraped Content and File Previews):\n{context_markdown}"
    ]

    if image_files:
        model = "gemini-2.5-flash"
        for image_file in image_files:
            contents.append(types.Part.from_bytes(
                data=image_file['content'],
                mime_type=image_file['content_type']
            ))
    
    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=128),
        tools=tools,
        system_instruction=[
            types.Part.from_text(text="""
Your primary role is to generate a single, fully functional, and executable Python script.
**Critical Instructions:**
1.  **Differentiate File Types:** Your first step is to differentiate between **image files** and **data files** (e.g., `.csv`, `.json`).
    *   **For Image Files:** You **MUST** perform OCR or visual analysis to extract all relevant data. This extracted data **MUST** be hardcoded directly into the Python script (e.g., as a `pandas.DataFrame`). **Do not write code that attempts to read the image file from disk.**
    *   **For Data Files (CSV, JSON, etc.):** You **MUST NOT** perform OCR. Instead, you **MUST** write Python code that reads the file directly from its file path. The file path will be provided in the context. **Do not hardcode the content of these files.**
2.  **Analyze All Inputs:** Carefully examine all provided content: the user's text prompt, any attached text files, and images.
3.  **Answer the Questions:** The generated script must use the data (either hardcoded from images or read from data files) to perform any requested analysis and answer the user's questions.
4.  **Output Formatting:**
    - If the user's prompt specifies a JSON structure (e.g., a JSON object with specific keys and data types), the script's final output **MUST** strictly match that structure.
    - If the user's prompt asks for an array of answers, the script's final output **MUST** be a single JSON array containing the answers in the correct order.
    - If no specific format is requested, the output should be a single JSON object or an array containing the answer(s).
5.  **JSON Output Only:** The script's final output, printed to standard output, **MUST** be a single, valid JSON object or array. Nothing else.
6.  **No Explanations:** Do not include any explanations, comments, or markdown formatting in your response. Your entire output should be the raw Python code.
7.  Make sure for decimals / float values go till 4 digits after the decimal
Your goal is to create a self-contained Python script that processes the data and produces the required JSON output in the format requested.
Make sure to answer correctly if asked which (unless mentioned) try to give the name / data point instead of the id of the data.
GENERATE THE CORRECT PYTHON CODE TO DO THIS, DO NOT ANSWER THE QUESTIONS. If the data is given in the "Additional Context" by user use that directly, clean the data add the data and use that to answer the quetsions via python
""")
        ]
    )
    print_with_timestamp(f"\nCalling Gemini Pro ({model})...", request_id=request_id)
    log_message(f"Calling Gemini Pro ({model}) to generate code.")
    full_response_text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        if time.time() - start_time > 170:
            print_with_timestamp("\n[TIMEOUT] Reached 170 seconds.", request_id=request_id)
            log_message("Timeout: Gemini Pro call timed out after 170 seconds.")
            break
        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
            part = chunk.candidates[0].content.parts[0]
            if part.text:
                print(part.text, end="")
                full_response_text += part.text
    
    final_code = extract_code_from_markdown(full_response_text)
    print_with_timestamp("\nFinal Python Code to Execute:\n", request_id=request_id)
    print(final_code)
    log_message(f"Generated Code: ```python\n{final_code[:1800]}```")
    return final_code


def fix_code_with_context(original_prompt: str, context_markdown: str, broken_code: str, error_message: str, start_time: float, loop=None, request_id=None):
    def log_message(msg):
        if loop:
            asyncio.run_coroutine_threadsafe(send_discord_webhook(msg, request_id=request_id), loop)

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.5-flash"
    
    fixer_prompt = f"""
The following Python code, which was generated based on the user's request, failed to execute correctly.
Original User Prompt:
---
{original_prompt}
---
Additional Context Provided to the AI:
---
{context_markdown}
---
The code that failed:
```python
{broken_code}
```
The error message was:
---
{error_message}
---
Your task is to analyze the original prompt, the code, and the error message to provide a corrected, fully functional version of the Python script.
The corrected script must successfully execute and produce the desired JSON output.
"""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=fixer_prompt)])]
    
    system_instruction = [
        types.Part.from_text(text="""
You are an expert Python debugging assistant. Your role is to provide a corrected version of a failing Python script.
Strictly follow these rules:
1.  Analyze the user's original prompt, the context, the broken code, and the error message.
2.  Generate a complete, corrected, and executable Python script that fixes the error and meets the user's original request.
3.  Your response MUST contain ONLY the raw Python code for the corrected script.
4.  Do NOT include any explanations, apologies, comments, or markdown formatting like ```python ... ```. Just the code.
5.  The corrected code must produce the final output in the specified JSON format.
6.  DO NOT inculde comments in python file and keep the python file short , like no multiople empty lines just for formatting nicely
7.  Make sure for decimals / float values go till 4 digits after the decimal
GENERATE THE CORRECT PYTHON CODE TO DO THIS, DO NOT ANSWER THE QUESTIONS. If the data is given in the "Additional Context" by user use that directly, clean the data add the data and use that to answer the quetsions via python

                                                          
""")
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=128),
        system_instruction=system_instruction
    )

    print_with_timestamp("\nAttempting to fix the code...", request_id=request_id)
    log_message(f"Fixing Code: Attempting to fix code. Error: ```{error_message[:1800]}```")
    full_response_text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        if time.time() - start_time > 170:
            print_with_timestamp("\n[TIMEOUT] Reached 170 seconds during fix attempt.", request_id=request_id)
            log_message("Timeout: Code fix attempt timed out after 170 seconds.")
            break
        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
            part = chunk.candidates[0].content.parts[0]
            if part.text:
                print(part.text, end="")
                full_response_text += part.text

    fixed_code = extract_code_from_markdown(full_response_text)
    print_with_timestamp("\nCorrected Python Code:\n", request_id=request_id)
    print(fixed_code)
    log_message(f"Corrected Code: Generated Corrected Code: ```python\n{fixed_code[:1800]}```")
    return fixed_code


def generate_approximate_json(questions_content: str, loop=None, request_id=None):
    """
    Calls Gemini to generate an approximate JSON response when code execution fails.
    """
    def log_message(msg):
        if loop:
            asyncio.run_coroutine_threadsafe(send_discord_webhook(msg, request_id=request_id), loop)

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.5-flash"

    prompt = f"""
Based on the user's question below, generate a JSON response that provides an approximate answer.
The code generation process failed, so we need a fallback. 
The JSON should have the correct structure, keys, and data types as expected by the question, but the values can be approximate or placeholder data.
For decimals / float values go till 4 digits after the decimal
User Question:
---
{questions_content}
---

Generate only the JSON object as the response.
"""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""
You are a data generation assistant. Your task is to create a valid JSON response based on a user's question.
The primary goal is to match the data structure (keys, data types, nesting) implied by the user's question.
The actual data values can be approximate or placeholders, but they must be of the correct type.
For example, if the user asks for a list of products with names and prices, you should return a JSON array of objects, where each object has a "name" (string) and a "price" (number).
Your entire output must be a single, valid JSON object or array. Do not include any explanations, comments, or markdown formatting.
For decimals / float values go till 4 digits after the decimal
""")
        ]
    )

    print_with_timestamp("\nGenerating approximate JSON response as a fallback...", request_id=request_id)
    log_message("Fallback: Code generation failed completely. Attempting to generate an approximate JSON response as a fallback.")

    output = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        if chunk.text:
            output += chunk.text
    
    print_with_timestamp("\nApproximate JSON response:\n", request_id=request_id)
    print(output)
    log_message(f"Approximate JSON: Generated approximate JSON: ```json\n{output[:1800]}```")

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        log_message("Error: Failed to parse the approximate JSON response.")
        return {"error": "Failed to generate a valid JSON response as a fallback."}


# --- Main API Endpoint ---
@app.get("/health", tags=["Health"])
async def health_check():
    return JSONResponse(content={"status": "ok"})

@app.post("/api/")
async def analyze_data(request: Request):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    loop = asyncio.get_running_loop()

    try:
        await send_discord_webhook(f"New request received.", request_id=request_id)

        form = await request.form()

        print_with_timestamp("Received form data:", request_id=request_id)
        for name, field in form.items():
            field_type = "file" if hasattr(field, 'filename') else "field"
            value_preview = ""
            if isinstance(field, str):
                value_preview = f"'{field[:100]}'"
            else:
                value_preview = f"({getattr(field, 'filename', 'N/A')}, {getattr(field, 'content_type', 'N/A')})"
            print_with_timestamp(f"- Name: {name}, Type: {field_type}, Value: {value_preview}", request_id=request_id)

        if "questions.txt" not in form or not hasattr(form["questions.txt"], 'filename'):
            await send_discord_webhook(f"Error: `questions.txt` is missing.", request_id=request_id)
            return JSONResponse(status_code=400, content={"message": "A file named questions.txt is required."})

        questions_content = ""
        file_context_parts = []
        image_files = []

        if os.getenv("RENDER") == "true":
            base_dir = "/tmp"
        else:
            base_dir = "tmp"

        request_dir = os.path.join(base_dir, request_id)
        os.makedirs(request_dir, exist_ok=True)
        await send_discord_webhook(f"Created temporary directory: `{request_dir}`", request_id=request_id)

        for name, file_or_field in form.items():
            if not hasattr(file_or_field, 'filename'):
                continue

            file = file_or_field
            content = await file.read()
            
            file_path = os.path.join(request_dir, file.filename)
            with open(file_path, 'wb') as f:
                f.write(content)

            if name == "questions.txt":
                questions_content = content.decode('utf-8', errors='ignore')
                continue
            
            if file.content_type and file.content_type.startswith('image/'):
                image_files.append({"name": file.filename, "content": content, "content_type": file.content_type})
                file_context_parts.append(f"- An image file named `{file.filename}` is available.")
                continue

            is_zip = file.filename.lower().endswith('.zip')
            is_tar = file.filename.lower().endswith(('.tar', '.tar.gz', '.tgz'))

            if is_zip or is_tar:
                archive_path = file_path
                file_context_parts.append(f"- An archive file named `{file.filename}` was uploaded and will be extracted.")
                try:
                    extracted_filenames = []
                    if is_zip:
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            extracted_filenames = zip_ref.namelist()
                            zip_ref.extractall(request_dir)
                    elif is_tar:
                        with tarfile.open(archive_path, 'r:*') as tar_ref:
                            extracted_filenames = tar_ref.getnames()
                            tar_ref.extractall(request_dir)
                    
                    # Create context for a sample of extracted files
                    for extracted_filename in extracted_filenames[:10]: # Limit to 10 files for context
                        extracted_filepath = os.path.join(request_dir, extracted_filename)
                        if os.path.isdir(extracted_filepath):
                            continue

                        try:
                            ext = extracted_filename.lower().split('.')[-1]
                            if ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg']:
                                file_context_parts.append(f"- Extracted image file: `{extracted_filename}`.")
                            else:
                                with open(extracted_filepath, 'r', encoding='utf-8', errors='ignore') as f_extracted:
                                    preview = "\n".join(f_extracted.read().splitlines()[:15])
                                    file_context_parts.append(f"- Extracted text file `{extracted_filename}`. Preview:\n---\n{preview}\n---")
                        except Exception:
                            file_context_parts.append(f"- Extracted binary file: `{extracted_filename}`.")
                except (zipfile.BadZipFile, tarfile.ReadError) as e:
                    file_context_parts.append(f"- Error: Could not extract archive `{file.filename}`. It may be corrupt. Error: {e}")
            else:
                try:
                    decoded_content = content.decode('utf-8', errors='ignore')
                    preview = "\n".join(decoded_content.splitlines()[:15])
                    file_context_parts.append(f"- A text file named `{file.filename}` is available at `{file.filename}`. Preview:\n---\n{preview}\n---")
                except Exception:
                    file_context_parts.append(f"- A binary file named `{file.filename}` is available at `{file.filename}`.")

        metadata = await asyncio.to_thread(extract_prompt_metadata, questions_content, loop, request_id)
        
        scraped_context = ""
        if 'url' in metadata and metadata['url']:
            scraped_context = await scrape_url_to_markdown(metadata['url'], request_id=request_id)
            
        full_context = "\n".join(file_context_parts + [scraped_context]).strip()

        generated_code = await asyncio.to_thread(generate_answer_with_context, questions_content, full_context, image_files, start_time, loop, request_id)
        stdout, stderr = await asyncio.to_thread(execute_code, generated_code, request_dir, loop, request_id)

        if stderr:
            error_message = stderr
        else:
            try:
                response_content = json.loads(stdout)
                time_taken = time.time() - start_time
                await send_discord_webhook(f"Success: Request successful. Sending final response. Time taken: {time_taken:.2f}s", request_id=request_id)
                print_with_timestamp(f"Success: Request successful. Time taken: {time_taken:.2f}s", request_id=request_id)
                return JSONResponse(content=response_content)
            except (json.JSONDecodeError, TypeError):
                error_message = f"Script output was not valid JSON. Output:\n{stdout}"

        if error_message:
            current_code = generated_code
            last_error = error_message
            for i in range(3):
                await send_discord_webhook(f"__Attempt {i + 1}/3 to fix code.__", request_id=request_id)
                fixed_code = await asyncio.to_thread(fix_code_with_context, questions_content, full_context, current_code, last_error, start_time, loop, request_id)
                stdout, stderr = await asyncio.to_thread(execute_code, fixed_code, request_dir, loop, request_id)

                if not stderr:
                    try:
                        response_content = json.loads(stdout)
                        time_taken = time.time() - start_time
                        await send_discord_webhook(f"Success: Request successful after fix attempt {i+1}. Sending final response. Time taken: {time_taken:.2f}s", request_id=request_id)
                        print_with_timestamp(f"Success: Request successful after fix. Time taken: {time_taken:.2f}s", request_id=request_id)
                        return JSONResponse(content=response_content)
                    except (json.JSONDecodeError, TypeError):
                        last_error = f"Script output was not valid JSON. Output:\n{stdout}"
                        current_code = fixed_code
                else:
                    last_error = stderr
                    current_code = fixed_code
            
            # If all fixing attempts fail, call the new fallback function
            await send_discord_webhook(f"Error: All code fixing attempts failed. Trying to generate approximate JSON.", request_id=request_id)
            approximate_json = await asyncio.to_thread(generate_approximate_json, questions_content, loop, request_id)
            
            time_taken = time.time() - start_time
            # Check if the fallback also failed
            if "error" in approximate_json:
                final_error_payload = {"error": "Code execution and fallback JSON generation failed.", "final_error": last_error}
                await send_discord_webhook(f"Error: Fallback JSON generation also failed. Final error: ```{last_error[:1500]}``` Time taken: {time_taken:.2f}s", request_id=request_id)
                return JSONResponse(status_code=500, content=final_error_payload)

            await send_discord_webhook(f"Success: Successfully generated approximate JSON. Time taken: {time_taken:.2f}s", request_id=request_id)
            return JSONResponse(content=approximate_json)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        time_taken = time.time() - start_time
        print_with_timestamp(f"FATAL ERROR: An unexpected error occurred: {e}", request_id=request_id)
        traceback.print_exc()
        await send_discord_webhook(f"FATAL ERROR: An unexpected server error occurred: {e}\n```{tb_str}``` Time taken: {time_taken:.2f}s", request_id=request_id)
        return JSONResponse(status_code=500, content={"error": "An unexpected server error occurred.", "detail": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
