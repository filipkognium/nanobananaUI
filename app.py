import streamlit as st
import base64
import io
from PIL import Image
from google import genai
from google.genai import types

# Pricing constants (USD per 1M tokens) - from Google AI pricing page
# https://ai.google.dev/gemini-api/docs/pricing
PRICING = {
    "gemini-3-pro-image-preview": {
        "input_per_1m": 2.00,           # $2.00 per 1M input tokens (text/image)
        "output_text_per_1m": 12.00,    # $12.00 per 1M output tokens (text + thinking)
        "output_image_per_1m": 120.00,  # $120.00 per 1M output image tokens
    },
    "gemini-2.5-flash-image": {
        "input_per_1m": 0.30,           # $0.30 per 1M input tokens
        "output_text_per_1m": 2.50,     # $2.50 per 1M output tokens
        "output_image_per_1m": 30.00,   # $30.00 per 1M output image tokens
    }
}

def calculate_cost_from_usage(model: str, usage_metadata, num_output_images: int = 0) -> dict:
    """Calculate actual cost from API response usage_metadata."""
    pricing = PRICING.get(model, PRICING["gemini-3-pro-image-preview"])

    # Extract token counts from usage_metadata
    prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
    candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
    total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0

    # For image output models, estimate image tokens from total
    # Image tokens: 1120 for 1K/2K, 2000 for 4K (Nano Banana Pro)
    # Image tokens: 1290 for Nano Banana Flash
    if model == "gemini-3-pro-image-preview":
        image_tokens_per_image = 1120  # Conservative estimate for 1K/2K
    else:
        image_tokens_per_image = 1290

    estimated_image_tokens = num_output_images * image_tokens_per_image
    text_output_tokens = max(0, candidates_tokens - estimated_image_tokens)

    # Calculate costs
    input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
    output_text_cost = (text_output_tokens / 1_000_000) * pricing["output_text_per_1m"]
    output_image_cost = (estimated_image_tokens / 1_000_000) * pricing["output_image_per_1m"]
    total_cost = input_cost + output_text_cost + output_image_cost

    return {
        "prompt_tokens": prompt_tokens,
        "candidates_tokens": candidates_tokens,
        "total_tokens": total_tokens,
        "text_output_tokens": text_output_tokens,
        "image_output_tokens": estimated_image_tokens,
        "input_cost": input_cost,
        "output_text_cost": output_text_cost,
        "output_image_cost": output_image_cost,
        "total_cost": total_cost
    }

def display_cost_breakdown(cost: dict, model: str):
    """Display actual cost breakdown from usage_metadata."""
    with st.expander("💰 Cost Breakdown (Actual Usage)", expanded=True):
        # Token usage section
        st.markdown("### 📊 Token Usage")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Tokens", f"{cost['prompt_tokens']:,}")
        with col2:
            st.metric("Output Tokens", f"{cost['candidates_tokens']:,}")
        with col3:
            st.metric("Total Tokens", f"{cost['total_tokens']:,}")

        st.divider()

        # Cost breakdown section
        st.markdown("### 💵 Cost Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input:**")
            st.write(f"• {cost['prompt_tokens']:,} tokens @ ${PRICING[model]['input_per_1m']}/1M")
            st.write(f"• **${cost['input_cost']:.6f}**")

        with col2:
            st.markdown("**Output:**")
            st.write(f"• Text: {cost['text_output_tokens']:,} tokens = ${cost['output_text_cost']:.6f}")
            st.write(f"• Images: {cost['image_output_tokens']:,} tokens = ${cost['output_image_cost']:.6f}")

        st.divider()
        st.markdown(f"## Total Cost: **${cost['total_cost']:.4f}**")

        # Pricing reference
        if model == "gemini-3-pro-image-preview":
            st.caption("📝 Nano Banana Pro: $2/1M input, $12/1M text output, $120/1M image output")
        else:
            st.caption("📝 Nano Banana Flash: $0.30/1M input, $2.50/1M text output, $30/1M image output")

# Page config
st.set_page_config(
    page_title="Nano Banana Pro Tester",
    page_icon="🍌",
    layout="wide"
)

# Password protection - uses Streamlit secrets (st.secrets["APP_PASSWORD"])
def check_password():
    """Returns True if the user has entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🍌 Nano Banana Pro API Tester")
    st.markdown("### 🔐 Please enter password to access the app")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Login", type="primary"):
        if password == st.secrets.get("APP_PASSWORD", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password")

    return False

if not check_password():
    st.stop()

# Title
st.title("🍌 Nano Banana Pro API Tester")
st.markdown("*Google's Gemini 3 Pro Image Preview - Full capabilities testing UI*")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google AI API Key", type="password", help="Enter your Google AI Studio API key")

    st.divider()
    st.header("📊 Model Selection")
    model_choice = st.selectbox(
        "Model",
        ["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],
        help="Nano Banana Pro (Gemini 3) for advanced features, Nano Banana (2.5 Flash) for speed"
    )

    st.divider()
    st.header("🖼️ Image Settings")

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        help="Output image aspect ratio"
    )

    # Image size only for Nano Banana Pro
    if model_choice == "gemini-3-pro-image-preview":
        image_size = st.selectbox(
            "Image Size",
            ["1K", "2K", "4K"],
            help="Output resolution (Nano Banana Pro only)"
        )
    else:
        image_size = None

    response_modality = st.selectbox(
        "Response Type",
        ["Text and Image", "Image Only"],
        help="What the model should return"
    )

# Initialize client
def get_client():
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_image(client, prompt, uploaded_images=None):
    """Generate or edit images using Nano Banana Pro"""
    try:
        # Build contents
        contents = []

        # Add uploaded images first
        if uploaded_images:
            for img in uploaded_images:
                contents.append(img)

        # Add text prompt
        contents.append(prompt)

        # Build config
        response_modalities = ["Image"] if response_modality == "Image Only" else ["Text", "Image"]

        image_config_dict = {"aspect_ratio": aspect_ratio}
        if image_size and model_choice == "gemini-3-pro-image-preview":
            image_config_dict["image_size"] = image_size

        config = types.GenerateContentConfig(
            response_modalities=response_modalities,
            image_config=types.ImageConfig(**image_config_dict)
        )

        # Generate
        response = client.models.generate_content(
            model=model_choice,
            contents=contents,
            config=config
        )

        return response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Text to Image",
    "✏️ Image Editing",
    "🔀 Multi-Image Composition",
    "📖 Documentation"
])

# Tab 1: Text to Image
with tab1:
    st.header("Text to Image Generation")
    st.markdown("Generate images from text descriptions. Be hyper-specific for best results!")

    prompt_t2i = st.text_area(
        "Enter your prompt",
        placeholder="Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
        height=100,
        key="t2i_prompt"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate", key="t2i_generate", type="primary")

    if generate_btn:
        if not api_key:
            st.error("Please enter your API key in the sidebar")
        elif not prompt_t2i:
            st.warning("Please enter a prompt")
        else:
            client = get_client()
            with st.spinner("Generating image..."):
                response = generate_image(client, prompt_t2i)

                if response and response.candidates:
                    num_generated_images = 0
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            st.markdown("**Model Response:**")
                            st.write(part.text)
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            num_generated_images += 1
                            # Get image data (may be bytes or base64 string)
                            raw_data = part.inline_data.data
                            if isinstance(raw_data, str):
                                image_data = base64.b64decode(raw_data)
                            else:
                                image_data = raw_data
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="Generated Image")

                            # Download button
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            st.download_button(
                                "📥 Download Image",
                                buf.getvalue(),
                                "generated_image.png",
                                "image/png"
                            )

                    # Display cost breakdown from actual usage
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        cost = calculate_cost_from_usage(
                            model=model_choice,
                            usage_metadata=response.usage_metadata,
                            num_output_images=max(1, num_generated_images)
                        )
                        display_cost_breakdown(cost, model_choice)


# Tab 2: Image Editing
with tab2:
    st.header("Image Editing")
    st.markdown("Upload an image and describe the changes you want to make.")

    uploaded_file = st.file_uploader(
        "Upload an image to edit",
        type=["png", "jpg", "jpeg", "webp"],
        key="edit_upload"
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image:**")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_container_width=True)

    edit_prompt = st.text_area(
        "Describe your edits",
        placeholder="Add a sunset sky in the background and make the lighting warmer",
        height=100,
        key="edit_prompt"
    )

    if st.button("✏️ Apply Edits", key="edit_apply", type="primary"):
        if not api_key:
            st.error("Please enter your API key in the sidebar")
        elif not uploaded_file:
            st.warning("Please upload an image first")
        elif not edit_prompt:
            st.warning("Please describe the edits you want")
        else:
            client = get_client()
            original_image = Image.open(uploaded_file)

            with st.spinner("Applying edits..."):
                response = generate_image(client, edit_prompt, [original_image])

                if response and response.candidates:
                    num_generated_images = 0
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            st.markdown("**Model Response:**")
                            st.write(part.text)
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            num_generated_images += 1
                            raw_data = part.inline_data.data
                            if isinstance(raw_data, str):
                                image_data = base64.b64decode(raw_data)
                            else:
                                image_data = raw_data
                            edited_image = Image.open(io.BytesIO(image_data))

                            with col2:
                                st.markdown("**Edited Image:**")
                                st.image(edited_image, use_container_width=True)

                            buf = io.BytesIO()
                            edited_image.save(buf, format="PNG")
                            st.download_button(
                                "📥 Download Edited Image",
                                buf.getvalue(),
                                "edited_image.png",
                                "image/png"
                            )

                    # Display cost breakdown from actual usage
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        cost = calculate_cost_from_usage(
                            model=model_choice,
                            usage_metadata=response.usage_metadata,
                            num_output_images=max(1, num_generated_images)
                        )
                        display_cost_breakdown(cost, model_choice)

# Tab 3: Multi-Image Composition
with tab3:
    st.header("Multi-Image Composition")
    st.markdown("""
    Upload multiple images and combine them creatively. Examples:
    - Transfer styles between images
    - Composite elements from different images
    - Create fashion photos with clothing from one image on a model from another
    """)

    num_images = st.slider("Number of images to upload", 2, 5 if model_choice == "gemini-3-pro-image-preview" else 3, 2)

    uploaded_images = []
    cols = st.columns(num_images)

    for i, col in enumerate(cols):
        with col:
            img_file = st.file_uploader(
                f"Image {i+1}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"multi_img_{i}"
            )
            if img_file:
                img = Image.open(img_file)
                st.image(img, use_container_width=True)
                uploaded_images.append(img)

    multi_prompt = st.text_area(
        "Describe how to combine the images",
        placeholder="Take the dress from the first image and put it on the woman from the second image. Create a professional e-commerce fashion photo.",
        height=100,
        key="multi_prompt"
    )

    if st.button("🔀 Compose Images", key="multi_compose", type="primary"):
        if not api_key:
            st.error("Please enter your API key in the sidebar")
        elif len(uploaded_images) < 2:
            st.warning("Please upload at least 2 images")
        elif not multi_prompt:
            st.warning("Please describe how to combine the images")
        else:
            client = get_client()
            num_input_imgs = len(uploaded_images)

            with st.spinner("Composing images..."):
                response = generate_image(client, multi_prompt, uploaded_images)

                if response and response.candidates:
                    st.markdown("### Result")
                    num_generated_images = 0
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            st.write(part.text)
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            num_generated_images += 1
                            raw_data = part.inline_data.data
                            if isinstance(raw_data, str):
                                image_data = base64.b64decode(raw_data)
                            else:
                                image_data = raw_data
                            result_image = Image.open(io.BytesIO(image_data))
                            st.image(result_image, caption="Composed Image")

                            buf = io.BytesIO()
                            result_image.save(buf, format="PNG")
                            st.download_button(
                                "📥 Download Composed Image",
                                buf.getvalue(),
                                "composed_image.png",
                                "image/png"
                            )

                    # Display cost breakdown from actual usage
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        cost = calculate_cost_from_usage(
                            model=model_choice,
                            usage_metadata=response.usage_metadata,
                            num_output_images=max(1, num_generated_images)
                        )
                        display_cost_breakdown(cost, model_choice)

# Tab 4: Documentation
with tab4:
    st.header("📖 Nano Banana Pro API Documentation")

    st.markdown("""
    ## Model Overview

    **Nano Banana Pro** (`gemini-3-pro-image-preview`) is Google's most advanced image generation model,
    part of the Gemini 3 family. It's designed for professional asset production and complex instructions.

    **Nano Banana** (`gemini-2.5-flash-image`) is optimized for speed and efficiency, ideal for
    high-volume, low-latency tasks.

    ---

    ## Key Features

    ### 🎨 Text to Image
    Generate high-quality images from text descriptions with unprecedented control.

    ### ✏️ Image Editing
    Edit images using natural language - add, remove, or modify elements without masks.

    ### 🔀 Multi-Image Composition
    Combine elements from multiple images with a single prompt.

    ### 📝 High-Fidelity Text Rendering
    Generate images with legible, well-placed text for logos, diagrams, and posters.

    ---

    ## Best Practices

    1. **Be Hyper-Specific**: Instead of "fantasy armor," describe "ornate elven plate armor,
       etched with silver leaf patterns, with a high collar and pauldrons shaped like falcon wings."

    2. **Provide Context**: Explain the purpose. "Create a logo for a high-end, minimalist skincare brand"
       works better than just "Create a logo."

    3. **Iterate and Refine**: Use follow-up prompts like "make the lighting warmer" or
       "change the expression to be more serious."

    4. **Step-by-Step for Complex Scenes**: Break prompts into steps for complex images.

    5. **Control the Camera**: Use photographic terms like "wide-angle shot," "macro shot,"
       "low-angle perspective."

    ---

    ## Resolution Options (Nano Banana Pro)

    | Size | Example Resolution (1:1) |
    |------|-------------------------|
    | 1K   | 1024x1024              |
    | 2K   | 2048x2048              |
    | 4K   | 4096x4096              |

    ---

    ## Supported Aspect Ratios

    `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

    ---

    ## Limitations

    - Best performance with: EN, ar-EG, de-DE, es-MX, fr-FR, hi-IN, id-ID, it-IT, ja-JP, ko-KR, pt-BR, ru-RU, ua-UA, vi-VN, zh-CN
    - No audio or video inputs
    - Nano Banana: Up to 3 input images
    - Nano Banana Pro: Up to 5 high-fidelity images, 14 total
    - All generated images include SynthID watermark
    """)

# Footer
st.divider()
st.caption("🍌 Nano Banana Pro Tester | Powered by Google Gemini API")

