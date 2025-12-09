import streamlit as st
import base64
import io
import replicate
import requests
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
    replicate_api_key = st.text_input("Replicate API Key", type="password", help="Enter your Replicate API key (for Flux Kontext Pro)")

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

def generate_flux_kontext(prompt, image_bytes, aspect_ratio="1:1"):
    """Generate image using Flux Kontext Pro from Replicate. Returns (output, cost)."""
    try:
        if not replicate_api_key:
            st.error("Please enter your Replicate API key in the sidebar")
            return None, 0.0

        # Set the API token
        import os
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

        # Convert image bytes to data URI
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:image/png;base64,{img_base64}"

        # Create prediction to get metrics including cost
        model = replicate.models.get("black-forest-labs/flux-kontext-pro")
        version = model.latest_version

        prediction = replicate.predictions.create(
            version=version,
            input={
                "prompt": prompt,
                "image": data_uri,
                "aspect_ratio": aspect_ratio,
            }
        )

        # Wait for completion
        prediction.wait()

        # Get cost from metrics if available
        cost = 0.0
        if prediction.metrics and 'predict_time' in prediction.metrics:
            # Flux Kontext Pro costs ~$0.05 per image (based on ~5 seconds at $0.01/sec)
            predict_time = prediction.metrics.get('predict_time', 5)
            cost = predict_time * 0.01  # Approximate cost per second
        else:
            cost = 0.05  # Fallback estimate

        return prediction.output, cost
    except Exception as e:
        st.error(f"Flux Kontext Pro Error: {str(e)}")
        return None, 0.0

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

def generate_image_with_model(client, prompt, uploaded_images, target_model):
    """Generate or edit images using a specific model"""
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
        if image_size and target_model == "gemini-3-pro-image-preview":
            image_config_dict["image_size"] = image_size

        config = types.GenerateContentConfig(
            response_modalities=response_modalities,
            image_config=types.ImageConfig(**image_config_dict)
        )

        # Generate with specified model
        response = client.models.generate_content(
            model=target_model,
            contents=contents,
            config=config
        )

        return response
    except Exception as e:
        st.error(f"Error ({target_model}): {str(e)}")
        return None

# Initialize session state for comparison history
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎨 Text to Image",
    "✏️ Image Editing",
    "🔀 Multi-Image Composition",
    "⚖️ Triple Compare",
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

# Tab 4: Bulk Triple Comparison
with tab4:
    st.header("⚖️ Bulk Triple Model Comparison")
    st.markdown("Upload **multiple images** and test three different prompts across three models. Each image will be processed by all three models.")

    # Multi-file uploader
    compare_images = st.file_uploader(
        "Upload images to compare",
        type=["png", "jpg", "jpeg", "webp"],
        key="compare_upload",
        accept_multiple_files=True
    )

    if compare_images:
        st.markdown(f"### 📷 {len(compare_images)} Image(s) Uploaded")

        # Show thumbnails of uploaded images
        thumb_cols = st.columns(min(len(compare_images), 6))
        for idx, img_file in enumerate(compare_images):
            with thumb_cols[idx % 6]:
                img = Image.open(img_file)
                st.image(img, caption=f"Image {idx+1}", use_container_width=True)

        st.divider()

        # Prompt inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Prompt A (Nano Banana Pro)")
            prompt_a = st.text_area("Enter Prompt A", placeholder="Make it look like a watercolor painting", height=100, key="prompt_a")

        with col2:
            st.markdown("### Prompt B (Nano Banana Pro)")
            prompt_b = st.text_area("Enter Prompt B", placeholder="Make it look like an oil painting", height=100, key="prompt_b")

        with col3:
            st.markdown("### Prompt C (Flux Kontext Pro)")
            prompt_c = st.text_area("Enter Prompt C", placeholder="Transform into a pencil sketch", height=100, key="prompt_c")

        # Estimate
        total_generations = len(compare_images) * 3
        st.info(f"📊 This will generate **{total_generations} images** ({len(compare_images)} images × 3 models)")

        if st.button("🚀 Run Bulk Comparison", type="primary", key="run_compare"):
            missing_keys = []
            if not api_key:
                missing_keys.append("Google AI API Key")
            if not replicate_api_key:
                missing_keys.append("Replicate API Key")

            if missing_keys:
                st.error(f"Please enter: {', '.join(missing_keys)} in the sidebar")
            elif not prompt_a or not prompt_b or not prompt_c:
                st.warning("Please enter all three prompts")
            else:
                client = get_client()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = []
                total_cost = {"gemini_pro": 0, "gemini_flash": 0, "flux": 0}

                for img_idx, img_file in enumerate(compare_images):
                    # Prepare image
                    img_file.seek(0)
                    img_bytes = img_file.read()
                    mime_type = f"image/{img_file.type.split('/')[-1]}" if '/' in img_file.type else "image/png"
                    uploaded_img = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                    source_img = Image.open(io.BytesIO(img_bytes))

                    result_row = {
                        "source": source_img,
                        "source_name": img_file.name,
                        "result_a": None,
                        "result_b": None,
                        "result_c": None,
                        "cost_a": None,
                        "cost_b": None,
                    }

                    base_progress = img_idx / len(compare_images)

                    # Generate with Gemini Pro
                    status_text.text(f"Image {img_idx+1}/{len(compare_images)}: Generating with Gemini Pro...")
                    progress_bar.progress(base_progress + 0.1 / len(compare_images))

                    response_a = generate_image_with_model(client, prompt_a, [uploaded_img], "gemini-3-pro-image-preview")
                    if response_a and response_a.candidates:
                        for part in response_a.candidates[0].content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                raw_data = part.inline_data.data
                                if isinstance(raw_data, str):
                                    image_data = base64.b64decode(raw_data)
                                else:
                                    image_data = raw_data
                                result_row["result_a"] = Image.open(io.BytesIO(image_data))
                        if hasattr(response_a, 'usage_metadata') and response_a.usage_metadata:
                            cost = calculate_cost_from_usage("gemini-3-pro-image-preview", response_a.usage_metadata, 1)
                            result_row["cost_a"] = cost
                            total_cost["gemini_pro"] += cost['total_cost']

                    # Generate with Nano Banana Pro (Prompt B)
                    status_text.text(f"Image {img_idx+1}/{len(compare_images)}: Generating with Nano Banana Pro (Prompt B)...")
                    progress_bar.progress(base_progress + 0.2 / len(compare_images))

                    response_b = generate_image_with_model(client, prompt_b, [uploaded_img], "gemini-3-pro-image-preview")
                    if response_b and response_b.candidates:
                        for part in response_b.candidates[0].content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                raw_data = part.inline_data.data
                                if isinstance(raw_data, str):
                                    image_data = base64.b64decode(raw_data)
                                else:
                                    image_data = raw_data
                                result_row["result_b"] = Image.open(io.BytesIO(image_data))
                        if hasattr(response_b, 'usage_metadata') and response_b.usage_metadata:
                            cost = calculate_cost_from_usage("gemini-3-pro-image-preview", response_b.usage_metadata, 1)
                            result_row["cost_b"] = cost
                            total_cost["gemini_pro"] += cost['total_cost']

                    # Generate with Flux Kontext Pro
                    status_text.text(f"Image {img_idx+1}/{len(compare_images)}: Generating with Flux Kontext Pro...")
                    progress_bar.progress(base_progress + 0.3 / len(compare_images))

                    output_c, flux_cost = generate_flux_kontext(prompt_c, img_bytes, aspect_ratio)
                    if output_c:
                        if hasattr(output_c, 'read'):
                            result_row["result_c"] = Image.open(output_c)
                        elif isinstance(output_c, str):
                            resp = requests.get(output_c)
                            result_row["result_c"] = Image.open(io.BytesIO(resp.content))
                        else:
                            img_url = output_c[0] if isinstance(output_c, list) else str(output_c)
                            resp = requests.get(img_url)
                            result_row["result_c"] = Image.open(io.BytesIO(resp.content))
                        result_row["cost_c"] = flux_cost
                        total_cost["flux"] += flux_cost

                    all_results.append(result_row)
                    progress_bar.progress((img_idx + 1) / len(compare_images))

                status_text.text("✅ All generations complete!")
                progress_bar.progress(1.0)

                # Display results grid
                st.divider()
                st.markdown("## 📊 Results")

                # Header row
                header_cols = st.columns([1, 1, 1, 1])
                with header_cols[0]:
                    st.markdown("**Source**")
                with header_cols[1]:
                    st.markdown(f"**Nano Banana Pro (A)**\n\n_{prompt_a[:50]}..._" if len(prompt_a) > 50 else f"**Nano Banana Pro (A)**\n\n_{prompt_a}_")
                with header_cols[2]:
                    st.markdown(f"**Nano Banana Pro (B)**\n\n_{prompt_b[:50]}..._" if len(prompt_b) > 50 else f"**Nano Banana Pro (B)**\n\n_{prompt_b}_")
                with header_cols[3]:
                    st.markdown(f"**Flux Kontext Pro**\n\n_{prompt_c[:50]}..._" if len(prompt_c) > 50 else f"**Flux Kontext Pro**\n\n_{prompt_c}_")

                # Result rows
                for row_idx, row in enumerate(all_results):
                    cols = st.columns([1, 1, 1, 1])

                    with cols[0]:
                        st.image(row["source"], caption=row["source_name"], use_container_width=True)

                    with cols[1]:
                        if row["result_a"]:
                            st.image(row["result_a"], use_container_width=True)
                            if row["cost_a"]:
                                st.caption(f"${row['cost_a']['total_cost']:.4f}")
                            buf = io.BytesIO()
                            row["result_a"].save(buf, format="PNG")
                            st.download_button("📥", buf.getvalue(), f"gemini_pro_{row_idx}.png", "image/png", key=f"dl_a_{row_idx}")
                        else:
                            st.warning("Failed")

                    with cols[2]:
                        if row["result_b"]:
                            st.image(row["result_b"], use_container_width=True)
                            if row["cost_b"]:
                                st.caption(f"${row['cost_b']['total_cost']:.4f}")
                            buf = io.BytesIO()
                            row["result_b"].save(buf, format="PNG")
                            st.download_button("📥", buf.getvalue(), f"nano_banana_pro_b_{row_idx}.png", "image/png", key=f"dl_b_{row_idx}")
                        else:
                            st.warning("Failed")

                    with cols[3]:
                        if row["result_c"]:
                            st.image(row["result_c"], use_container_width=True)
                            flux_cost = row.get("cost_c", 0)
                            st.caption(f"${flux_cost:.4f}")
                            buf = io.BytesIO()
                            row["result_c"].save(buf, format="PNG")
                            st.download_button("📥", buf.getvalue(), f"flux_{row_idx}.png", "image/png", key=f"dl_c_{row_idx}")
                        else:
                            st.warning("Failed")

                # Total cost summary
                st.divider()
                total_all = total_cost["gemini_pro"] + total_cost["flux"]
                st.markdown(f"""
                ### 💰 Total Cost Summary
                | Model | Cost |
                |-------|------|
                | Nano Banana Pro (A+B) | ${total_cost['gemini_pro']:.4f} |
                | Flux Kontext Pro | ${total_cost['flux']:.2f} |
                | **Total** | **${total_all:.4f}** |
                """)

                # Generate Report
                st.divider()
                st.markdown("### 📄 Generate Report")

                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Build HTML report
                def pil_to_base64_html(img):
                    if img is None:
                        return ""
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return base64.b64encode(buf.getvalue()).decode()

                html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bulk Comparison Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .image-section {{ background: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); page-break-inside: avoid; }}
        .image-section h2 {{ color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .reference {{ text-align: center; margin-bottom: 20px; }}
        .reference img {{ max-width: 300px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .edits {{ display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }}
        .edit {{ flex: 1; min-width: 280px; max-width: 350px; background: #fafafa; padding: 15px; border-radius: 8px; }}
        .edit h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .edit .model {{ color: #007bff; font-weight: bold; font-size: 16px; margin-bottom: 5px; }}
        .edit .prompt {{ color: #888; font-size: 12px; font-style: italic; margin-bottom: 10px; word-wrap: break-word; }}
        .edit img {{ width: 100%; border-radius: 6px; }}
        .cost {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        .cost-table {{ width: 100%; border-collapse: collapse; }}
        .cost-table th, .cost-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .cost-table th {{ background: #f0f0f0; }}
        @media print {{ .image-section {{ page-break-inside: avoid; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍌 Bulk Comparison Report</h1>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Total Images:</strong> {len(all_results)}</p>
        <p><strong>Models Used:</strong> Nano Banana Pro (×2), Flux Kontext Pro</p>
        <div class="cost">
            <table class="cost-table">
                <tr><th>Model</th><th>Prompt</th><th>Cost</th></tr>
                <tr><td>Nano Banana Pro (A)</td><td>{prompt_a[:100]}{'...' if len(prompt_a) > 100 else ''}</td><td>(included below)</td></tr>
                <tr><td>Nano Banana Pro (B)</td><td>{prompt_b[:100]}{'...' if len(prompt_b) > 100 else ''}</td><td>(included below)</td></tr>
                <tr><td>Nano Banana Pro Total</td><td></td><td>${total_cost['gemini_pro']:.4f}</td></tr>
                <tr><td>Flux Kontext Pro</td><td>{prompt_c[:100]}{'...' if len(prompt_c) > 100 else ''}</td><td>${total_cost['flux']:.2f}</td></tr>
                <tr><th>Total</th><th></th><th>${total_all:.4f}</th></tr>
            </table>
        </div>
    </div>
"""

                for idx, row in enumerate(all_results):
                    source_b64 = pil_to_base64_html(row.get("source"))
                    result_a_b64 = pil_to_base64_html(row.get("result_a"))
                    result_b_b64 = pil_to_base64_html(row.get("result_b"))
                    result_c_b64 = pil_to_base64_html(row.get("result_c"))

                    cost_a = row.get("cost_a", {}).get("total_cost", 0) if row.get("cost_a") else 0
                    cost_b = row.get("cost_b", {}).get("total_cost", 0) if row.get("cost_b") else 0
                    cost_c = row.get("cost_c", 0)

                    html_report += f"""
    <div class="image-section">
        <h2>Image {idx + 1}: {row.get('source_name', 'Unknown')}</h2>

        <div class="reference">
            <h3>Reference Image</h3>
            <img src="data:image/png;base64,{source_b64}" alt="Reference">
        </div>

        <div class="edits">
            <div class="edit">
                <h3>Edit 1</h3>
                <div class="model">Nano Banana Pro</div>
                <div class="prompt">"{prompt_a}"</div>
                {'<img src="data:image/png;base64,' + result_a_b64 + '" alt="Nano Banana Pro Result A">' if result_a_b64 else '<p style="color: red;">Generation failed</p>'}
                <p style="font-size: 11px; color: #666;">Cost: ${cost_a:.4f}</p>
            </div>

            <div class="edit">
                <h3>Edit 2</h3>
                <div class="model">Nano Banana Pro</div>
                <div class="prompt">"{prompt_b}"</div>
                {'<img src="data:image/png;base64,' + result_b_b64 + '" alt="Nano Banana Pro Result B">' if result_b_b64 else '<p style="color: red;">Generation failed</p>'}
                <p style="font-size: 11px; color: #666;">Cost: ${cost_b:.4f}</p>
            </div>

            <div class="edit">
                <h3>Edit 3</h3>
                <div class="model">Flux Kontext Pro</div>
                <div class="prompt">"{prompt_c}"</div>
                {'<img src="data:image/png;base64,' + result_c_b64 + '" alt="Flux Kontext Pro Result">' if result_c_b64 else '<p style="color: red;">Generation failed</p>'}
                <p style="font-size: 11px; color: #666;">Cost: ${cost_c:.4f}</p>
            </div>
        </div>
    </div>
"""

                html_report += """
</body>
</html>
"""

                # Download button for HTML report
                st.download_button(
                    "📄 Download HTML Report",
                    html_report,
                    f"comparison_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    "text/html",
                    key="download_report"
                )

                # Save to history
                comparison_entry = {
                    "timestamp": timestamp,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "prompt_c": prompt_c,
                    "results": all_results,
                    "total_cost": total_cost,
                    "num_images": len(compare_images),
                    "model": "Bulk Triple Comparison"
                }
                st.session_state.comparison_history.insert(0, comparison_entry)
                st.success("✅ Bulk comparison saved to history!")

    # Display history
    if st.session_state.comparison_history:
        st.divider()
        st.markdown("## 📜 Comparison History")

        for i, entry in enumerate(st.session_state.comparison_history):
            is_bulk = 'results' in entry
            label = f"🕐 {entry['timestamp']} - {entry.get('model', 'Unknown')}"
            if is_bulk:
                label += f" ({entry.get('num_images', '?')} images)"

            with st.expander(label, expanded=(i == 0)):
                if is_bulk:
                    # Bulk comparison display
                    st.markdown(f"**Prompts:** A: {entry['prompt_a'][:50]}... | B: {entry['prompt_b'][:50]}... | C: {entry['prompt_c'][:50]}...")

                    for row_idx, row in enumerate(entry.get('results', [])):
                        cols = st.columns([1, 1, 1, 1])
                        with cols[0]:
                            if row.get("source"):
                                st.image(row["source"], caption=row.get("source_name", f"Image {row_idx+1}"), use_container_width=True)
                        with cols[1]:
                            if row.get("result_a"):
                                st.image(row["result_a"], use_container_width=True)
                        with cols[2]:
                            if row.get("result_b"):
                                st.image(row["result_b"], use_container_width=True)
                        with cols[3]:
                            if row.get("result_c"):
                                st.image(row["result_c"], use_container_width=True)

                    tc = entry.get('total_cost', {})
                    st.caption(f"Total: ${tc.get('gemini_pro', 0) + tc.get('gemini_flash', 0) + tc.get('flux', 0):.4f}")
                else:
                    # Legacy single comparison display
                    has_prompt_c = 'prompt_c' in entry
                    if has_prompt_c:
                        col1, col2, col3 = st.columns(3)
                    else:
                        col1, col2 = st.columns(2)
                        col3 = None

                    with col1:
                        st.markdown(f"**Prompt A:** {entry['prompt_a']}")
                        if entry.get('result_a'):
                            st.image(entry['result_a'], caption="Result A", use_container_width=True)
                        if entry.get('cost_a'):
                            st.caption(f"Cost: ${entry['cost_a']['total_cost']:.4f}")

                    with col2:
                        st.markdown(f"**Prompt B:** {entry['prompt_b']}")
                        if entry.get('result_b'):
                            st.image(entry['result_b'], caption="Result B", use_container_width=True)
                        if entry.get('cost_b'):
                            st.caption(f"Cost: ${entry['cost_b']['total_cost']:.4f}")

                    if has_prompt_c and col3:
                        with col3:
                            st.markdown(f"**Prompt C:** {entry['prompt_c']}")
                            if entry.get('result_c'):
                                st.image(entry['result_c'], caption="Result C", use_container_width=True)
                                st.caption("Cost: ~$0.05")

        if st.button("🗑️ Clear History"):
            st.session_state.comparison_history = []
            st.rerun()

# Tab 5: Documentation
with tab5:
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

