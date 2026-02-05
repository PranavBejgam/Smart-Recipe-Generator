import base64
import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
# import pytesseract  # OCR disabled for Streamlit Cloud
import os
import google.generativeai as genai
import re
import hashlib
from db_config import get_db_connection
from table import create_user_recipe_table, alter_user_recipe_table, add_user_recipe, check_user_recipes, create_user_table, add_user, get_user, check_duplicate_recipe, get_user_recipes

# pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # Local OCR only

# ✅ Gemini configuration (Streamlit secrets)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini error: {str(e)}")
        return None

# Load the pre-trained Hugging Face model
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
labels = list(model.config.id2label.values())

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_ingredient(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    input_tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs.logits, dim=1).item()
        return labels[predicted_idx]

"""
# OCR kept for local use only
def detect_text_from_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"Error: The image '{image_path}' does not exist.")
        return None

    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()
"""

def identify_ingredient_from_text(text):
    try:
        prompt = (
            f"Identify the primary cooking ingredient from this text:\n{text}\n"
            "Return only the ingredient name."
        )
        ingredient = ask_gemini(prompt)
        return clean_ingredient(ingredient) if ingredient else None
    except Exception as e:
        st.error(str(e))
        return None

def clean_ingredient(ingredient):
    ingredient = re.findall(r'\b[A-Za-z]+\b', ingredient or "")
    return ' '.join(ingredient).capitalize()

def detect_ingredient(image_path):
    detected_ingredients = []

    # OCR disabled for deployment
    # text = detect_text_from_image(image_path)
    # if text:
    #     ingredient_from_text = identify_ingredient_from_text(text)
    #     if ingredient_from_text:
    #         detected_ingredients.append(ingredient_from_text)

    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error: Unable to read image '{image_path}'.")
        return None

    ingredient_from_image = classify_ingredient(image)
    detected_ingredients.append(clean_ingredient(ingredient_from_image))

    return detected_ingredients
# Function to set the background image
def set_bg_hack(main_bg):
    main_bg_ext = "jpg"

    if not os.path.exists(main_bg):
        st.warning("Background image not found.")
        return

    encoded = base64.b64encode(open(main_bg, "rb").read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{encoded});
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3, h4, h5, h6, p, li {{
            color: black;
        }}
        .italic {{
            font-style: italic;
        }}
        .stButton > button {{
            background-color: white;
            color: black;
            border: 2px solid black;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{main_bg_ext};base64,{encoded});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Use relative path for cloud deployment
set_bg_hack("background.jpg")


def display_recipe(title, cooking_time, cuisine, nutrition_value, all_ingredients, instructions):
    st.subheader(f"🍽️ {title}")

    st.markdown("<h3 class='italic'><b>Instructions</b></h3>", unsafe_allow_html=True)

    if instructions:
        st.write('\n'.join(instructions.split('\n')))
    else:
        st.error("Recipe generation failed.")

    formatted_recipe = f"""
    Generated Recipe: {title}

    Complete Recipe Instructions:

    Cooking Time: {cooking_time}

    Cuisine Type: {cuisine}

    Nutrition Value (per serving):
    {nutrition_value}

    Ingredients:
    {', '.join(all_ingredients)}

    Instructions:
    {instructions}
    """

    return formatted_recipe


# ✅ Gemini recipe generator
def get_chatgpt_response(ingredients, mode):
    ingredient_list = ', '.join(ingredients)

    if mode == "Veg":
        prompt = (
            f"Generate a vegetarian recipe using: {ingredient_list}. "
            "Include title, cooking time, cuisine, nutrition, and full instructions."
        )
    else:
        prompt = (
            f"Generate a non-vegetarian recipe using: {ingredient_list}. "
            "Include title, cooking time, cuisine, nutrition, and full instructions."
        )

    return ask_gemini(prompt)


# ✅ Gemini recipe parser
def parse_recipe(full_recipe):
    if not full_recipe:
        return None, None, None, None, None

    lines = full_recipe.split('\n')
    title = lines[0]

    info_prompt = (
        "Extract cooking time, cuisine type, and nutrition kcal from this recipe:\n"
        f"{full_recipe}"
    )

    info = ask_gemini(info_prompt)

    cooking_time = "30 minutes"
    cuisine = "Mixed"
    nutrition_value = "400 kcal"

    if info:
        for line in info.split('\n'):
            if "time" in line.lower():
                cooking_time = line
            if "cuisine" in line.lower():
                cuisine = line
            if "nutrition" in line.lower():
                nutrition_value = line

    instructions = "\n".join(lines[1:])
    return title, cooking_time, cuisine, nutrition_value, instructions


def login_page():
    st.title('Smart Recipe Generator')
    create_user_recipe_table()
    alter_user_recipe_table()
    create_user_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'new_user' not in st.session_state:
        st.session_state.new_user = ''
        st.session_state.new_password = ''
        st.session_state.recipe_generated = False
        st.session_state.show_buttons = False

    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sign Up":
        st.subheader("Create a New Account")

        new_user = st.text_input("Username", key="sign_up_user")
        new_password = st.text_input("Password", type='password', key="sign_up_password")
        phone_no = st.text_input("Phone Number")
        email = st.text_input("Email")
        profile_picture = st.file_uploader("Profile Picture", type=["jpg", "jpeg", "png"])

        if st.button("Sign Up"):
            if new_user and new_password and email:
                if not re.match(r"^[6-9]\d{9}$", phone_no):
                    st.warning("Invalid phone number.")
                elif "@" not in email:
                    st.warning("Invalid email.")
                else:
                    hashed_password = hash_password(new_password)
                    profile_picture_path = None

                    if profile_picture is not None:
                        os.makedirs("profile_pictures", exist_ok=True)
                        profile_picture_path = os.path.join("profile_pictures", new_user + "_" + profile_picture.name)
                        with open(profile_picture_path, "wb") as f:
                            f.write(profile_picture.getbuffer())

                    add_user(new_user, phone_no, email, hashed_password, profile_picture_path)
                    st.success("Account created. Login now.")
            else:
                st.warning("Fill all required fields.")

    elif choice == "Login":
        st.subheader("Login to Your Account")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type='password', key="login_password")

        if st.button("Login"):
            hashed_password = hash_password(password)
            user = get_user(username)

            if user and user[4] == hashed_password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.phone_no = user[2]
                st.session_state.email = user[3]
                st.session_state.profile_picture = user[5]
                st.success(f"Welcome {username}")
            else:
                st.warning("Invalid credentials")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def main_page():
    st.markdown(
        """
        <style>
        .small-title {
            font-size: 39px;
            color: brown;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="small-title">🌟 Welcome to the Recipe Generator 🌟</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🚪 Logout"):
            logout()

    st.sidebar.header(f"👋 Welcome, {st.session_state.username}!")
    if st.session_state.profile_picture:
        st.sidebar.image(st.session_state.profile_picture, width=150)
    st.sidebar.markdown(f"""
    **📞 Phone Number:**  
    {st.session_state.phone_no}

    **✉️ Email:**  
    {st.session_state.email}
    """)

    tab1, tab2 = st.tabs(["Recipe Generator", "Saved Recipes"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload Images of Ingredients (One Ingredient per Image)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="upload_images"
        )

        mode = st.radio("Select Mode:", ("Veg", "Non-Veg"))

        if st.button('🧾 Detect Ingredients'):
            st.session_state.recipe_generated = False
            st.session_state.show_buttons = False
            st.session_state.mode = mode

            all_ingredients = []

            for uploaded_file in uploaded_files:
                os.makedirs("temp_images", exist_ok=True)

                image_path = os.path.join("temp_images", uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                detected_ingredients = detect_ingredient(image_path)

                if detected_ingredients:
                    for ingredient in detected_ingredients:
                        ingredient_name = ingredient.split()[-1]
                        all_ingredients.append(ingredient_name.strip())

                os.remove(image_path)

            if not all_ingredients:
                st.warning("⚠️ No valid ingredients detected.")
                return

            st.write("### 🥕 Detected Ingredients:")
            for ingredient in all_ingredients:
                st.write(f"- **{ingredient}**")

            full_recipe = get_chatgpt_response(all_ingredients, mode)
            title, cooking_time, cuisine, nutrition_value, instructions = parse_recipe(full_recipe)

            formatted_recipe = display_recipe(
                title, cooking_time, cuisine,
                nutrition_value, all_ingredients, instructions
            )

            st.session_state.generated_recipe = {
                'title': title,
                'cooking_time': cooking_time,
                'cuisine': cuisine,
                'nutrition_value': nutrition_value,
                'all_ingredients': all_ingredients,
                'instructions': instructions,
                'formatted_recipe': f"{title}\n\n{instructions}"
            }

            st.session_state.recipe_generated = True
            st.session_state.show_buttons = True

        if st.session_state.show_buttons and st.session_state.recipe_generated:

            if st.button("Store the Recipe"):
                if not check_duplicate_recipe(
                    st.session_state.username,
                    st.session_state.generated_recipe['formatted_recipe']
                ):
                    add_user_recipe(
                        username=st.session_state.username,
                        ingredients=', '.join(st.session_state.generated_recipe['all_ingredients']),
                        recipe_generated=st.session_state.generated_recipe['formatted_recipe'],
                        cooking_time=st.session_state.generated_recipe['cooking_time'],
                        cuisine=st.session_state.generated_recipe['cuisine'],
                        nutrition_value=st.session_state.generated_recipe['nutrition_value']
                    )
                    st.success("Recipe stored successfully!")
                    st.session_state.show_buttons = False
                else:
                    st.warning("This recipe already exists in your database.")

            if st.button("Generate Another Recipe"):

                new_prompt = (
                    f"Generate a different recipe using: "
                    f"{', '.join(st.session_state.generated_recipe['all_ingredients'])}. "
                    f"Suitable for a {st.session_state.mode} meal. "
                    "Include title, cooking time, cuisine, nutrition, and instructions."
                )

                full_recipe = ask_gemini(new_prompt)

                title, cooking_time, cuisine, nutrition_value, instructions = parse_recipe(full_recipe)

                formatted_recipe = f"{title}\n\n{instructions}"

                st.session_state.generated_recipe = {
                    'title': title,
                    'cooking_time': cooking_time,
                    'cuisine': cuisine,
                    'nutrition_value': nutrition_value,
                    'all_ingredients': st.session_state.generated_recipe['all_ingredients'],
                    'instructions': instructions,
                    'formatted_recipe': formatted_recipe
                }

                st.session_state.recipe_generated = True

                display_recipe(
                    title, cooking_time, cuisine,
                    nutrition_value,
                    st.session_state.generated_recipe['all_ingredients'],
                    instructions
                )

    with tab2:
        st.subheader("Your Saved Recipes")

        user_recipes = get_user_recipes(st.session_state.username)

        if user_recipes:
            for recipe_generated in user_recipes:
                recipe_title = recipe_generated.split('\n')[0]
                if st.button(recipe_title):
                    st.session_state.selected_recipe = recipe_generated

            if 'selected_recipe' in st.session_state:
                st.write("### 🍽️ Recipe Details")
                st.write(st.session_state.selected_recipe)
        else:
            st.write("You have no saved recipes.")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.phone_no = None
    st.session_state.email = None
    st.session_state.profile_picture = None
    st.experimental_set_query_params(page=None)


def start_page():
    st.markdown("""
        <style>
        .title { font-size: 50px; text-align: center; color: #FF6347; }
        .subtitle { font-size: 20px; text-align: center; color: #6A5ACD; }
        .description { font-size: 16px; text-align: center; margin: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Smart Recipe Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Culinary Companion</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="description">
        Upload ingredient images and generate recipes instantly.
        </div>
    """, unsafe_allow_html=True)

    if st.button("Get Started"):
        st.session_state.started = True


def main():
    if 'started' not in st.session_state:
        st.session_state.started = False

    if st.session_state.started:
        if st.session_state.get("logged_in", False):
            main_page()
        else:
            login_page()
    else:
        start_page()


if __name__ == "__main__":
    main()
