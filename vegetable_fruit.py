
import base64
import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
import pytesseract
import os
import openai
from dotenv import load_dotenv
import re  # Importing the 're' module for regular expressions
import hashlib
from db_config import get_db_connection
from table import create_user_recipe_table, alter_user_recipe_table, add_user_recipe, check_user_recipes, create_user_table, add_user, get_user, check_duplicate_recipe, get_user_recipes

# Initialize pytesseract
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

# Load environment variables
load_dotenv('open_ai.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the pre-trained Hugging Face model
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
labels = list(model.config.id2label.values())

# Image preprocessing for the classifier
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

def detect_text_from_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"Error: The image '{image_path}' does not exist.")
        return None

    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def identify_ingredient_from_text(text):
    try:
        prompt = (
            f"Identify the primary ingredient based on the following product description text: '{text}'. "
            "Return only the main ingredient or ingredients relevant for cooking."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant that extracts ingredients from product descriptions."},
                      {"role": "user", "content": prompt}]
        )
        ingredient = response.choices[0].message['content'].strip()
        return clean_ingredient(ingredient)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def clean_ingredient(ingredient):
    ingredient = re.findall(r'\b[A-Za-z]+\b', ingredient)
    return ' '.join(ingredient).capitalize()

def detect_ingredient(image_path):
    detected_ingredients = []

    text = detect_text_from_image(image_path)
    if text:
        ingredient_from_text = identify_ingredient_from_text(text)
        if ingredient_from_text:
            detected_ingredients.append(ingredient_from_text)

    if not detected_ingredients:
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Error: Unable to read the image '{image_path}'.")
            return None

        ingredient_from_image = classify_ingredient(image)
        detected_ingredients.append(clean_ingredient(ingredient_from_image))

    return detected_ingredients

# Function to set the background image
def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
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
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
            background-position: center;
        }}
        [data-testid="stSidebar"] [aria-expanded="false"] > div:first-child {{
            background-color: black;  /* Close button color */
        }}
        [data-testid="stSidebar"] [aria-expanded="false"] > div:first-child::before {{
            color: black;  /* Inside button color */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (use your local image file name here)
set_bg_hack("/Users/pranavbejgam/Desktop/Smart Recipe/wooden-frame-with-green-plant-it_889056-159057.jpg")

def display_recipe(title, cooking_time, cuisine, nutrition_value, all_ingredients, instructions):
    st.subheader(f"🍽️ {title}")

    st.markdown("<h3 class='italic'><b>Instructions</b></h3>", unsafe_allow_html=True)
    st.write('\n'.join(instructions.split('\n')))

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

def get_chatgpt_response(ingredients, mode):
    try:
        ingredient_list = ', '.join(ingredients)
        if mode == "Veg":
            prompt = (
                f"Generate a vegetarian recipe using the following ingredients: {ingredient_list}. "
                "Ensure the recipe is purely vegetarian and uses these ingredients along with common items like salt, pepper, and spices. "
                "Include the recipe title, cooking time, cuisine type, nutrition value, and complete recipe instructions."
            )
        else:  # Non-Veg
            prompt = (
                f"Generate a non-vegetarian recipe using the following ingredients: {ingredient_list}. "
                "Ensure the recipe includes at least one non-vegetarian ingredient like chicken, fish, or meat, and uses these ingredients along with common items like salt, pepper, and spices. "
                "Include the recipe title, cooking time, cuisine type, nutrition value, and complete recipe instructions."
            )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful recipe generator assistant."},
                      {"role": "user", "content": prompt}]
        )
        full_recipe = response.choices[0].message['content'].strip()
        return full_recipe
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def parse_recipe(full_recipe):
    try:
        lines = full_recipe.split('\n')
        title = lines[0]
        prompt = (
            f"Analyze the following recipe and provide cooking time, cuisine type, "
            f"and nutrition value in kcal:\n\n{full_recipe}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Extract recipe details."},
                      {"role": "user", "content": prompt}]
        )
        additional_info = response.choices[0].message['content'].strip().split('\n')
        cooking_time = next((line for line in additional_info if "Cooking time" in line), "30 minutes")
        cuisine = next((line for line in additional_info if "Cuisine" in line), "Mixed")
        nutrition_value = next((line for line in additional_info if "Nutrition" in line), "400 kcal")
        instructions = "\n".join(lines[1:])
        return title, cooking_time, cuisine, nutrition_value, instructions
    except Exception as e:
        st.error(f"Error parsing recipe: {str(e)}")
        return None, None, None, None, None

def login_page():
    st.title('Smart Recipe Generator')
    create_user_recipe_table()
    alter_user_recipe_table()  # Ensure the table is altered to include Recipe_title
    create_user_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'new_user' not in st.session_state:
        st.session_state.new_user = ''
        st.session_state.new_password = ''
        st.session_state.recipe_generated = False  # Initialize the flag for recipe generation
        st.session_state.show_buttons = False  # Initialize the button visibility flag

    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sign Up":
        st.subheader("Create a New Account")
        if 'sign_up_user' not in st.session_state:
            st.session_state.sign_up_user = ''
        if 'sign_up_password' not in st.session_state:
            st.session_state.sign_up_password = ''
        
        new_user = st.text_input("Username", key="sign_up_user")
        new_password = st.text_input("Password", type='password', key="sign_up_password")
        phone_no = st.text_input("Phone Number")
        email = st.text_input("Email")
        profile_picture = st.file_uploader("Profile Picture", type=["jpg", "jpeg", "png"], key="profile_pic")

        if st.button("Sign Up"):
            if new_user and new_password and email:
                if not re.match(r"^[6-9]\d{9}$", phone_no):
                    st.warning("Phone number should start with 6, 7, 8, or 9 and should be 10 digits long.")
                elif "@" not in email or not email.endswith(".com"):
                    st.warning("Email should contain '@' and '.com'.")
                else:
                    hashed_password = hash_password(new_password)
                    profile_picture_path = None
                    if profile_picture is not None:
                        if not os.path.exists("profile_pictures"):
                            os.makedirs("profile_pictures")
                        profile_picture_path = os.path.join("profile_pictures", new_user + "_" + profile_picture.name)
                        with open(profile_picture_path, "wb") as f:
                            f.write(profile_picture.getbuffer())
                    add_user(new_user, phone_no, email, hashed_password, profile_picture_path)
                    st.success("You have successfully created an account.")
                    st.info("Go to Login Menu to login.")
            else:
                st.warning("Please fill out all required fields.")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type='password', key="login_password")

        if st.button("Login"):
            if username and password:
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
                    st.warning("Invalid Username or Password")
            else:
                st.warning("Please enter your username and password.")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def main_page():
    st.markdown(
        """
        <style>
        .small-title {
            font-size: 39px;
            color: brown;  /* Change title color to brown */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="small-title">🌟 Welcome to the Recipe Generator 🌟</h1>', unsafe_allow_html=True)

    # Create a layout for the top right corner button
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🚪 Logout"):
            logout()

    # Enhanced sidebar user details
    st.sidebar.header(f"👋 Welcome, {st.session_state.username}!")
    if st.session_state.profile_picture:
        st.sidebar.image(st.session_state.profile_picture, width=150)
    st.sidebar.markdown(f"""
    **📞 Phone Number:**  
    {st.session_state.phone_no}

    **✉️ Email:**  
    {st.session_state.email}
    """)

    # Tab layout for Recipe Generator and Saved Recipes
    tab1, tab2 = st.tabs(["Recipe Generator", "Saved Recipes"])

    with tab1:
        # Step 1: Upload Images
        uploaded_files = st.file_uploader("Upload Images of Ingredients (One Ingredient per Image)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="upload_images")

        # Step 2: Select Veg or Non-Veg
        mode = st.radio("Select Mode:", ("Veg", "Non-Veg"))

        if st.button('🧾 Detect Ingredients'):
            # Reset state flags for recipe generation
            st.session_state.recipe_generated = False
            st.session_state.show_buttons = False
            st.session_state.mode = mode

            all_ingredients = []
            for uploaded_file in uploaded_files:
                if not os.path.exists("temp_images"):
                    os.makedirs("temp_images")
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
                st.session_state.show_buttons = False  # Ensure buttons are hidden if no ingredients are detected
                return

            st.write("### 🥕 Detected Ingredients:")
            for ingredient in all_ingredients:
                st.write(f"- **{ingredient}**")

            full_recipe = get_chatgpt_response(all_ingredients, mode)
            title, cooking_time, cuisine, nutrition_value, instructions = parse_recipe(full_recipe)

            # Display the full recipe for the user
            formatted_recipe = display_recipe(title, cooking_time, cuisine, nutrition_value, all_ingredients, instructions)

            # Store title and instructions in session state for saving
            st.session_state.generated_recipe = {
                'title': title,
                'cooking_time': cooking_time,
                'cuisine': cuisine,
                'nutrition_value': nutrition_value,
                'all_ingredients': all_ingredients,
                'instructions': instructions,
                'formatted_recipe': f"{title}\n\n{instructions}"  # Include the title with the instructions
            }

            st.session_state.recipe_generated = True
            st.session_state.show_buttons = True

        # Display buttons only after a recipe has been generated
        if st.session_state.show_buttons and st.session_state.recipe_generated:
            if st.button("Store the Recipe"):
                if not check_duplicate_recipe(st.session_state.username, st.session_state.generated_recipe['formatted_recipe']):
                    add_user_recipe(
                        username=st.session_state.username,
                        ingredients=', '.join(st.session_state.generated_recipe['all_ingredients']),
                        recipe_generated=st.session_state.generated_recipe['formatted_recipe'],
                        cooking_time=st.session_state.generated_recipe['cooking_time'],
                        cuisine=st.session_state.generated_recipe['cuisine'],
                        nutrition_value=st.session_state.generated_recipe['nutrition_value']
                    )
                    st.success("Recipe stored successfully!")
                    st.session_state.show_buttons = False  # Hide buttons after storing the recipe
                else:
                    st.warning("This recipe already exists in your database.")

            if st.button("Generate Another Recipe"):
                # Generate another recipe using the same ingredients and the same mode
                new_prompt = (
                    f"Generate a different recipe using the following ingredients: {', '.join(st.session_state.generated_recipe['all_ingredients'])}. "
                    f"Ensure the recipe is suitable for a {st.session_state.mode} meal. "
                    "Include the recipe title, cooking time, cuisine type, nutrition value, and complete recipe instructions."
                )
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful recipe generator assistant."},
                            {"role": "user", "content": new_prompt}
                        ]
                    )
                    full_recipe = response.choices[0].message['content'].strip()
                    title, cooking_time, cuisine, nutrition_value, instructions = parse_recipe(full_recipe)
                    formatted_recipe = f"{title}\n\n{instructions}"  # Include the title with the instructions

                    # Update session state with new recipe
                    st.session_state.generated_recipe = {
                        'title': title,
                        'cooking_time': cooking_time,
                        'cuisine': cuisine,
                        'nutrition_value': nutrition_value,
                        'all_ingredients': st.session_state.generated_recipe['all_ingredients'],
                        'instructions': instructions,
                        'formatted_recipe': formatted_recipe
                    }

                    st.session_state.recipe_generated = True  # Keep the flag set for generating another recipe

                    # Display the new recipe
                    display_recipe(title, cooking_time, cuisine, nutrition_value, st.session_state.generated_recipe['all_ingredients'], instructions)

                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")

    with tab2:
        st.subheader("Your Saved Recipes")
        user_recipes = get_user_recipes(st.session_state.username)
        if user_recipes:
            for i, recipe_generated in enumerate(user_recipes):
                # Extract the first line which contains the title
                recipe_title = recipe_generated.split('\n')[0]
                if st.button(recipe_title):
                    st.session_state.selected_recipe = recipe_generated

            if 'selected_recipe' in st.session_state:
                recipe = st.session_state.selected_recipe
                st.write("### 🍽️ Recipe Details")
                st.write(recipe)
        else:
            st.write("You have no saved recipes.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.phone_no = None
    st.session_state.email = None
    st.session_state.profile_picture = None
    st.experimental_set_query_params(page=None)  # Clear query params on logout

def start_page():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Gone&type=face');

        .title {
            font-size: 50px;
            font-family: 'Gone Typeface', sans-serif;
            color: #FF6347;  /* Tomato color */
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #6A5ACD;  /* Slate blue color */
        }
        .description {
            font-size: 16px;
            text-align: center;
            color: black;  /* Text color */
            margin: 20px;
        }
        .center-btn {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .center-btn button {
            background-color: black;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">Smart Recipe Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Culinary Companion</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="description">
            Welcome to the Smart Recipe Generator! Our tool helps you create delicious recipes 
            based on the ingredients you have. Upload images of your ingredients, and let our 
            intelligent system suggest the best recipes for you. Whether you're a vegetarian 
            or a non-vegetarian, we've got you covered. Let's get cooking!
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="center-btn">', unsafe_allow_html=True)
    if st.button("Get Started", key="start_button"):
        st.session_state.started = True
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    if 'started' not in st.session_state:
        st.session_state.started = False

    if st.session_state.started:
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
            st.session_state.recipe_generated = False  # Initialize the flag for recipe generation
            st.session_state.show_buttons = False

        if st.session_state.get("logged_in", False):
            main_page()
        else:
            login_page()
    else:
        start_page()

if __name__ == "__main__":
    main()
