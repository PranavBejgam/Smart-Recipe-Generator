# import psycopg2
# from db_config import get_db_connection

# def create_user_recipe_table():
#     conn = get_db_connection()
#     if conn:
#         try:
#             with conn.cursor() as cur:
#                 sql_command = """
#                 CREATE TABLE IF NOT EXISTS "User_recipe" (
#                     id SERIAL PRIMARY KEY,
#                     Username TEXT NOT NULL,
#                     ingredients TEXT,
#                     recipe_generated TEXT,
#                     cooking_time TEXT,
#                     cuisine TEXT,
#                     nutrition_value TEXT,
#                     UNIQUE (Username, ingredients)  -- Ensure combination is unique
#                 );
#                 """
#                 cur.execute(sql_command)
#                 print("Table 'User_recipe' checked/created successfully.")
#             conn.commit()
#         except Exception as error:
#             print(f"Error creating table: {error}")
#         finally:
#             conn.close()

# def check_duplicate_recipe(username, ingredients):
#     conn = get_db_connection()
#     if conn:
#         try:
#             with conn.cursor() as cur:
#                 query = """
#                 SELECT COUNT(*) FROM "User_recipe"
#                 WHERE Username = %s AND ingredients = %s;
#                 """
#                 cur.execute(query, (username, ingredients))
#                 count = cur.fetchone()[0]
#                 print(f"Checking for duplicates: {count} found.")
#                 return count > 0  # Returns True if duplicate exists
#         except Exception as error:
#             print(f"Error checking for duplicates: {error}")
#             return False
#         finally:
#             conn.close()

# def add_user_recipe(username, ingredients, recipe_generated, cooking_time, cuisine, nutrition_value):
#     if check_duplicate_recipe(username, ingredients):
#         print("Duplicate recipe detected. Recipe not added.")
#         return  # Exit if duplicate found

#     conn = get_db_connection()
#     if conn:
#         try:
#             with conn.cursor() as cur:
#                 insert_query = """
#                 INSERT INTO "User_recipe" (Username, ingredients, recipe_generated, cooking_time, cuisine, nutrition_value)
#                 VALUES (%s, %s, %s, %s, %s, %s);
#                 """
#                 cur.execute(insert_query, (username, ingredients, recipe_generated, cooking_time, cuisine, nutrition_value))
#                 print("New user recipe added successfully.")
#             conn.commit()
#         except Exception as error:
#             print(f"Error adding recipe: {error}")
#         finally:
#             conn.close()

# def remove_user_recipe(username):
#     conn = get_db_connection()
#     if conn:
#         try:
#             with conn.cursor() as cur:
#                 delete_query = """
#                 DELETE FROM "User_recipe" WHERE Username = %s;
#                 """
#                 cur.execute(delete_query, (username,))
#                 if cur.rowcount > 0:
#                     print(f"User recipe for '{username}' removed successfully.")
#                 else:
#                     print(f"No recipe found for user '{username}'.")
#             conn.commit()
#         except Exception as error:
#             print(f"Error removing recipe: {error}")
#         finally:
#             conn.close()

# def check_user_recipes():
#     conn = get_db_connection()
#     if conn:
#         try:
#             with conn.cursor() as cur:
#                 select_query = """
#                 SELECT * FROM "User_recipe";
#                 """
#                 cur.execute(select_query)
#                 rows = cur.fetchall()
#                 if rows:
#                     print("Current recipes in 'User_recipe' table:")
#                     for row in rows:
#                         print(row)
#                 else:
#                     print("No recipes found in 'User_recipe' table.")
#         except Exception as error:
#             print(f"Error checking recipes: {error}")
#         finally:
#             conn.close()

import psycopg2
from db_config import get_db_connection

def create_user_recipe_table():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                sql_command = """
                CREATE TABLE IF NOT EXISTS "User_recipe" (
                    id SERIAL PRIMARY KEY,
                    Username TEXT NOT NULL,
                    ingredients TEXT,
                    recipe_generated TEXT,
                    cooking_time TEXT,
                    cuisine TEXT,
                    nutrition_value TEXT,
                    UNIQUE (Username, recipe_generated)  -- Ensure combination is unique
                );
                """
                cur.execute(sql_command)
                print("Table 'User_recipe' checked/created successfully.")
            conn.commit()
        except Exception as error:
            print(f"Error creating table: {error}")
        finally:
            conn.close()

def alter_user_recipe_table():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                sql_command = """
                ALTER TABLE "User_recipe"
                DROP COLUMN IF EXISTS "Recipe_title";
                """
                cur.execute(sql_command)
                print("Table 'User_recipe' altered successfully.")
            conn.commit()
        except Exception as error:
            print(f"Error altering table: {error}")
        finally:
            conn.close()

def create_user_table():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                sql_command = """
                CREATE TABLE IF NOT EXISTS "Users" (
                    id SERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    phone_no TEXT,
                    email TEXT,
                    password TEXT,
                    profile_picture TEXT
                );
                """
                cur.execute(sql_command)
                print("Table 'Users' checked/created successfully.")
            conn.commit()
        except Exception as error:
            print(f"Error creating table: {error}")
        finally:
            conn.close()

def check_duplicate_recipe(username, recipe_generated):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                SELECT COUNT(*) FROM "User_recipe"
                WHERE Username = %s AND recipe_generated = %s;
                """
                cur.execute(query, (username, recipe_generated))
                count = cur.fetchone()[0]
                print(f"Checking for duplicates: {count} found.")
                return count > 0  # Returns True if duplicate exists
        except Exception as error:
            print(f"Error checking for duplicates: {error}")
            return False
        finally:
            conn.close()

def add_user_recipe(username, ingredients, recipe_generated, cooking_time, cuisine, nutrition_value):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                INSERT INTO "User_recipe" (Username, Ingredients, Recipe_generated, Cooking_time, Cuisine, Nutrition_value)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
                cur.execute(query, (username, ingredients, recipe_generated, cooking_time, cuisine, nutrition_value))
                conn.commit()
                print("Recipe added successfully.")
        except Exception as error:
            print(f"Error adding recipe: {error}")
        finally:
            conn.close()

def add_user(username, phone_no, email, password, profile_picture):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                insert_query = """
                INSERT INTO "Users" (username, phone_no, email, password, profile_picture)
                VALUES (%s, %s, %s, %s, %s);
                """
                cur.execute(insert_query, (username, phone_no, email, password, profile_picture))
                print("New user added successfully.")
            conn.commit()
        except Exception as error:
            print(f"Error adding user: {error}")
        finally:
            conn.close()

def get_user(username):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                select_query = """
                SELECT * FROM "Users" WHERE username = %s;
                """
                cur.execute(select_query, (username,))
                user = cur.fetchone()
                return user
        except Exception as error:
            print(f"Error fetching user: {error}")
            return None
        finally:
            conn.close()

def get_user_recipes(username):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                select_query = """
                SELECT recipe_generated FROM "User_recipe" WHERE Username = %s;
                """
                cur.execute(select_query, (username,))
                recipes = cur.fetchall()
                return [recipe[0] for recipe in recipes]
        except Exception as error:
            print(f"Error fetching user recipes: {error}")
            return None
        finally:
            conn.close()

def remove_user_recipe(username):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                delete_query = """
                DELETE FROM "User_recipe" WHERE Username = %s;
                """
                cur.execute(delete_query, (username,))
                if cur.rowcount > 0:
                    print(f"User recipe for '{username}' removed successfully.")
                else:
                    print(f"No recipe found for user '{username}'.")
            conn.commit()
        except Exception as error:
            print(f"Error removing recipe: {error}")
        finally:
            conn.close()

def check_user_recipes():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                select_query = """
                SELECT * FROM "User_recipe";
                """
                cur.execute(select_query)
                rows = cur.fetchall()
                if rows:
                    print("Current recipes in 'User_recipe' table:")
                    for row in rows:
                        print(row)
                else:
                    print("No recipes found in 'User_recipe' table.")
        except Exception as error:
            print(f"Error checking recipes: {error}")
        finally:
            conn.close()
