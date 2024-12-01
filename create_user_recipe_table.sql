CREATE TABLE User_recipe (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    ingredients TEXT NOT NULL,
    recipe_generated TEXT,
    cooking_time TEXT,
    cuisine TEXT,
    nutrition_value TEXT
);
