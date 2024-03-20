import os, json

with open("categories.json", "r") as f:
    data = json.load(f)
    categories = data["categories"]
    saved_categories = []
    for c in categories:
        id = c["id"]
        name = c["name"]
        syns = c["synonyms"]
        saved_categories.append({
            "id": id,
            "name": name,
            "synonyms": syns
        })
    
with open("new_categories.json", "w") as f:
    json.dump(saved_categories, f, indent=2)