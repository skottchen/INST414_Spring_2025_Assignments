import json

def process_files(x, y):
    collaborations = []
    BEYONCE_MISSPELLED = "Beyonc\u00e9"
    for _ in range(0, 7):
        with open(f"artists_{x}_to_{y}.json", "r") as file:
            data = json.load(file)
            for item in data:
                if item["artist_name"] == BEYONCE_MISSPELLED:
                    item["artist_name"] = "Beyonce"
                if item["artist_collaborations"]:  # Remove artists who had no collaborations
                    for key in item["artist_collaborations"].copy():
                        if key == BEYONCE_MISSPELLED:
                            item["artist_collaborations"]["Beyonce"] = item["artist_collaborations"].get(
                                BEYONCE_MISSPELLED)
                            item["artist_collaborations"].pop(
                                BEYONCE_MISSPELLED)
                    collaborations.append(item)

            x += 10
            if y == 60:
                y += 8
            else:
                y += 10

    return collaborations


def main():
    x, y = 1, 10
    collaborations = process_files(x, y)

    with open("artist_collaborations.json", "w") as file:
        json.dump(collaborations, file, indent=2)


main()
