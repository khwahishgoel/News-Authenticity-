import os

def load_dataset(data_path):
    texts = []
    labels = []

    categories = {
        "fake": 0,
        "real": 1
    }

    for category, label in categories.items():
        folder_path = os.path.join(data_path, category)

        if not os.path.exists(folder_path):
            print(f"❌ Folder missing: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)

    print("✅ Labels loaded:", labels)
    return texts, labels

