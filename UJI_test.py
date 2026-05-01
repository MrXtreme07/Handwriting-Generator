import matplotlib.pyplot as plt


def load_uji_characters(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    characters = []
    current_char = []
    current_stroke = []
    recording = False

    for line in lines:
        line = line.strip()

        if line.startswith(".SEGMENT"):
            # save previous character
            if current_char:
                characters.append(current_char)
            current_char = []

        elif line == ".PEN_DOWN":
            current_stroke = []
            recording = True

        elif line == ".PEN_UP":
            if current_stroke:
                current_char.append(current_stroke)
            recording = False

        elif recording:
            parts = line.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                current_stroke.append((x, y))

    # append last character
    if current_char:
        characters.append(current_char)

    return characters


# 🔧 CHANGE THIS PATH
characters = load_uji_characters("uji+pen+characters/UJIpenchars-w01")

print("Total characters:", len(characters))

char = characters[0]  # first character

plt.figure(figsize=(5,5))

for stroke in char:
    x = [p[0] for p in stroke]
    y = [p[1] for p in stroke]
    plt.plot(x, y)

plt.gca().invert_yaxis()
plt.title("Proper Single Character")
plt.show()