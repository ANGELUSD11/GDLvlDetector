import json

input_file = "class_indices.json"

def convert_json(input_file):
    try:
        with open(input_file, "r") as file:
            data = json.load(file)
        # cargar json y mostrarlo con indentación correcta
        print(json.dumps(data, indent=4, ensure_ascii=False))
    except:
        print(f"Error al leer el archivo {input_file}")

convert_json(input_file)