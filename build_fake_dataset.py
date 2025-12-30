import json
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

if __name__ == "__main__":
    path_des = "./fake_8k_data.json"
    DATA_NUM = 512
    MAX_LENGTH = 8192

    width, height = 1024, 1024
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_array)
    random_image.save("./fake_1024_1024_pic.jpg")

    tokenizer = AutoTokenizer.from_pretrained("/home/data/sxy/models/Qwen/Qwen3-VL-30B-A3B-Instruct")
    seed_data = "This image captures a stunning tropical beach scene that feels like a postcard-perfect paradise; the foreground is dominated by crystal-clear, turquoise waters that gently fade from a bright, almost neon aquamarine near the shore to a deeper shade of blue further out, with subtle ripples glinting under the sunlight—hinting at calm, warm seas ideal for swimming or snorkeling, while stretching along the right side is a soft, powdery white sand beach where the shoreline curves gently into the distance, lining the edge of the sand is a lush, dense cluster of tall palm trees whose fronds (implied by their relaxed posture) sway and cast dappled shadows on the ground below, and nestled beneath the palms are a few low-profile beach chairs in warm orange tones and shaded areas that suggest a quiet, inviting spot for relaxation; above, the sky is a vibrant, cloudless blue (save for a few fluffy, white cumulus clouds drifting lazily), amplifying the scene’s bright, cheerful atmosphere, and in the far background, the horizon blends seamlessly between the deep blue sea and the sky with a tiny, distant speck that might be a boat, adding a subtle touch of life to the otherwise serene landscape, so the overall impression is one of tranquility, natural beauty, and tropical escape—perfect for a peaceful vacation retreat."

    text = seed_data
    while len(tokenizer.encode(text)) < MAX_LENGTH:
        text += ' ' + seed_data

    mock_data = {
        "messages": [
            {
                "content": "<image>\nPlease describe the image.",
                "role": "assistant"
            }
        ],
        "images": [
            "fake_1024_1024_pic.jpg"
        ]
    }

    all_data = []
    for _ in range(DATA_NUM):
        all_data.append(mock_data)

    with open(path_des, "w", encoding='utf-8') as fw:
        json.dump(all_data, fw, ensure_ascii=False, indent=4)