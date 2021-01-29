from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont

import argparse
import numpy as np
import re
import time

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    
    return labels

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_object(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(image, results, labels, filename='sample.jpg'):
    draw = ImageDraw.Draw(image)
    size = image.size

    for result in results:
        ymin, xmin, ymax, xmax = result['bounding_box']
        
        xmin = int(xmin * size[0])
        ymin = int(ymin * size[1])
        xmax = int(xmax * size[0])
        ymax = int(ymax * size[1])

        draw.rectangle([xmin, ymin, xmax, ymax])
        draw.text([xmin, ymin], f"{result['score']}: {labels[result['class_id']]}")

    image.save(filename)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True
    )

    args = parser.parse_args()

    labels = load_labels('model/coco_labels.txt')

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()

    _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape']

    start = time.time()

    with Image.open('交差点.jpg').convert('RGB') as image:
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)

        results = detect_object(interpreter, resized_image, 0.4)

        annotate_objects(image, results, labels, filename="imagecross.jpeg")

        print(f"時間: {time.time() - start}秒")

        for obj in results:
            print(f"{labels[obj['class_id']]}: {obj['score']}")
            # print(f"{obj['class_id']}: {obj['score']}")

if __name__ == '__main__':
    main()