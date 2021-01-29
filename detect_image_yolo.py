from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont

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

    output_tensor0 = get_output_tensor(interpreter, 0)
    output_tensor1 = get_output_tensor(interpreter, 1)

    class_num = 80
    results = []
    for i in range(6300):
        index, max_per = (-1, -1)
        for j in range(class_num):
            pred = output_tensor1[i][j]
            index, max_per = (j, pred) if pred > max_per else (index, max_per)

        if max_per >= threshold:
            result = {
                'bounding_box' : output_tensor0[index],
                'score' : max_per,
                'class_id' : index
            }
            results.append(result)

    return results


def annotate_objects(image, results, labels, filename='sample.jpg'):
    draw = ImageDraw.Draw(image)

    for result in results:
        box = result['bounding_box']
        draw.rectangle([box[0], box[1], box[2], box[3]])
        draw.text([box[0], box[1]], f"{result['score']}: {labels[result['class_id']]}")

    image.save(filename)

def main():
    labels = load_labels('model/coco_labels_copy.txt')

    interpreter = Interpreter('model/yolov3-320.tflite')
    interpreter.allocate_tensors()

    _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape']

    start = time.time()

    with Image.open('image.jpeg').convert('RGB') as image:
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)

        results = detect_object(interpreter, resized_image, 0.5)

        print(results)

        annotate_objects(image, results, labels, filename="imagenew.jpeg")

        print(f"時間: {time.time() - start}秒")

        for obj in results:
            print(f"{labels[obj['class_id']]}: {obj['score']}")
            # print(f"{obj['class_id']}: {obj['score']}")

if __name__ == '__main__':
    main()