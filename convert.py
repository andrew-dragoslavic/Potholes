# filepath: /home/andrewdragoslavic/ECE416/convert_voc_to_yolo.py
import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    with open(output_file, 'w') as f_out:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            # assuming 'pothole' is the only class with id 0
            cls_id = 0  
            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            ymin = float(xmlbox.find("ymin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymax = float(xmlbox.find("ymax").text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h

            f_out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

if __name__ == "__main__":
    annotations_dir = "/home/andrewdragoslavic/ECE416/data/1/annotations"
    labels_dir = "/home/andrewdragoslavic/ECE416/data/1/labels"
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(annotations_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(annotations_dir, file)
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(labels_dir, txt_filename)
            convert_annotation(xml_path, txt_path)
            print(f"Converted {xml_path} to {txt_path}")