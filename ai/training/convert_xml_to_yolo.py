"""Convert XML annotations to YOLO text format."""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


def xml_to_yolo(xml_file: Path, img_width: int = 640, img_height: int = 480) -> list[str]:
    """
    Convert a single XML annotation to YOLO format.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1 range).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    
    # YOLOv8 expects class_id=0 for 'swimmer' (single class)
    class_id = 0
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        
        try:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
        except (ValueError, TypeError, AttributeError):
            continue
        
        # Convert to YOLO format (normalized center coordinates and size)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Clamp to valid range [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


def convert_directory(xml_dir: Path, output_dir: Path) -> None:
    """Convert all XML files in a directory to YOLO format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(xml_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    
    success_count = 0
    for xml_file in xml_files:
        try:
            yolo_lines = xml_to_yolo(xml_file)
            
            # Save to txt file with same name
            txt_file = output_dir / xml_file.stem
            txt_file = txt_file.with_suffix('.txt')
            
            with open(txt_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"  Converted {success_count}/{len(xml_files)} files")
        except Exception as e:
            print(f"Error converting {xml_file.name}: {e}")
    
    print(f"✓ Conversion complete: {success_count}/{len(xml_files)} successful")


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ai/training/dataset/labels"),
        help="Directory containing XML files (recursively converts train/ and val/)"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("ai/training/dataset/labels"),
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    # Convert train and val directories
    for split in ['train', 'val']:
        xml_dir = args.input_dir / split
        output_dir = args.output_base / split
        
        if xml_dir.exists():
            print(f"\nConverting {split} split...")
            convert_directory(xml_dir, output_dir)
        else:
            print(f"Warning: {xml_dir} not found")


if __name__ == "__main__":
    main()
