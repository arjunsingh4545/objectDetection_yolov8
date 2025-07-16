import rasterio
import argparse
import os
import json
import shapely

def polygon_to_yolo_bbox(polygon, img_width, img_height):
    minx, miny, maxx, maxy = polygon.bounds
    bbox = [
        (minx + maxx) / 2.0,
        (miny + maxy) / 2.0,
        maxx - minx,
        maxy - miny,
    ]
    return list(map(lambda v, dim: v / dim, bbox, [img_width, img_height, img_width, img_height]))
"""
def main(geojson_path , image_dir ,output_dir):
    os.makedirs(output_dir , exist_ok=True)

    with open(geojson_path , 'r') as f:
        data = json.load(f)


    for feature in data['features']:
        props = feature['properties']
        img_name = props['image_id']
        class_id = int(props['type_id'])  # already 0–59 in xView


        img_path = os.path.join(image_dir , img_name)
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        try:
            geom = shapely.geometry.shape(feature['geometry'])
            if not geom.is_valid:
                print(f"⚠️ Invalid polygon in {img_name}, skipping.")
                continue
        except Exception as e:
            print(f"❌ Failed to parse polygon: {e}")
            continue

        with rasterio.open(img_path) as img:
            width , height = img.width , img.height

        bbox = polygon_to_yolo_bbox(geom , width, height)

        if any(v < 0 or v > 1 for v in bbox):
            print(f"⚠️ Skipping bbox out of bounds in {img_name}")
            continue


        label_file = os.path.join(output_dir, img_name.replace('.tif', '.txt'))
        with open(label_file, 'a') as out:
            out.write(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n")
"""
def main(geojson_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    for feature in data['features']:
        props = feature['properties']
        img_name = props['image_id']
        class_id = int(props['type_id'])

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        # === ✅ Use bounds_imcoords if available ===
        if 'bounds_imcoords' in props and props['bounds_imcoords']:
            try:
                coords = list(map(int, props['bounds_imcoords'].split(',')))
                if len(coords) != 4:
                    print(f"⚠️ Malformed bounds_imcoords in {img_name}, skipping.")
                    continue
                """
                minx, miny, maxx, maxy = coords
                bbox = [
                    (minx + maxx) / 2.0,
                    (miny + maxy) / 2.0,
                    maxx - minx,
                    maxy - miny,
                ]
                with rasterio.open(img_path) as img:
                    width, height = img.width, img.height
                bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                """
                minx , miny , maxx , maxy = coords
                with rasterio.open(img_path) as img:
                    width, height = img.width, img.height
                minx = max(0, minx)
                miny = max(0, miny)
                maxx = min(width, maxx)
                maxy = min(height, maxy)
                if minx >= maxx or miny >= maxy:
                    print(f"⚠️ Clamped bbox is invalid in {img_name}, skipping.")
                    continue
                bbox = [
                    (minx + maxx) / 2.0,
                    (miny + maxy) / 2.0,
                    maxx - minx,
                    maxy - miny,
                ]
                bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]

            except Exception as e:
                print(f"❌ Error parsing bounds_imcoords in {img_name}: {e}")
                continue
        else:
            # === 🔄 Fallback to geometry polygon ===
            try:
                geom = shapely.geometry.shape(feature['geometry'])
                if not geom.is_valid:
                    print(f"⚠️ Invalid polygon in {img_name}, skipping.")
                    continue
                with rasterio.open(img_path) as img:
                    width, height = img.width, img.height
                bbox = polygon_to_yolo_bbox(geom, width, height)
            except Exception as e:
                print(f"❌ Failed to parse geometry in {img_name}: {e}")
                continue

        # === 🧪 Final sanity check ===
        if any(v < 0 or v > 1 for v in bbox):
            print(f"⚠️ Skipping bbox out of bounds in {img_name}")
            print(f"   bbox: {bbox}")
            print(f"   width: {width}, height: {height}")
            print(f"   coords: {props.get('bounds_imcoords')}")
            continue

        # === ✏️ Write YOLO label ===
        label_file = os.path.join(output_dir, img_name.replace('.tif', '.txt'))
        with open(label_file, 'a') as out:
            out.write(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n")
        print(f"✅ Saved label for {img_name}")



if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Convert xView GeoJSON to YOLO format")
    parser.add_argument("--geojson", type=str, required=True, help="Path to train_labels.geojson")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image .tif files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for YOLO .txt labels")
    args = parser.parse_args()
    main(args.geojson, args.img_dir, args.output_dir)
    """
    geojson_path = None
    img_dir = None
    output_dir = None
    main(geojson_path , img_dir, output_dir)
