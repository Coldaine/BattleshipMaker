import os
import sys
import argparse
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
import base64

# Add project root to sys.path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.image_processing.loader import load_image_pil
from src.image_processing.splitter import split_image_if_needed
from src.image_processing.vectorizer import generate_hull_silhouette_mask
import src.config as config

def extract_top_view(pil_image):
    """Split the image and return the first view (assumed top view)."""
    split_views = split_image_if_needed(pil_image)
    if not split_views:
        raise RuntimeError("No views found in image.")
    return split_views[0]

def try_gemini_hull_mask(analysis_result):
    """Try to extract hull mask from Gemini analysis result (if available)."""
    hull_silhouette_mask_base64 = analysis_result.get("hull_silhouette_mask_base64")
    if hull_silhouette_mask_base64:
        try:
            mask_bytes = base64.b64decode(hull_silhouette_mask_base64)
            hull_mask = Image.open(BytesIO(mask_bytes)).convert('L')
            return hull_mask
        except Exception:
            return None
    return None

def mask_is_poor(mask_img):
    """Heuristic: mask is poor if it is empty, too small, or too noisy."""
    arr = np.array(mask_img)
    area = np.count_nonzero(arr > 128)
    total = arr.size
    if area < 0.01 * total:
        return True
    if area > 0.8 * total:
        return True
    return False

def outermost_contour_mask(pil_img):
    import cv2
    arr = np.array(pil_img.convert('L'))
    _, bin_img = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(arr)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return Image.fromarray(mask)

def visualize_mask_on_image(orig_img, mask_img, out_path):
    vis = orig_img.convert('RGBA').copy()
    mask_rgba = Image.new('RGBA', vis.size, (255,0,0,80))
    mask_bin = mask_img.point(lambda p: 255 if p > 128 else 0)
    vis.paste(mask_rgba, (0,0), mask_bin)
    vis.save(out_path)
    print(f"Saved visualization: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Test top view hull mask extraction.")
    parser.add_argument('--image', type=str, required=True, help='Input ship image path')
    parser.add_argument('--output', type=str, default='output/test_top_view_hull_mask', help='Output directory')
    args = parser.parse_args()

    # Use a dedicated subfolder for all test outputs
    test_output_dir = os.path.join(args.output, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    pil_image = load_image_pil(args.image)
    split_views = split_image_if_needed(pil_image)
    if not split_views:
        print("No views found in image.")
        return

    from src.gemini_api.client import analyze_image_with_gemini
    top_view = None
    top_view_analysis = None
    # Analyze each split view and find the one Gemini says is 'top_view'
    for idx, view in enumerate(split_views):
        analysis_result = analyze_image_with_gemini(view, f"test_view_{idx+1}")
        view_type = analysis_result.get("view_type", "").lower() if analysis_result else ""
        if view_type == "top_view":
            top_view = view
            top_view_analysis = analysis_result
            break
    if top_view is None:
        print("No top view found by Gemini in this image. Exiting.")
        return
    top_view.save(os.path.join(test_output_dir, 'top_view.png'))
    analysis_result = top_view_analysis

    # --- Method 1: Gemini mask (if available) and bounding boxes ---
    gemini_mask = None
    gemini_bboxes = {}
    gemini_mask_path = os.path.join(test_output_dir, 'hull_mask_gemini.png')
    gemini_bbox_overlay_path = os.path.join(test_output_dir, 'gemini_bboxes_overlay.png')
    gemini_bbox_image_path = os.path.join(test_output_dir, 'gemini_bboxes_only.png')
    gemini_bbox_diag_path = os.path.join(test_output_dir, 'gemini_bboxes_diag.png')
    if analysis_result:
        # Save all bounding boxes to a JSON file
        for comp in ["hull", "superstructure", "turrets", "funnels", "bridge_or_tower"]:
            if comp in analysis_result:
                gemini_bboxes[comp] = analysis_result[comp]
        import json
        with open(os.path.join(test_output_dir, 'gemini_bboxes.json'), 'w', encoding='utf-8') as f:
            json.dump(gemini_bboxes, f, indent=2, ensure_ascii=False)
        with open(os.path.join(test_output_dir, 'gemini_analysis_raw.json'), 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"Saved Gemini bounding boxes JSON: {os.path.join(test_output_dir, 'gemini_bboxes.json')}")
        print(f"Saved Gemini raw analysis JSON: {os.path.join(test_output_dir, 'gemini_analysis_raw.json')}")
        # Only draw bounding boxes for turrets
        color_map = {
            'turrets': 'green',
        }
        vis_img = top_view.convert('RGB').copy()
        draw = ImageDraw.Draw(vis_img)
        turret_list = analysis_result.get('turrets', [])
        print(f"Image size: {top_view.size}")
        for idx, item in enumerate(turret_list):
            if isinstance(item, dict) and 'bounding_box_2d' in item:
                bbox = item['bounding_box_2d']
                print(f"Turret {idx+1} bbox: {bbox}")
                label = f"turret_{idx+1}"
                draw.rectangle(bbox, outline=color_map['turrets'], width=3)
                draw.text((bbox[0], bbox[1]), label, fill=color_map['turrets'])
        vis_img.save(gemini_bbox_overlay_path)
        print(f"Saved Gemini bounding box overlay (turrets only): {gemini_bbox_overlay_path}")
        # Export just the bounding boxes (no image, white background)
        bbox_img = Image.new('RGB', top_view.size, (255,255,255))
        bbox_draw = ImageDraw.Draw(bbox_img)
        for idx, item in enumerate(turret_list):
            if isinstance(item, dict) and 'bounding_box_2d' in item:
                bbox = item['bounding_box_2d']
                bbox_draw.rectangle(bbox, outline=color_map['turrets'], width=3)
        bbox_img.save(gemini_bbox_image_path)
        print(f"Saved Gemini bounding boxes only image (turrets only): {gemini_bbox_image_path}")
        # Diagnostic: draw all raw turret boxes in red
        diag_img = top_view.convert('RGB').copy()
        diag_draw = ImageDraw.Draw(diag_img)
        for idx, item in enumerate(turret_list):
            if isinstance(item, dict) and 'bounding_box_2d' in item:
                bbox = item['bounding_box_2d']
                diag_draw.rectangle(bbox, outline='red', width=2)
        diag_img.save(gemini_bbox_diag_path)
        print(f"Saved diagnostic Gemini turret bounding boxes: {gemini_bbox_diag_path}")
        # Hull mask
        if analysis_result.get("hull_silhouette_mask_base64"):
            mask_bytes = base64.b64decode(analysis_result["hull_silhouette_mask_base64"])
            gemini_mask = Image.open(BytesIO(mask_bytes)).convert('L')
            gemini_mask.save(gemini_mask_path)
            print(f"Saved Gemini hull mask: {gemini_mask_path}")

    # --- Method 2: OpenCV/Vectorizer mask ---
    hull_mask = generate_hull_silhouette_mask(top_view, 'top_view')
    hull_mask_path = os.path.join(test_output_dir, 'hull_mask_opencv.png')
    if hull_mask is not None:
        hull_mask.save(hull_mask_path)
        print(f"Saved OpenCV/vectorizer hull mask: {hull_mask_path}")
    else:
        print("OpenCV/vectorizer hull mask extraction failed.")

    # --- Method 3: Outermost contour fallback ---
    fallback_mask = outermost_contour_mask(top_view)
    fallback_mask_path = os.path.join(test_output_dir, 'hull_mask_outer_contour.png')
    if fallback_mask is not None:
        fallback_mask.save(fallback_mask_path)
        print(f"Saved outermost contour fallback mask: {fallback_mask_path}")
    else:
        print("Outermost contour fallback mask extraction failed.")

    # --- Visualizations for all methods ---
    def vis_mask(mask_img, name):
        if mask_img is not None:
            out_path = os.path.join(test_output_dir, f'{name}_overlay.png')
            visualize_mask_on_image(top_view, mask_img, out_path)
    vis_mask(gemini_mask, 'hull_mask_gemini')
    vis_mask(hull_mask, 'hull_mask_opencv')
    vis_mask(fallback_mask, 'hull_mask_outer_contour')

    print(f"All outputs saved to {test_output_dir}")

if __name__ == '__main__':
    main()
