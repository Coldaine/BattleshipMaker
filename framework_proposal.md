# Battleship 2D to 3D Conversion Project: Framework Proposal

This document outlines the proposed framework for the Battleship 2D to 3D Conversion Project, based on the requirements specified in `Prompt.md`.

## 1. High-Level Architecture

The system will follow a pipeline architecture with distinct stages:

1.  **Setup & Configuration:**
    *   Initialize environment, load configurations (e.g., API keys if externalized, debug flags).
    *   Setup logging.
    *   Initialize caching mechanism.

2.  **Input Acquisition:**
    *   Accept an input image (PNG, JPG) or vector graphic path.
    *   Load the image/vector data.

3.  **Image Pre-processing & Splitting:**
    *   Apply the horizontal splitting heuristic to detect and separate multiple views (top/side) if present.
    *   Each resulting image (or the original if not split) is processed individually.

4.  **View Analysis & Metadata Extraction (Gemini API):**
    *   **Caching Check:** For each image, check if results exist in the cache. If so, use cached data (unless `debug_flag` is set).
    *   **Gemini API Call:** If not cached or debugging, send the image to the Gemini API.
        *   Prompt for: view type (top/side), ship class/name, transcribed text.
        *   Use specified model (`gemini-1.5-flash-latest`) and structured JSON response.
    *   **Cache Update:** Store the API response in the cache.
    *   **Output Directory Creation:** Create output subdirectories based on ship class/name and source filename.
    *   Save transcribed text to a file.

5.  **Raster-to-Vector Conversion & Intermediate Drawing Generation:**
    *   For each identified view (top/side):
        *   Convert the raster image to a vector representation (if not already vector).
        *   Generate the three specified intermediate drawings based on view type:
            *   Hull Shape Only
            *   Structures Except Turrets
            *   Best Detail
        *   Save these drawings in the appropriate output subdirectory.

6.  **3D Mesh Generation (Basic):**
    *   **Hull Generation (Enhanced):**
        *   Use the top-view and side-view "Hull Shape Only" vector drawings.
        *   Apply calibration data (derived from analyzing high-poly models) to inform the 3D hull shape, accounting for depth, curvature, and tapering.
    *   **Superstructure & Turret Generation:**
        *   Use top-view "Structures Except Turrets" and "Best Detail" vector drawings.
        *   Extrude shapes to estimated heights. (Turrets are excluded from some drawings but might be part of "Best Detail" for separate processing if needed, or this step focuses on superstructure first).
    *   Save the generated basic 3D mesh (e.g., in .obj format) in the output subdirectory.

7.  **Logging & Reporting:**
    *   Log all major steps, processed files, Gemini API results, and any errors encountered.
    *   Generate a final run log file.

## 2. Key Python Library Recommendations

*   **Image Splitting & Processing:**
    *   **Pillow (PIL Fork):** `pip install Pillow`
        *   *Justification:* Widely used, good for image manipulation (opening, resizing, cropping, pixel access for the splitting heuristic). Lightweight and easy to use for common image tasks.
    *   **OpenCV (cv2):** `pip install opencv-python`
        *   *Justification:* More powerful for complex image analysis, but Pillow might be sufficient for initial splitting. Could be useful later for more advanced pre-processing or feature detection if needed.

*   **Gemini API Interaction:**
    *   **google-generativeai:** `pip install google-generativeai`
        *   *Justification:* The official Python SDK for Google's Generative AI models, including Gemini. Simplifies API calls and response handling.
    *   **requests:** `pip install requests` (often a dependency, but good to list)
        *   *Justification:* For making HTTP requests if direct API interaction without the SDK is preferred or for other web tasks. The SDK likely uses it underneath.

*   **Raster-to-Vector Conversion:**
    *   **pypotrace:** (Wrapper for Potrace) `pip install pypotrace` (Potrace itself needs to be installed on the system).
        *   *Justification:* Potrace is a well-regarded open-source tool for converting bitmaps into vector graphics (SVG, EPS, etc.). `pypotrace` provides a Python interface. This is good for generating precise line data.
    *   **svgwrite:** `pip install svgwrite`
        *   *Justification:* If line extraction is done via other image processing techniques (e.g., edge detection with OpenCV), `svgwrite` can be used to programmatically create SVG files from these lines.
    *   **scikit-image:** `pip install scikit-image`
        *   *Justification:* Provides algorithms for image segmentation, edge detection, and contour finding, which can be precursors to vectorization. The output of these could be fed into a library like `svgwrite` or a custom vectorization logic.

*   **3D Model Analysis for Calibration (Research & High-Poly Models):**
    *   **Trimesh:** `pip install trimesh`
        *   *Justification:* Excellent for loading, processing, and analyzing 3D meshes (supports various formats like OBJ, STL). Can be used to load existing high-poly models, extract geometric information (vertices, faces, curvatures), and derive calibration parameters.
    *   **NumPy:** `pip install numpy`
        *   *Justification:* Fundamental for numerical operations, essential for handling vertex data, transformations, and any mathematical calculations involved in analyzing models or generating new ones.
    *   **SciPy:** `pip install scipy`
        *   *Justification:* Builds on NumPy and provides many scientific and technical computing routines, including spatial data structures and algorithms (e.g., `scipy.spatial` for KD-trees, convex hulls) that could be useful for analyzing reference models.

*   **3D Mesh Generation (Basic):**
    *   **Trimesh:** (As above)
        *   *Justification:* Can also be used to *create* basic meshes programmatically by defining vertices and faces, performing extrusions, and boolean operations (though booleans can be complex). Suitable for generating the basic hull and superstructure.
    *   **PyVista:** `pip install pyvista`
        *   *Justification:* User-friendly interface to VTK (Visualization Toolkit), powerful for 3D mesh creation, manipulation, and visualization. Good for creating meshes from scratch or from point clouds/lines. Might offer more advanced meshing algorithms if needed.

*   **Jupyter Notebook Environment:**
    *   **jupyterlab, notebook:** `pip install jupyterlab notebook`
        *   *Justification:* As requested, for interactive development, experimentation, and visualization.

*   **Caching:**
    *   **diskcache:** `pip install diskcache`
        *   *Justification:* A simple and effective library for disk-based caching. Useful for storing Gemini API responses to avoid redundant calls.
    *   **Standard library `json` and `os`:** For a simpler file-based cache if `diskcache` is overkill.

## 3. Core Class/Module Structure

Here's a potential outline:

*   `main.py`: Entry point of the script, orchestrates the pipeline.
*   `config.py`: Stores configuration variables (API endpoint, model name, paths, debug flags).
*   `utils/`
    *   `logger.py`: Sets up and provides a logging instance.
    *   `cache.py`: Implements the caching mechanism (e.g., using `diskcache` or file-based).
    *   `file_utils.py`: Helper functions for file/directory operations (creating output structure).
*   `image_processing/`
    *   `loader.py`: Loads images from disk.
    *   `splitter.py`: Implements the multi-view splitting heuristic.
    *   `vectorizer.py`: Handles raster-to-vector conversion (e.g., using Potrace). Contains logic for generating the 6 intermediate drawing types. This module will be complex.
*   `gemini_api/`
    *   `client.py`: Manages interaction with the Gemini API (forming requests, sending, parsing responses). Includes the schema for the structured response.
*   `mesh_generator/`
    *   `hull_generator.py`: Implements the enhanced hull generation logic using top/side views and calibration data.
    *   `superstructure_generator.py`: Implements superstructure/turret extrusion.
    *   `calibration.py`: (Potentially) Contains functions to load and analyze high-poly reference models to extract calibration parameters. This might be a separate utility script initially.
*   `output/`: (Managed by the script, not a module with code)
*   `notebooks/`: (If Jupyter notebooks are kept separate)
    *   `experimentation.ipynb`: Main notebook for development.

**Key Classes (Conceptual):**

*   `BattleshipImage`: Represents an input image, its views, and associated metadata.
*   `GeminiVisionClient`: Handles all communication with the Gemini API.
*   `VectorizationService`: Responsible for converting images to vectors and producing the 6 drawing types.
*   `HullModel`: Represents the 3D hull, with methods for generation based on 2D profiles and calibration.
*   `SuperstructureModel`: Represents 3D superstructure elements.
*   `CacheManager`: Manages storing and retrieving API call results.

## 4. Pseudocode for Critical Algorithms

**a. Multi-View Image Splitting Heuristic:**

```pseudocode
FUNCTION split_image_if_needed(image_path):
    image = load_image(image_path)
    height = image.height
    width = image.width

    middle_y = height / 2
    background_pixel_count = 0
    threshold_background_pixels = width * 0.95 // e.g., 95% of the line must be background

    FOR x FROM 0 TO width - 1:
        pixel_color = image.get_pixel(x, middle_y)
        IF is_background_color(pixel_color): // is_background_color needs a defined range
            background_pixel_count += 1

    IF background_pixel_count >= threshold_background_pixels:
        top_image = image.crop(0, 0, width, middle_y)
        bottom_image = image.crop(0, middle_y, width, height)
        RETURN [top_image, bottom_image]
    ELSE:
        RETURN [image]
```

**b. Core Logic for Interacting with the Gemini API (including caching):**

```pseudocode
FUNCTION get_image_analysis(image_data, image_identifier, force_api_call = DEBUG_FLAG):
    cached_result = cache.get(image_identifier)
    IF cached_result IS NOT NULL AND NOT force_api_call:
        PRINT "Using cached result for:", image_identifier
        RETURN cached_result

    // Prepare API request
    prompt = "Analyze this battleship image. Determine view_type (top or side), ship_identification (class or name), and transcribe_text."
    // Include image_data in the request payload
    // Define responseSchema: { type: OBJECT, properties: { view_type: {type: STRING}, ship_identification: {type: STRING}, transcribed_text: {type: STRING} } }
    // Set generationConfig: { responseMimeType: "application/json" }

    api_response = gemini_client.call_api(
        model=\"gemini-1.5-flash-latest\",
        prompt=prompt,
        image_data=image_data,
        generation_config={responseMimeType: "application/json"},
        response_schema=RESPONSE_SCHEMA_DEFINITION // Define this clearly
    )

    IF api_response.is_successful:
        parsed_data = api_response.get_json_payload()
        cache.set(image_identifier, parsed_data)
        RETURN parsed_data
    ELSE:
        LOG_ERROR "Gemini API call failed for:", image_identifier, api_response.error
        RETURN NULL
```

**c. High-Level Process for Generating "Top view with hull shape only":**

```pseudocode
FUNCTION generate_top_view_hull_only(image_data_top_view):
    // 1. Ensure image_data_top_view is a clean representation (e.g., binary or good contrast)
    //    This might involve pre-processing steps like thresholding.

    // 2. Convert to vector using a library (e.g., Potrace via pypotrace)
    vector_data = vectorize_image(image_data_top_view, mode="silhouette") // Potrace has options for this

    // 3. Post-process vector_data if needed:
    //    - Identify the largest closed path (likely the hull outline).
    //    - Smooth the path if it's too noisy.
    //    - Remove small, irrelevant details or internal paths.
    //    This step is crucial and might require heuristics or image processing knowledge.

    hull_outline_vector = extract_main_silhouette(vector_data)

    // 4. Save the hull_outline_vector as an SVG or other vector format.
    save_vector_drawing(hull_outline_vector, "output_path/top_hull_only.svg")

    RETURN hull_outline_vector // For use in 3D generation
```

**d. Pseudocode for Enhanced Hull Generation:**

```pseudocode
FUNCTION generate_enhanced_hull(top_view_hull_vector, side_view_hull_vector, calibration_data):
    // 1. Align and Scale Vectors:
    //    - Ensure top_view_hull_vector and side_view_hull_vector are consistently scaled and aligned.
    //    - This might involve normalizing them or using known reference points if available.

    // 2. Extract Key Profile Points:
    //    - From top_view_hull_vector: Get points defining the deck outline.
    //    - From side_view_hull_vector: Get points defining the ship's profile (sheer line, keel line).

    // 3. Use Calibration Data:
    //    - calibration_data = {
    //        reference_hull_forms: [analyzed data from high-poly models],
    //        typical_cross_section_shapes: [e.g., U-shape, V-shape at different hull percentages],
    //        bow_stern_tapering_curves: [functions or point sets]
    //      }
    //    - Select or interpolate calibration parameters based on the input ship's overall proportions
    //      (length-to-beam ratio from top/side views).

    // 4. Generate Hull Cross-Sections:
    //    - Iterate along the length of the ship (defined by top_view_hull_vector).
    //    - At each longitudinal station:
    //        - Determine the width from top_view_hull_vector.
    //        - Determine the height/depth profile from side_view_hull_vector (sheer and keel).
    //        - Use calibration_data (e.g., typical_cross_section_shapes, adjusted by current width/depth)
    //          to define the 2D shape of the hull cross-section at this station.
    //        - This involves deforming a template cross-section or generating points based on a parametric formula
    //          informed by the calibration data.

    // 5. Loft Cross-Sections to Create Mesh:
    //    - Create a series of 3D vertices for each cross-section, positioned correctly in 3D space.
    //    - "Loft" these cross-sections by creating faces (quads or triangles) that connect
    //      corresponding vertices between adjacent sections.
    //    - Pay special attention to bow and stern sections, using tapering_curves from calibration_data
    //      to close the hull smoothly.

    // 6. Create Mesh Object:
    //    - hull_mesh = create_mesh_from_vertices_and_faces(all_vertices, all_faces)

    // 7. Post-processing (Optional but likely needed for "basic but informed"):
    //    - Weld vertices, ensure manifold geometry.
    //    - Basic smoothing if required.

    RETURN hull_mesh
```

## 5. Anticipated Challenges & Initial Solutions

*   **Image Quality & Background Noise (Input Handling):**
    *   *Challenge:* Input images may vary in quality, have noise, or complex backgrounds, making splitting and feature extraction difficult.
    *   *Solution:* Implement robust pre-processing: adaptive thresholding, noise reduction filters (e.g., median filter). For splitting, allow configurable background color range.

*   **Splitting Heuristic Accuracy:**
    *   *Challenge:* The simple horizontal line heuristic might fail for unusually composed images or if ships are angled.
    *   *Solution:* Log failures. For a more advanced system, consider basic object detection or segmentation to find ships before attempting to split views. For now, stick to the heuristic and document its limitations.

*   **Gemini API - View Type Ambiguity & Accuracy:**
    *   *Challenge:* Gemini might misclassify views or ship types, especially for obscure or poorly drawn images.
    *   *Solution:* Rely on the structured JSON output. Implement robust parsing. Log Gemini's confidence if available. For critical applications, manual review might be flagged. The caching helps avoid repeated costs for consistently problematic images.

*   **Gemini API - Text Transcription Quality:**
    *   *Challenge:* Transcribed text might be inaccurate or incomplete.
    *   *Solution:* Accept the output as-is for this project's scope. Log it.

*   **Raster-to-Vector Conversion (Detail vs. Noise):**
    *   *Challenge:* Achieving clean vector lines that represent the true ship structure without capturing noise or excessive detail is hard. Line thickness variations.
    *   *Solution:* Experiment with Potrace parameters. Implement vector post-processing: simplification, smoothing, filtering small contours. For the 6 drawing types, this will require specific strategies (e.g., for "Hull Shape Only", find the largest external contour).

*   **Generating Specific Intermediate Drawings:**
    *   *Challenge:* Programmatically separating "hull only" from "superstructure" or "turrets" from vector data is non-trivial. It requires semantic understanding of the shapes.
    *   *Solution:*
        *   **Hull Only:** After vectorization, identify the outermost continuous contour.
        *   **Structures Except Turrets:** This is the hardest. May require:
            *   Identifying the hull.
            *   Identifying turret-like circular/polygonal shapes (based on typical turret characteristics) and excluding them.
            *   Remaining elements are superstructure. This might need some machine learning or advanced shape analysis for high reliability, but simpler heuristics can be tried first (e.g., size, position relative to hull).
        *   **Best Detail:** Use the vectorization output with minimal filtering.

*   **3D Hull Generation - Inferring Complex Curvature:**
    *   *Challenge:* Accurately inferring 3D curvature from just two 2D orthogonal profiles is an ill-posed problem without assumptions or further data.
    *   *Solution:* This is where `calibration_data` is key. The system isn't inventing the curvature from scratch but rather *adapting* known, typical hull forms (from high-poly models) to fit the input 2D profiles. The quality depends heavily on the quality and relevance of the calibration data.

*   **Calibration Data - Robustness & Generalization:**
    *   *Challenge:* Ensuring the calibration data derived from existing high-poly models is general enough to apply to new, potentially different ship designs. Overfitting to the reference models.
    *   *Solution:* Use a diverse set of reference models for calibration. Parameterize the calibration data rather than using fixed shapes (e.g., use functions that describe curvature based on length/beam ratios). Start with simpler calibration (e.g., average cross-section shapes) and refine.

*   **Output File Management:**
    *   *Challenge:* Ensuring correct naming and directory creation, especially if Gemini API returns unexpected ship names.
    *   *Solution:* Sanitize ship names for use in file paths (remove special characters). Have a default "Unknown_Ship" folder if identification fails.

## 6. Phased Development Plan

1.  **Phase 1: Core Setup & Image Input:**
    *   Project structure, virtual environment, core packages.
    *   Basic image loading and the multi-view splitting heuristic.
    *   Logging setup.
    *   Jupyter Notebook for testing.

2.  **Phase 2: Gemini API Integration & Caching:**
    *   Implement `GeminiVisionClient` for view type, ship ID, text transcription.
    *   Implement API response caching.
    *   Basic output directory creation based on Gemini results.
    *   Save transcribed text.

3.  **Phase 3: Raster-to-Vector & Initial Intermediate Drawings:**
    *   Integrate Potrace (or alternative) for basic vectorization.
    *   Focus on generating the "Best Detail" drawing first for both top and side views.
    *   Implement logic for "Hull Shape Only" (e.g., largest contour extraction).

4.  **Phase 4: Advanced Intermediate Drawings & Refinement:**
    *   Develop strategies for "Structures Except Turrets" drawings. This will likely be iterative and may require significant image processing logic or simple heuristics.
    *   Refine all 6 intermediate drawing types.

5.  **Phase 5: Basic 3D Hull Generation (Extrusion & Placeholder Calibration):**
    *   Implement initial 3D hull generation, perhaps starting with a simpler extrusion of the top view hull shape, modified by the side view height profile.
    *   Set up the structure for `calibration_data` even if it's initially populated with placeholder/generic values.

6.  **Phase 6: Enhanced Hull Generation & Calibration Research:**
    *   Implement the pseudocode for `generate_enhanced_hull`.
    *   Develop scripts/tools to analyze existing high-poly 3D models to extract actual `calibration_data`. This is a significant sub-project.
    *   Iterate on hull quality based on calibration.

7.  **Phase 7: Superstructure Generation & Final Output:**
    *   Implement basic superstructure extrusion based on relevant intermediate drawings.
    *   Finalize output file structure and run log generation.
    *   Testing and bug fixing.

## 7. Research Findings

**a. Gemini API Token Cost (Approximate Estimate):**

*   **Disclaimer:** Token costs can vary based on the exact model version, input complexity (image size, detail), and the length/complexity of the text prompt and desired JSON schema. The following is a general estimation based on publicly available information about Gemini models.
*   Gemini models (including Flash versions) often have pricing based on input characters (for text) and per image. For `gemini-pro-vision` (a common reference, Flash models aim for lower cost/latency), image input might be charged per image, and text input/output per 1000 characters.
*   **Image Input:** A single image is typically a fixed cost component for analysis.
*   **Text Prompt & Schema:** The prompt "Analyze this battleship image. Determine view_type (top or side), ship_identification (class or name), and transcribe_text." plus the JSON schema definition is relatively small, likely a few hundred characters.
*   **Output Text:**
    *   `view_type`: Short string (e.g., "top_view").
    *   `ship_identification`: Variable, could be a few words (e.g., "Iowa Class Battleship").
    *   `transcribed_text`: Highly variable. Could be none, a few words, or several sentences if there's a lot of text on the image.
*   **Estimation:**
    *   For a typical image with moderate detail and a small amount of transcribed text, the cost per API call to `gemini-1.5-flash-latest` (which is optimized for speed and cost) would be on the lower end of Gemini vision model pricing.
    *   **It's crucial to consult the official Google Cloud AI Platform pricing page for `gemini-1.5-flash-latest` for the most current and precise figures.** As of my last training, specific token counts for image features are not always detailed in the same way as pure text models. However, expect a charge per image processed plus a smaller charge for the input/output text tokens.
    *   The `responseMimeType: "application/json"` and `responseSchema` will help control the output structure, potentially making it more concise and predictable in terms of token usage for the JSON structure itself, but the content within (like transcribed text) remains variable.
    *   **Caching is essential to manage these costs effectively.**

**b. Python Packages for Raster-to-Vector Conversion:**

*   **Potrace (via `pypotrace` or `subprocess`):**
    *   *Recommendation:* Strong contender.
    *   *Pros:* Excellent results for black & white images, designed for tracing bitmaps. Produces SVG, EPS, etc. Open source and well-established.
    *   *Cons:* Requires Potrace to be installed on the system. `pypotrace` is a wrapper; direct `subprocess` calls are also an option. Best for images that can be cleanly binarized.
*   **OpenCV (cv2):**
    *   *Recommendation:* Good for pre-processing and contour extraction, which can then be converted to vectors.
    *   *Pros:* `cv2.findContours` can find outlines of shapes. These contours are lists of points.
    *   *Cons:* Not a direct "raster-to-SVG" tool. You'd get points that you then need to format into SVG paths (e.g., using `svgwrite`). More manual control, but also more flexibility.
*   **scikit-image:**
    *   *Recommendation:* Similar to OpenCV for providing tools that are steps towards vectorization.
    *   *Pros:* Algorithms like Canny edge detection, contour finding.
    *   *Cons:* Like OpenCV, gives you intermediate data (pixel coordinates of edges/contours) that needs further processing into a vector format.
*   **`svg.path` and `svgwrite`:**
    *   *Recommendation:* For constructing SVG data if you have line/curve information from other sources (like OpenCV contours).
    *   *Pros:* `svgwrite` makes it easy to generate SVG files. `svg.path` can help parse and manipulate SVG path data if you need to modify existing SVGs or work with path definitions.
    *   *Cons:* Not a tracer itself.
*   **Commercial Libraries/APIs:** Some exist but are likely outside the scope unless high-fidelity, complex conversion is a primary, unsolved need.

**Initial Choice:** Start with **Potrace (via `pypotrace`)** for its direct tracing capabilities, especially for the "Hull Shape Only" and "Best Detail" if images can be binarized effectively. Use **OpenCV** for pre-processing (thresholding, noise reduction) and potentially for more complex feature identification that Potrace might miss or that needs to be isolated before tracing.

**c. Initial Research on Methods for Analyzing High-Poly 3D Models (Calibration Data):**

*   **Goal:** Extract parameters or reference shapes from existing high-poly battleship models to inform the generation of new, basic 3D hulls from 2D outlines.
*   **Methods:**
    1.  **Cross-Sectional Analysis:**
        *   Load a high-poly model (e.g., OBJ, STL) using `Trimesh` or `PyVista`.
        *   Define a series of planes along the model's main axis (e.g., longitudinal).
        *   Compute the intersection of the mesh with each plane. This yields 2D cross-sectional polylines/contours.
        *   Analyze these cross-sections:
            *   Normalize them (e.g., by local beam/width and draft/height).
            *   Parameterize their shape (e.g., using Fourier descriptors, or by fitting splines and storing control points).
            *   Cluster similar cross-section shapes to find "typical" forms (e.g., U-shape near midship, V-shape near bow/stern).
            *   Store these parameterized shapes or representative samples as `typical_cross_section_shapes`.
    2.  **Profile Curve Extraction:**
        *   Extract key profile curves:
            *   **Deck Outline (Top View):** Project all vertices onto the XY plane and find the convex hull or alpha shape.
            *   **Sheer Line (Side View - Top Edge):** Identify vertices along the top edge of the hull in side view.
            *   **Keel Line (Side View - Bottom Edge):** Identify vertices along the bottom center profile.
        *   Parameterize these curves (e.g., fit splines, store control points). These can serve as references for `bow_stern_tapering_curves` or overall hull form.
    3.  **Geometric Feature Ratios:**
        *   Calculate key ratios from the high-poly models: length-to-beam, beam-to-draft, block coefficient, prismatic coefficient (if calculable).
        *   These ratios can help select or weight appropriate calibration data when generating a new model based on its 2D input's derived ratios.
    4.  **Principal Component Analysis (PCA) on Vertex Data:**
        *   For a collection of aligned reference models, PCA could identify principal modes of variation in hull shape. These modes could potentially be used to deform a template mesh. More complex.
    5.  **Statistical Shape Modeling:**
        *   If many reference models are available, build a statistical model of hull shapes. This is advanced but powerful for generating plausible new shapes that fit within the learned distribution.

*   **Tools for Implementation:**
    *   `Trimesh`: Mesh loading, slicing (intersection with planes), vertex/face manipulation, geometric queries.
    *   `NumPy`/`SciPy`: Numerical calculations, spline fitting, PCA, clustering.
    *   `Matplotlib`/`PyVista`: For visualizing meshes, cross-sections, and curves during the analysis phase.

*   **Initial Approach for this Project:**
    *   Focus on **Cross-Sectional Analysis** and **Profile Curve Extraction**.
    *   Manually (or semi-automatically) select a few representative high-poly models.
    *   Write scripts using `Trimesh` to slice them at regular intervals (e.g., every 5% of the length).
    *   Store the resulting 2D cross-section polylines.
    *   Develop a way to normalize and parameterize these (e.g., scale to unit width/height, then sample points or fit a simple curve).
    *   This `calibration_data` would then be used by `generate_enhanced_hull` to select and deform appropriate cross-sections based on the input 2D views.
