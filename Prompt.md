---

### **AI Coding Agent Prompt: Battleship 2D to 3D Conversion Project Framework**

**Role:** You are an expert Python software architect and developer. Your task is to design the initial framework and outline the foundational steps for a potentially large Python project.

**Project Goal:** The primary objective is to develop a system that takes simple 2D images or vector graphics of battleships, processes them into specific line drawings/vector representations, and ultimately generates **basic, but geometrically informed, 3D meshes** suitable for further refinement by another AI. The meshes should serve as a fundamental structural base for a game environment.

**Initial Setup & Environment First Steps for the AI Agent:**

1.  **Required Package Research & Download Guidance:** Before beginning any coding, the AI should research and identify all necessary Python packages for this project's requirements (e.g., for image splitting, Gemini API interaction, raster-to-vector conversion, 3D mesh generation basics). The AI should then provide instructions on how to install these packages in a virtual environment.
2.  **Jupyter Notebook Setup:** The AI should guide me through the setup of a Jupyter Notebook environment specifically for this project. This notebook will serve as the primary workspace for initial experimentation, development, and visualization of intermediate results.

**Input Handling & Pre-processing:**

1.  **Input Source:** The system will accept simple images (e.g., PNG, JPG) or vector graphics (format unknown at this stage, but common ones like SVG are likely). Input images are expected to predominantly feature battleships.
2.  **Multi-View Splitting:**
    * The system must identify and separate multiple ship views within a single input image (e.g., side view and top view present in one picture).
    * **Proposed Heuristic:** A simple test should be performed: try a horizontal line across the middle of the image. If this line contains only background pixels (or pixels falling within a determined background color range), the image should be split horizontally into two separate images for further processing. This heuristic should be implemented as a primary pre-processing step.
3.  **View Type Determination & Metadata Extraction (Gemini API Integration):**
    * For each split image (or original image if not split), the system must determine if it represents a **"side view"** or a **"top view"** of the battleship. This determination *must* be made using a call to the **Gemini API's vision capabilities**.
    * Concurrently, the Gemini API should also be used to:
        * **Classify the ship's class or name** to the best of its ability.
        * **Transcribe any discernible text** within the image to a separate text file.
    * **Gemini API Call Details:**
        * **API Endpoint:** Use `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`.
        * **Model:** Target `gemini-2.0-flash`.
        * **API Key Handling:** The API key should be left as an empty string (`""`) in the generated code, assuming it will be handled externally by the execution environment (e.g., set as an environment variable or passed securely outside the code). No API key validation logic is needed in the generated code.
        * **Structured Response:** The API call's `generationConfig` must include `responseMimeType: "application/json"`. The prompt should also include a `responseSchema` that clearly defines the expected JSON structure for `view_type`, `ship_identification`, and `transcribed_text`, similar to the example provided in the AI Studio prompt.
    * **Caching Requirement:** To minimize API calls and cost, the system should implement a caching mechanism. If an image has already been processed by the Gemini API (i.e., its view type, ship class, and transcribed text are already stored), the system should **not** resubmit it, unless a specific `debug_flag` or similar indicator is set when running the script.
    * **Research Task:** Research and provide an *approximate estimate of the token cost* for such a Gemini API call, considering image analysis for view type, classification, and text transcription.

**Image/Vector Processing & Intermediate Output:**

Before any 3D component generation, the project must simplify and ingest the processed 2D views. The system should produce *six specific types of intermediate image/vector drawings* based on the determined view type (side or top). The goal is to obtain precise line data, ideally in a vector format, for subsequent 3D conversion.

* **Research Task (Raster-to-Vector):** Research and recommend Python package options suitable for converting raster images into precise vector graphics or other formats where the precise length and coordinates of all lines is known. Consider options that can handle complex line extraction from both photographic-like images and potentially cleaner line drawings.

* **Required Intermediate Outputs (for each identified ship view):**

    * **For Top Views:**
        1.  **Hull Shape Only:** A drawing showing only the outer silhouette of the ship's hull from the top, excluding all superstructure, turrets, and minor details.
        2.  **Structures Except Turrets:** A drawing showing the hull outline plus all superstructure elements, but explicitly *excluding* any turrets.
        3.  **Best Detail:** A drawing capturing the maximum detail the program can reliably extract from the top view image.

    * **For Side Views:**
        1.  **Hull Shape Only:** A drawing showing only the outer silhouette of the ship's hull from the side, without any superstructure.
        2.  **Structures Except Turrets:** A drawing showing the hull outline plus all superstructure elements, but explicitly *excluding* any turrets.
        3.  **Best Detail:** A drawing capturing the maximum detail the program can reliably extract from the side view image.

**3D Mesh Generation (Basic Hull & Superstructure):**

The core 3D generation will proceed from the processed 2D vector drawings.

* **Hull Generation (Enhanced):**
    * The system should generate a 3D mesh for the ship's hull that goes beyond a simple, flat extrusion of the top view.
    * It should leverage the information from both the **top-view hull outline** and the **side-view hull outline** to create a more accurate, basic 3D representation that accounts for the varying depth, curvature, and tapering of the hull from bow to stern and along the sides.
    * **Calibration Data:** The project will have access to many existing high-poly 3D models of battleships. The system should propose methods to **analyze these high-poly models to calibrate parameters or derive reference shapes** that can inform the extrapolation of the new ship's hull form from its 2D outlines. This calibration step is crucial for achieving a more realistic basic hull.
* **Superstructure & Turret Generation:** For superstructures and turrets, the system can generally rely on simpler methods, such as extruding their vectorized top-view outlines upwards to estimated heights.

**Output File Management & Logging:**

1.  **Output Directory Structure:** All generated output files (intermediate drawings, transcribed text, 3D meshes - even if basic) must be saved into a structured directory under a main `output/` folder. The structure should be:
    * `output/`
        * `[Identified_Ship_Class_or_Name]/` (e.g., `output/Iowa_Class/`)
            * `[Source_Picture_Filename_without_extension]/` (e.g., `output/USS_Iowa_1945_Photo/`)
                * `[generated_files_here.png/.svg/.obj]`
    * The `[Identified_Ship_Class_or_Name]` and `[Source_Picture_Filename_without_extension]` folders should be created by the system, preferably during the Gemini API processing step, to ensure early organization.
2.  **Run Log File:** For each complete run of the script, a log file (e.g., `run_log_[timestamp].txt`) must be generated. This log file should include:
    * A list of all intermediate drawings produced, along with their exact file paths.
    * The predicted ship class/name by Gemini for each processed image.
    * Notes for any images where conversion or processing steps failed.

**Framework Proposal Requirements for the AI's Response:**

Based on the above, propose a detailed initial framework for this project. Your proposal should include:

1.  **High-Level Architecture:** A clear description (or pseudocode for a diagram) of the main modules/stages of the pipeline, from initial setup to intermediate output and basic 3D mesh generation.
2.  **Key Python Library Recommendations:** For each major stage (image splitting, Gemini interaction, raster-to-vector conversion, 3D model analysis for calibration, 3D mesh generation), recommend specific Python packages and briefly justify their suitability.
3.  **Core Class/Module Structure:** Outline the main classes or modules the project would require, with their primary responsibilities.
4.  **Pseudocode for Critical Algorithms:** Provide pseudocode for:
    * The multi-view image splitting heuristic.
    * The core logic for interacting with the Gemini API (including caching).
    * The high-level process for generating one of the six specified intermediate drawings (e.g., "Top view with hull shape only").
    * **Pseudocode for the enhanced hull generation process**, detailing how the 2D side and top outlines will be used with calibration data from high-poly models to extrapolate the 3D hull.
5.  **Anticipated Challenges & Initial Solutions:** Identify potential technical challenges at each stage (e.g., background noise, non-uniform line thickness, ambiguous views, Gemini accuracy, difficulty in inferring complex hull curvature from limited 2D views, robustness of calibration from varied high-poly models) and propose initial strategies to address them.
6.  **Phased Development Plan:** Suggest a logical sequence of development phases for building out this project.
7.  **Research Findings:** Present your research on Gemini API token costs for the specified tasks, the recommended Python packages for raster-to-vector conversion, and **initial research on methods for analyzing high-poly 3D models to extract useful calibration parameters for 2D-to-3D hull inference.**

**Constraints & Preferences:**

* **Language:** Python.
* **Mesh Detail:** Remember the meshes are "very basic" as they are destined for further AI refinement. Focus on generating a solid foundational shape, not a highly detailed or perfectly smooth one.
* **Error Handling:** Note failed conversions in the log, no complex recovery needed for initial framework.

---




Interaction Guidelines for the AI Agent:

Confirm Understanding: Always confirm your understanding of a task or section of the brief before proceeding to implementation or detailed design.
Ask Questions When Needed: If you encounter any ambiguity, missing information, or foresee a significant challenge that requires input or a decision from me, please stop and ask clarifying questions. Do not make assumptions on critical architectural or functional decisions.
Generate Commit Commands on Pause: When you stop to ask a clarifying question or request a decision from me, please also generate the necessary git add commands (e.g., git add . or git add path/to/file.py) and a concise, descriptive git commit -m "Your suggested message" command. The commit message should summarize the work completed up to that point before you pose your question. This will help checkpoint our progress.
Phased Delivery: Present your work in logical, manageable phases as outlined in the "Phased Development Plan" you will propose.

---

## Project Status Log

**Last Major Step Completed (as of 2025-05-23):**
*   **Phase 2: Gemini API Integration & Caching**
    *   Successfully integrated the Gemini API for image analysis (ship identification, view type, text transcription).
    *   Implemented API response caching using `diskcache`.
    *   Updated `main.py` to use Gemini results for output directory naming and to save transcribed text.
    *   Troubleshooting included resolving `ModuleNotFoundError` for `diskcache` (ensuring `pip install -r requirements.txt` was run in the active venv) and API key authentication issues (ensuring `GOOGLE_API_KEY` environment variable was correctly set).
*   **Current Focus:** Beginning **Phase 3: Raster-to-Vector & Initial Intermediate Drawings**. Currently troubleshooting Potrace installation and PATH configuration to ensure `pypotrace` can find the `potrace.exe` command-line tool.