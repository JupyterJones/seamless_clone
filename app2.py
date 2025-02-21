from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Upload Route --------------------
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "background" not in request.files or "object" not in request.files:
            return "Missing files", 400

        background = request.files["background"]
        object_img = request.files["object"]

        bg_path = os.path.join(UPLOAD_FOLDER, "background.jpg")
        obj_path = os.path.join(UPLOAD_FOLDER, "object.jpg")

        background.save(bg_path)
        object_img.save(obj_path)

        return render_template("index2.html", bg_path=bg_path, obj_path=obj_path)

    return render_template("index2.html", bg_path=None, obj_path=None)

# -------------------- Cloning Route --------------------
@app.route("/clone", methods=["POST"])
def clone():
    data = request.json
    poly_points = np.array(data["points"], np.int32)
    center = tuple(data["center"])

    bg_path = os.path.join(UPLOAD_FOLDER, "background.jpg")
    obj_path = os.path.join(UPLOAD_FOLDER, "object.jpg")

    background = cv2.imread(bg_path)
    object_img = cv2.imread(obj_path)

    if background is None or object_img is None:
        return jsonify({"error": "Error loading images"}), 500

    # Ensure the object and background are in the same size range
    if object_img.shape[:2] != background.shape[:2]:
        object_img = cv2.resize(object_img, (background.shape[1], background.shape[0]))

    # Create mask
    mask = np.zeros(object_img.shape, object_img.dtype)
    cv2.fillPoly(mask, [poly_points], (255, 255, 255))

    # Perform seamless cloning (Object pasted onto Background)
    result = cv2.seamlessClone(object_img, background, mask, center, cv2.NORMAL_CLONE)

    # Save result
    result_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(result_path, result)

    return jsonify({"result_url": f"/{result_path}"})

# -------------------- Serve Uploaded Files --------------------
@app.route("/uploads/<filename>")
def serve_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
