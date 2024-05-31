import os
from flask import Blueprint, render_template, request, url_for, redirect
from datetime import datetime
from .braille import *

bp = Blueprint("main", __name__, url_prefix="/")


@bp.route("/")
def home():
    return render_template("homepage.html")


@bp.route("upload", methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files["img_file"]
        repeat = int(request.form["repeat"])
        extension = "." + file.filename.rsplit(".", 1)[1]
        img_file = datetime.now().strftime("%y%m%d%H%M%S") + extension
        curr_dir = os.getcwd()
        image_path = curr_dir + "/Braille/static/temp/" + img_file
        deleteAllFiles(curr_dir + "/Braille/static/temp/")
        file.save(image_path)
        upload_file = url_for("static", filename=f"temp/{img_file}")
        sentence = braille_to_text(image_path, repeat)
    return render_template(
        "resultpage.html", upload_file=upload_file, sentence=sentence
    )


def braille_to_text(image_path, repeat=2):
    while True:
        try:
            image, ctrs, paper, gray, canny, thresh = get_image(
                image_path, iter=repeat, width=1500
            )

            diam = get_diameter(ctrs)
            dotCtrs = get_circles(ctrs, diam)

            questionCtrs, boundingBoxes, xs, ys = sort_contours(dotCtrs, diam)
            paper = draw_contours(paper, boundingBoxes, questionCtrs)

            linesV, d1, d2, d3, spacingX, spacingY = get_spacing(
                boundingBoxes, diam, xs
            )

            letters = get_letters(boundingBoxes, diam, spacingY, linesV)
            sentence = translate(letters)
            if "■" in sentence and repeat != 0:
                repeat -= 1
                continue
            break
        except:
            if repeat == 0:
                sentence = (
                    "점자를 식별할 수 없습니다. 좀 더 선명한 이미지로 다시 시도하세요."
                )
                break
            repeat -= 1
    print(repeat)
    return sentence


def deleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"
