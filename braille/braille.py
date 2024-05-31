import numpy as np
import cv2, imutils, re
import matplotlib.pyplot as plt
from collections import Counter


def get_image(image_path, iter=2, width=None):
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(
        image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    if width:
        image = imutils.resize(image, width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 200)

    paper = image.copy()

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=iter)
    thresh = cv2.dilate(thresh, kernel, iterations=iter)

    ctrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    return image, ctrs, paper, gray, canny, thresh


def get_diameter(ctrs):
    boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
    c = Counter([i[2] for i in boundingBoxes])
    mode = c.most_common(1)[0][0]
    if mode > 1:
        diam = mode
    else:
        diam = c.most_common(2)[1][0]
    return diam


def sort_contours(ctrs, diam):
    boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
    # tol = 0.7 * diam
    tol = diam * 1.2

    def sort(i):
        # S = sorted(BB, key=lambda x: x[i])
        # s = [b[i] for b in S]
        # m = s[0]

        # for b in S:
        #     if m - tol < b[i] < m or m < b[i] < m + tol:
        #         b[i] = m
        #     elif b[i] > m + diam:
        #         for e in s[s.index(m) :]:
        #             if e > m + diam:
        #                 m = e
        #                 break
        # return sorted(set(s))

        S = sorted(boundingBoxes, key=lambda x: x[i])
        s = [b[i] for b in S]
        m = s[0]

        for idx, b in enumerate(S):
            if m - tol < b[i] < m or m < b[i] < m + tol:
                if i == 0:
                    gap = b[i] - m
                    y_line = b[1]
                    for index in range(S.index(b), len(S)):
                        if S[index][1] == y_line:
                            S[index][i] -= gap
                            s[index] -= gap
                else:
                    b[1] = m
                    s[idx] = m
            elif b[i] > m + diam:
                for e in s[s.index(m) :]:
                    if e > m + diam:
                        m = e
                        break
        return sorted(set(s))

    ys = sort(1)
    xs = sort(0)

    (ctrs, boundingBoxes) = zip(
        *sorted(zip(ctrs, boundingBoxes), key=lambda b: (b[1][1], b[1][0]))
    )
    return ctrs, boundingBoxes, xs, ys


def get_circles(ctrs, diam):
    questionCtrs = []
    for c in ctrs:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if diam * 0.8 <= w <= diam * 1.2 and 0.8 <= ar <= 1.2:
            questionCtrs.append(c)
    return questionCtrs


def draw_contours(paper, boundingBoxes, questionCtrs):
    color = (0, 255, 0)
    i = 0
    for q in range(len(questionCtrs)):
        cv2.drawContours(paper, questionCtrs[q], -1, color, 3)
        cv2.putText(
            paper,
            str(i),
            (
                boundingBoxes[q][0] + boundingBoxes[q][2] // 2,
                boundingBoxes[q][1] + boundingBoxes[q][3] // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        i += 1
    return paper


def get_spacing(boundingBoxes, diam, xs):

    def spacing(x):
        space = []
        coor = [b[x] for b in boundingBoxes]
        for i in range(len(coor) - 1):
            c = coor[i + 1] - coor[i]
            if c > diam // 2:
                space.append(c)
        return sorted(list(set(space)))

    spacingX = spacing(0)
    spacingY = spacing(1)

    d1 = spacingX[0]
    d2 = 0
    d3 = 0

    for x in spacingX:
        if d2 == 0 and x > d1 * 1.3:
            d2 = x
        if d2 > 0 and x > d2 * 1.3:
            d3 = x
            break

    linesV = []
    prev = 0  # outside

    linesV.append(min(xs) - (d2 - diam) / 2)

    for i in range(1, len(xs)):
        diff = xs[i] - xs[i - 1]
        if i == 1 and d2 * 0.9 < diff:
            linesV.append(min(xs) - d2 - diam / 2)
            prev = 1
        if d1 * 0.8 < diff < d1 * 1.2:
            linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
            prev = 1
        elif d2 * 0.8 < diff < d2 * 1.1:
            linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
            prev = 0
        elif d3 * 0.9 < diff < d3 * 1.1:
            if prev == 1:
                linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d2 + diam + (d1 - diam) / 2)
            else:
                linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + diam + (d2 - diam) / 2)
        elif d3 * 1.1 < diff:
            if prev == 1:
                linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d2 + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d3 + diam + (d2 - diam) / 2)
                prev = 0
            else:
                linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + d2 + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + d3 + diam + (d2 - diam) / 2)
                prev = 1

    linesV.append(max(xs) + diam * 1.5)
    if len(linesV) % 2 == 0:
        linesV.append(max(xs) + d2 + diam)

    return linesV, d1, d2, d3, spacingX, spacingY


def display_contours(paper, linesV, lines=False):

    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.axis("off")
    plt.imshow(paper)
    if lines:
        for x in linesV:
            plt.axvline(x)

    plt.show()


def get_letters(boundingBoxes, diam, spacingY, linesV):

    boxes = list(boundingBoxes)
    boxes.append((100000, 0))

    dots = [[]]
    for y in sorted(list(set(spacingY))):
        if y > 1.3 * diam:
            minYD = y * 1.5
            break

    for b in range(len(boxes) - 1):
        if boxes[b][0] < boxes[b + 1][0]:
            dots[-1].append(boxes[b][0])
        else:
            if abs(boxes[b + 1][1] - boxes[b][1]) < minYD:
                dots[-1].append(boxes[b][0])
                dots.append([])
            else:
                dots[-1].append(boxes[b][0])
                dots.append([])
                if len(dots) % 3 == 0 and not dots[-1]:
                    dots.append([])

    letters = []

    for r in range(len(dots)):
        if not dots[r]:
            letters.append([0 for _ in range(len(linesV) - 1)])
            continue
        else:
            letters.append([])
            c = 0
            i = 0
            while i < len(linesV) - 1:
                if c < len(dots[r]):
                    if linesV[i] < dots[r][c] < linesV[i + 1]:
                        letters[-1].append(1)
                        c += 1
                    else:
                        letters[-1].append(0)
                else:
                    letters[-1].append(0)
                i += 1

    return letters


def translate(letters):

    alpha = {
        "a": "1",
        "b": "13",
        "c": "12",
        "d": "124",
        "e": "14",
        "f": "123",
        "g": "1234",
        "h": "134",
        "i": "23",
        "j": "234",
        "k": "15",
        "l": "135",
        "m": "125",
        "n": "1245",
        "o": "145",
        "p": "1235",
        "q": "12345",
        "r": "1345",
        "s": "235",
        "t": "2345",
        "u": "156",
        "v": "1356",
        "w": "2346",
        "x": "1256",
        "y": "12456",
        "z": "1456",
        "#": "2456",
        "^": "6",
        ",": "3",
        ".": "346",
        '"': "356",
        "^": "26",
        ":": "34",
        "'": "5",
    }

    nums = {
        "a": "1",
        "b": "2",
        "c": "3",
        "d": "4",
        "e": "5",
        "f": "6",
        "g": "7",
        "h": "8",
        "i": "9",
        "j": "0",
    }

    braille = {v: k for k, v in alpha.items()}

    letters = np.array([np.array(l) for l in letters])

    sentence = ""

    for r in range(0, len(letters), 3):
        for c in range(0, len(letters[0]), 2):
            f = letters[r : r + 3, c : c + 2].flatten()
            f = "".join([str(i + 1) for i, d in enumerate(f) if d == 1])
            if f == "6":
                f = "26"
            if not f:
                if sentence:
                    if sentence[-1] != " ":
                        sentence += " "
            elif f in braille.keys():
                sentence += braille[f]
            else:
                sentence += "■"
        if sentence[-1] != " ":
            sentence += " "

    # replace numbers
    def replace_nums(m):
        char = m.group(0)[1:].strip()
        return "".join(nums.get(c, c) for c in char)

    sentence = re.sub("#(?P<key>[a-zA-Z]+)", replace_nums, sentence)

    # capitalize
    def capitalize(m):
        return m.group(0).upper()[1]

    sentence = re.sub("\^(?P<key>[a-zA-Z])", capitalize, sentence)

    return sentence


def main():
    # image_path = "./data/4ggIni9.jpeg"  # iter=0 not work
    # image_path = "./data/4nC067a.jpeg"  # iter=2
    # image_path = "./data/EjBz4nI.jpeg"  # iter=0
    # image_path = "./data/ihU7tFt.jpeg"  # iter=2
    # image_path = "./data/maU4r0t.jpeg"  # iter=2
    # image_path = "./data/nFT74Mv.jpeg"  # iter=2
    # image_path = "./data/NwLqmz2.jpeg"  # iter=3
    # image_path = "./data/osNCAx3.jpeg"  # iter=2
    # image_path = "./data/ttq5PzE.jpeg"  # iter=1
    # image_path = "./data/UBqs60s.jpeg"  # iter=2
    image_path = "./data/result.brf.png"  # iter=2
    repeat = 3
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
                # print(repeat)
                # if repeat == 0:
                #     raise ValueError(
                #         "점자가 균일하지 않습니다. 규격에 맞는 이미지로 다시 시도하세요."
                #     )
                repeat -= 1
                continue
            break
        except:
            if repeat == 0:
                raise ValueError(
                    "점자를 식별할 수 없습니다. 선명한 이미지로 다시 시도하세요."
                )
            repeat -= 1

    for img in [image, gray, canny, thresh]:
        plt.axis("off")
        plt.imshow(img, cmap="binary")
        plt.show()

    print(sentence)

    display_contours(paper, linesV, True)


if __name__ == "__main__":
    main()
