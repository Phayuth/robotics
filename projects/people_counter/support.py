import cv2


# https://www.jeffreythompson.org/collision-detection/line-rect.php
# Check Collision
def lineline_collision(l1x1, l1y1, l1x2, l1y2, l2x1, l2y1, l2x2, l2y2):

    uA = ((l2x2 - l2x1) * (l1y1 - l2y1) - (l2y2 - l2y1) * (l1x1 - l2x1)) / ((l2y2 - l2y1) * (l1x2 - l1x1) - (l2x2 - l2x1) * (l1y2 - l1y1))
    uB = ((l1x2 - l1x1) * (l1y1 - l2y1) - (l1y2 - l1y1) * (l1x1 - l2x1)) / ((l2y2 - l2y1) * (l1x2 - l1x1) - (l2x2 - l2x1) * (l1y2 - l1y1))

    if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
        return True
    else:
        return False


def linerectangle_collsion(l1x1, l1y1, l1x2, l1y2, bx1, by1, box_height, box_width):
    left = lineline_collision(l1x1, l1y1, l1x2, l1y2, bx1, by1, bx1, by1 + box_height)
    right = lineline_collision(l1x1, l1y1, l1x2, l1y2, bx1 + box_width, by1, bx1 + box_width, by1 + box_height)
    top = lineline_collision(l1x1, l1y1, l1x2, l1y2, bx1, by1, bx1 + box_width, by1)
    bottom = lineline_collision(l1x1, l1y1, l1x2, l1y2, bx1, by1 + box_height, bx1 + box_width, by1 + box_height)

    # if ANY of the above are true, the line has hit the rectangle
    if left or right or top or bottom:
        return True
    else:
        return False


def draw_rectangle(img, x1, y1, x2, y2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), thickness=1)


def draw_text(img, content, x1, y1):
    cv2.putText(img, text=content, org=(int(x1), int(y1)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=1, color=(255, 0, 255))


def draw_line(img, line):
    cv2.line(img, pt1=(line[0], line[1]), pt2=(line[2], line[3]), color=(0, 255, 0), thickness=4)
