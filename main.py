import cv2
import numpy as np
import pyautogui
import math
from copy import deepcopy

GRID_X1, GRID_Y1 = 747, 276
GRID_X2, GRID_Y2 = 1301, 832

BLOCK_SIZE = 90
SPACING = 3
GRID_SIZE = 6

def identify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    vertices = len(approx)

    area = cv2.contourArea(contour)
    if peri == 0:
        return "unknown", {"vertices": 0, "perimeter": 0.0, "area": 0.0, "circularity": 0.0}

    circularity = 4 * math.pi * area / (peri * peri)

    #print(f"Vertices: {vertices}, Perimeter: {peri:.2f}, Area: {area:.2f}, Circularity: {circularity:.2f}")

    shape = "unknown"

    if circularity > 0.9:
        shape = "circle"
    elif vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        pts = approx.reshape(4, 2)
        edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        ratio = max(edges) / min(edges)

        if ratio < 1.2:
            shape = "rhombus"
        else:
            shape = "nonstnd"
    elif vertices >= 5:
        shape = "nonstnd"

    return shape, {
        "vertices": vertices,
        "perimeter": round(peri, 2),
        "area": round(area, 2),
        "circularity": round(circularity, 2)
    }

def detect_block(block_img):
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    outlined = block_img.copy()
    shape_name = "unknown"

    if contours:
        merged = np.vstack(contours)
        hull = cv2.convexHull(merged)

        cv2.drawContours(outlined, [hull], -1, (0, 0, 0), 2)
        shape_name, data = identify_shape(hull)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness_outline = 2
        thickness_text = 1
        line_height = 12

        cv2.putText(outlined, shape_name, (5, 15), font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
        cv2.putText(outlined, shape_name, (5, 15), font, scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

        y0 = 30
        for i, (k, v) in enumerate(data.items()):
            text = f"{k[0]}: {v}"
            y = y0 + i * line_height
            cv2.putText(outlined, text, (5, y), font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
            cv2.putText(outlined, text, (5, y), font, scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

    return outlined, shape_name

def find_matches(grid):
    matched = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]

    for r in range(GRID_SIZE):
        c = 0
        while c < GRID_SIZE - 2:
            val = grid[r][c]
            if val != -1 and val == grid[r][c + 1] == grid[r][c + 2]:
                k = c
                while k < GRID_SIZE and grid[r][k] == val:
                    matched[r][k] = True
                    k += 1
                c = k
            else:
                c += 1

    for c in range(GRID_SIZE):
        r = 0
        while r < GRID_SIZE - 2:
            val = grid[r][c]
            if val != -1 and val == grid[r + 1][c] == grid[r + 2][c]:
                k = r
                while k < GRID_SIZE and grid[k][c] == val:
                    matched[k][c] = True
                    k += 1
                r = k
            else:
                r += 1

    total_unique = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if matched[r][c])
    return matched, total_unique

def apply_gravity_no_refill(grid):
    for c in range(GRID_SIZE):
        write_row = GRID_SIZE - 1
        for r in range(GRID_SIZE - 1, -1, -1):
            if grid[r][c] != -1:
                grid[write_row][c] = grid[r][c]
                write_row -= 1
        for r in range(write_row, -1, -1):
            grid[r][c] = -1

def simulate_once_no_random(grid):
    matched, gained = find_matches(grid)
    if gained == 0:
        return 0

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if matched[r][c]:
                grid[r][c] = -1

    apply_gravity_no_refill(grid)
    return gained

def best_move(board):
    best_score = -1
    best_swap = None

    directions = [(0, 1), (1, 0)]

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    grid_copy = deepcopy(board)
                    grid_copy[r][c], grid_copy[nr][nc] = grid_copy[nr][nc], grid_copy[r][c]
                    score = simulate_once_no_random(grid_copy)

                    if score > best_score:
                        best_score = score
                        best_swap = ((r, c), (nr, nc))
                    elif score == best_score and best_swap is not None:
                        if ((r, c), (nr, nc)) < best_swap:
                            best_swap = ((r, c), (nr, nc))

    if best_score < 0:
        best_score = 0

    return best_swap, best_score

SHAPE_TO_COLOR = {
    "circle": 0,
    "triangle": 1,
    "rhombus": 2,
    "nonstnd": 3,
    "unknown": -1
}

def main():
    screenshot = pyautogui.screenshot(region=(
        GRID_X1, GRID_Y1, GRID_X2 - GRID_X1, GRID_Y2 - GRID_Y1
    ))
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    board = [[-1] * GRID_SIZE for _ in range(GRID_SIZE)]

    canvas = np.zeros_like(img)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * (BLOCK_SIZE + SPACING)
            y1 = row * (BLOCK_SIZE + SPACING)
            x2 = x1 + BLOCK_SIZE
            y2 = y1 + BLOCK_SIZE

            debug_block, shape = detect_block(img[y1:y2, x1:x2])
            board[row][col] = SHAPE_TO_COLOR.get(shape, -1)
            canvas[y1:y2, x1:x2] = debug_block

    for row in board: print(row)

    move, score = best_move(board)
    if move:
        (r1, c1), (r2, c2) = move
        print("Best move: (Row {}, Col {}) <-> (Row {}, Col {}) | Score: {}".format(
                r1 + 1, c1 + 1, r2 + 1, c2 + 1, score))

    #cv2.imshow("Shape Debugging", canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()