import cv2
import numpy as np
import pyautogui
import hashlib
import time
import pydirectinput, mouse
import keyboard
pydirectinput.PAUSE = 0

GRID_X1, GRID_Y1 = 725, 272
GRID_X2, GRID_Y2 = 1300, 847
GRID_SIZE = 6
BLOCK_SIZE = 90
SPACING = 6

def classify_color(hue_values):
    if len(hue_values) == 0:
        return "unknown"
    hist, bins = np.histogram(hue_values, bins=36, range=(0,180))
    top_bin_idx = np.argmax(hist)
    hue_center = (bins[top_bin_idx] + bins[top_bin_idx + 1]) / 2
    if hue_center >= 170 or hue_center < 15:
        return "red"
    elif 15 <= hue_center < 35:
        return "yellow"
    elif 85 <= hue_center < 115:
        return "blue"
    elif 120 <= hue_center < 150:
        return "purple"
    else:
        return "unknown"

color_board = {
    "red": 1,
    "yellow": 2,
    "blue": 3,
    "purple": 4,
    "unknown": -1,
}

def find_matches(grid):
    to_remove = set()
    n = len(grid)
    for r in range(n):
        c = 0
        while c < n:
            val = grid[r][c]
            if val <= 0:
                c += 1
                continue
            start = c
            while c + 1 < n and grid[r][c + 1] == val:
                c += 1
            if c - start + 1 >= 3:
                for cc in range(start, c + 1):
                    to_remove.add((r, cc))
            c += 1
    for c in range(n):
        r = 0
        while r < n:
            val = grid[r][c]
            if val <= 0:
                r += 1
                continue
            start = r
            while r + 1 < n and grid[r + 1][c] == val:
                r += 1
            if r - start + 1 >= 3:
                for rr in range(start, r + 1):
                    to_remove.add((rr, c))
            r += 1
    return to_remove

def drop_blocks(grid):
    n = len(grid)
    for c in range(n):
        write_row = n - 1
        for r in range(n - 1, -1, -1):
            if grid[r][c] > 0:
                if write_row != r:
                    grid[write_row][c] = grid[r][c]
                write_row -= 1
        for r in range(write_row, -1, -1):
            grid[r][c] = 0

def resolve_cascade(grid):
    total_points = 0
    while True:
        matches = find_matches(grid)
        if not matches:
            break
        total_points += len(matches)
        for (r, c) in matches:
            grid[r][c] = 0
        drop_blocks(grid)
    return total_points

def simulate_swap_and_score(orig_board, r1, c1, r2, c2):
    board = [row[:] for row in orig_board]
    board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
    if not find_matches(board):
        return 0
    grid = [row[:] for row in board]
    return resolve_cascade(grid)

def all_valid_swaps(board):
    swaps = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if c + 1 < GRID_SIZE:
                swaps.append((r, c, r, c + 1))
            if r + 1 < GRID_SIZE:
                swaps.append((r, c, r + 1, c))
    return swaps

def block_to_screen(col, row):
    x = GRID_X1 + col * (BLOCK_SIZE + SPACING) + BLOCK_SIZE // 2
    y = GRID_Y1 + row * (BLOCK_SIZE + SPACING) + BLOCK_SIZE // 2
    return x, y

def wait_for_single_click():
    mouse.wait(button='left', target_types=('down',))
    mouse.wait(button='left', target_types=('up',))

def main():
    screenshot = pyautogui.screenshot(region=(GRID_X1, GRID_Y1, GRID_X2 - GRID_X1, GRID_Y2 - GRID_Y1))
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    board = [[-1] * GRID_SIZE for _ in range(GRID_SIZE)]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * (BLOCK_SIZE + SPACING)
            y1 = row * (BLOCK_SIZE + SPACING)
            x2 = x1 + BLOCK_SIZE
            y2 = y1 + BLOCK_SIZE
            block = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mask = (s > 40) & (v < 255)
            h_masked = h[mask]
            color = classify_color(h_masked)
            board[row][col] = color_board.get(color, -1)

    board_hash = hashlib.md5(np.array(board, dtype=np.int8).tobytes()).hexdigest()
    print(f"hash: {board_hash}")
    for r in board:
        print(r)

    sim_board = [[(cell if cell != -1 else 0) for cell in row] for row in board]
    best = None
    best_score = 0

    for (r1, c1, r2, c2) in all_valid_swaps(sim_board):
        if sim_board[r1][c1] == sim_board[r2][c2]:
            continue
        score = simulate_swap_and_score(sim_board, r1, c1, r2, c2)
        if score > best_score:
            best_score = score
            best = ((r1, c1), (r2, c2))

    if best is None:
        return

    (r1, c1), (r2, c2) = best
    x1, y1 = c1 + 1, r1 + 1
    x2, y2 = c2 + 1, r2 + 1
    print(f"move: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"points: {best_score}")

    sx1, sy1 = block_to_screen(c1, r1)
    sx2, sy2 = block_to_screen(c2, r2)

    pydirectinput.moveTo(sx1, sy1)
    wait_for_single_click()

    time.sleep(0.3)

    pydirectinput.moveTo(sx2, sy2)
    wait_for_single_click()

    time.sleep(0.3)

    pydirectinput.press('esc')
    pydirectinput.moveTo(1300, 835)

if __name__ == "__main__":
    f2_was_pressed = False
    while True:
        if keyboard.is_pressed('f2'):
            if not f2_was_pressed:
                main()
                f2_was_pressed = True
        else:
            f2_was_pressed = False

        time.sleep(0.01)