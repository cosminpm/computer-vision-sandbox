import time

import cv2 as cv
import numpy as np
import pyautogui
from PIL import ImageGrab

# left, top, right, bottom
bbox = (81, 600, 1079, 820)
ball: np.ndarray = cv.imread("ball.png", cv.IMREAD_UNCHANGED)
player: np.ndarray = cv.imread("player.png", cv.IMREAD_UNCHANGED)
cv.namedWindow("Template Matching", cv.WINDOW_NORMAL)
time.sleep(3)


class Player:
    current_key: str = ""

    def move_right(self):
        self._move("d")

    def move_left(self):
        self._move("a")

    def _move(self, new_key: str):
        if self.current_key != new_key:
            self.release_movement()
            self.current_key = new_key
            pyautogui.keyDown(self.current_key)

    def release_movement(self):
        pyautogui.keyUp("a")
        pyautogui.keyUp("d")
        pyautogui.keyUp("w")
        self.current_key = ""


def take_screenshot():
    p = Player()
    while True:
        screenshot = ImageGrab.grab(bbox)
        screenshot_np = np.array(screenshot)
        screen = cv.cvtColor(screenshot_np, cv.COLOR_RGB2BGR)

        ball_position = compare(screen, ball)
        player_position = compare(screen, player)

        if ball_position and player_position:
            ball_center = (
                ball_position[0][0] + ball.shape[1] // 2,
                ball_position[0][1] + ball.shape[0] // 2,
            )
            player_center = (
                player_position[0][0] + player.shape[1] // 2,
                player_position[0][1] + player.shape[0] // 2,
            )

            if ball_center[0] > player_center[0]:
                p.move_right()
            else:
                p.move_left()

        elif ball_position or player_position:
            p.move_right()
        else:
            p.release_movement()

        cv.imshow("Template Matching", screen)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()


def compare(screen: np.ndarray, desired: np.ndarray) -> tuple | None:
    result = cv.matchTemplate(screen, desired, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    threshold = 0.45
    if max_val >= threshold:
        desired_w = desired.shape[1]
        desired_h = desired.shape[0]
        top_left = max_loc
        bottom_right = (top_left[0] + desired_w, top_left[1] + desired_h)
        cv.rectangle(
            screen, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4
        )
        return top_left, bottom_right
    return None


if __name__ == "__main__":
    take_screenshot()
