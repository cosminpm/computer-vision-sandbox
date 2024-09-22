import os
from pathlib import Path

import cv2
from numpy import ndarray

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
KEYPOINTS_LIMIT = 20


def extract_sift_features(image: ndarray) -> tuple[list, list]:
    """Extract the SIFT features (keypoints and descriptors) of the image.

    Args:
    ----
        image (ndarray): The image we are going to extract the features

    Returns:
    -------
        A tuple with the keypoints and the descriptors.

    """
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def load_images_and_extract_features(folder_path: str) -> list[tuple]:
    """Load all the image in the folder and extract their SIFT features.

    Args:
    ----
        folder_path (str): Folder path

    Returns:
    -------
        A list with all the features.

    """
    image_features: list = []
    for image_name in os.listdir(folder_path):
        image_path: Path = Path(folder_path) / image_name
        image: ndarray = cv2.imread(str(image_path))
        keypoints, descriptors = extract_sift_features(image)
        image_features.append((image_name, image, keypoints, descriptors))
    return image_features


def find_best_match(frame: ndarray, image_features: list[tuple]) -> tuple | None:
    """Find the best match for the frame and their SIFT features.

    Args:
    ----
        frame (ndarray): The current frame
        image_features (list[tuple]): The image features

    Returns:
    -------
        Returns the best match for each frame.

    """
    keypoints_frame, descriptors_frame = extract_sift_features(frame)
    if descriptors_frame is None:
        return None

    best_match = None
    max_good_matches: int = 0
    best_good_matches = []
    best_keypoints = None
    keypoints_best_image = None

    for image_name, image, keypoints, descriptors in image_features:
        matches = bf.knnMatch(descriptors_frame, descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match = (image_name, image)
            best_good_matches = good_matches
            best_keypoints = keypoints_frame
            keypoints_best_image = keypoints
    if max_good_matches > KEYPOINTS_LIMIT:
        return best_match, best_keypoints, keypoints_best_image, best_good_matches
    return None


def draw_matches_on_frame(frame: ndarray, best_image: ndarray, keypoints_frame: list,
                          keypoints_best_image: list, good_matches: list) -> ndarray:
    """Draw SIFT keypoints matches in the frame.

    Args:
    ----
        frame (ndarray): Image to be draw.
        best_image (ndarray): The best match
        keypoints_frame (list): The keypoints in the frame
        keypoints_best_image (list): THe keypoints in the best match.
        good_matches (int): The number of good matches

    Returns:
    -------
        A list of frames

    """
    matches_img = cv2.drawMatches(
        frame,
        keypoints_frame,
        best_image,
        keypoints_best_image,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    overlay_height, overlay_width = matches_img.shape[:2]

    if overlay_height > frame.shape[0] or overlay_width > frame.shape[1]:
        scale_factor = min(frame.shape[0] / overlay_height, frame.shape[1] / overlay_width)
        matches_img: ndarray = cv2.resize(matches_img, (0, 0), fx=scale_factor, fy=scale_factor)

    frame[0: matches_img.shape[0], 0: matches_img.shape[1]] = matches_img

    return frame


def draw_keypoints_on_image(path: str) -> None:
    """Draw keypoints on the given image."""
    image: ndarray = cv2.imread(str(path))
    keypoints, descriptors = extract_sift_features(image)
    img_key: ndarray = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('./image_with_keypoints.jpg', img_key)


def main(folder_path: str) -> None:
    """Run the main application.

    Args:
    ----
        folder_path (str): The folder with the boardgames images.

    """
    image_features = load_images_and_extract_features(folder_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        match_result: tuple = find_best_match(frame, image_features)

        if match_result is not None:
            best_match, keypoints_frame, keypoints_best_image, good_matches = match_result
            _, best_image = best_match
            frame = draw_matches_on_frame(
                frame, best_image, keypoints_frame, keypoints_best_image, good_matches
            )

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("./boardgame-images")
    draw_keypoints_on_image("./boardgame-images/rey-paparajote.jpg")
