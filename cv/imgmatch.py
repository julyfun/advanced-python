import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


def _create_feature_detector(name: str):
    name = name.upper()
    if name == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create()
        # Fallback to ORB if SIFT unavailable
        return cv2.ORB_create(nfeatures=5000)
    if name == "ORB":
        return cv2.ORB_create(nfeatures=5000)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    # Default
    return cv2.SIFT_create() if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=5000)


def _match_descriptors(desc1: np.ndarray, desc2: np.ndarray, is_binary: bool, k: int = 2):
    if is_binary:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # FLANN for SIFT/AKAZE (float descriptors)
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=64)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # FLANN needs float32
        if desc1 is not None and desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2 is not None and desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
    matches = matcher.knnMatch(desc1, desc2, k=k)
    return matches


def _ratio_test_filter(matches: List[List[cv2.DMatch]], ratio: float = 0.75) -> List[cv2.DMatch]:
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _extract_matched_keypoints(kp1, kp2, matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


def _compute_transforms(pts1: np.ndarray, pts2: np.ndarray, ransac_thresh: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    H, maskH = (None, None)
    A, maskA = (None, None)
    if len(pts1) >= 4:
        H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if len(pts1) >= 3:
        A, maskA = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    return H, maskH, A, maskA


def _choose_best_transform(maskH: Optional[np.ndarray], maskA: Optional[np.ndarray]) -> str:
    inliersH = int(maskH.sum()) if maskH is not None else -1
    inliersA = int(maskA.sum()) if maskA is not None else -1
    if inliersH < 0 and inliersA < 0:
        return "NONE"
    if inliersH >= inliersA:
        return "HOMOGRAPHY"
    return "AFFINE"


def _draw_matches(img1, kp1, img2, kp2, good_matches: List[cv2.DMatch]):
    # Convert to list of list format expected by drawMatchesKnn
    matchesKnn = [[m] for m in good_matches]
    vis = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matchesKnn, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis


def _ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def align_images(imgA_path: str, imgB_path: str, feature: str = "SIFT", ratio: float = 0.75, ransac_thresh: float = 3.0,
                 save_prefix: Optional[str] = None):
    # Load images
    imgA = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_COLOR)
    if imgA is None:
        raise FileNotFoundError(f"Cannot read image A at {imgA_path}")
    if imgB is None:
        raise FileNotFoundError(f"Cannot read image B at {imgB_path}")

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Detect + compute
    detector = _create_feature_detector(feature)
    kpA, desA = detector.detectAndCompute(grayA, None)
    kpB, desB = detector.detectAndCompute(grayB, None)

    if desA is None or desB is None or len(kpA) == 0 or len(kpB) == 0:
        raise RuntimeError("No descriptors found in one of the images.")

    is_binary = isinstance(detector, cv2.ORB) or isinstance(detector, cv2.AKAZE)
    matches = _match_descriptors(desA, desB, is_binary=is_binary, k=2)

    good = _ratio_test_filter(matches, ratio=ratio)

    if len(good) < 3:
        raise RuntimeError(f"Not enough good matches after ratio test: {len(good)}")

    ptsA, ptsB = _extract_matched_keypoints(kpA, kpB, good)

    H, maskH, A, maskA = _compute_transforms(ptsA, ptsB, ransac_thresh)

    choice = _choose_best_transform(maskH, maskA)

    warped = None
    M_out = None

    if choice == "HOMOGRAPHY" and H is not None:
        M_out = H
        warped = cv2.warpPerspective(imgA, H, (imgB.shape[1], imgB.shape[0]))
    elif choice == "AFFINE" and A is not None:
        M_out = A
        warped = cv2.warpAffine(imgA, A, (imgB.shape[1], imgB.shape[0]))

    # Visualizations
    vis_matches = _draw_matches(imgA, kpA, imgB, kpB, good)

    blend = None
    if warped is not None:
        alpha = 0.5
        blend = cv2.addWeighted(warped, alpha, imgB, 1 - alpha, 0)

    if save_prefix:
        matches_path = f"{save_prefix}_matches.jpg"
        _ensure_dir_for_file(matches_path)
        cv2.imwrite(matches_path, vis_matches)
        if warped is not None:
            warp_path = f"{save_prefix}_warped.jpg"
            blend_path = f"{save_prefix}_blend.jpg"
            cv2.imwrite(warp_path, warped)
            if blend is not None:
                cv2.imwrite(blend_path, blend)

    # Stats
    inliersH = int(maskH.sum()) if maskH is not None else 0
    inliersA = int(maskA.sum()) if maskA is not None else 0

    return {
        "feature": type(detector).__name__,
        "good_matches": len(good),
        "homography_inliers": inliersH,
        "affine_inliers": inliersA,
        "choice": choice,
        "matrix": M_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Match features between two images and estimate best transform (Homography or Affine)")
    parser.add_argument("--imgA", default="ignore-/a.jpg", help="Path to image A (source)")
    parser.add_argument("--imgB", default="ignore-/b.jpg", help="Path to image B (destination)")
    parser.add_argument("--feature", default="SIFT", choices=["SIFT", "ORB", "AKAZE"], help="Feature detector/descriptor")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio for filtering matches")
    parser.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold")
    parser.add_argument("--save-prefix", default="ignore-/imgmatch", help="Prefix for saving result images")

    args = parser.parse_args()

    result = align_images(args.imgA, args.imgB, feature=args.feature, ratio=args.ratio, ransac_thresh=args.ransac, save_prefix=args.save_prefix)

    print("Feature:", result["feature"])    
    print("Good matches:", result["good_matches"])    
    print("Homography inliers:", result["homography_inliers"])    
    print("Affine inliers:", result["affine_inliers"])    
    print("Chosen transform:", result["choice"])    
    if result["matrix"] is not None:
        print("Matrix:\n", result["matrix"])    
    else:
        print("Could not estimate a robust transform.")


if __name__ == "__main__":
    main()