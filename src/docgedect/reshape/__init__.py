import cv2
import numpy as np
from math import hypot

# -------------------- 유틸: 4점 정렬(가장 견고한 방식) --------------------
def order_points_robust(pts: np.ndarray) -> np.ndarray:
    """
    (4,2) 또는 (4,1,2) -> TL, TR, BR, BL 순으로 정렬.
    y 기준 상단 2점/하단 2점 분리 후, 각 라인에서 x로 좌/우 결정.
    """
    p = pts.reshape(-1, 2).astype(np.float32)
    # y 오름차순
    idx = np.argsort(p[:, 1])
    top2 = p[idx[:2]]
    bottom2 = p[idx[2:]]
    # 상단 좌/우
    tl, tr = top2[np.argsort(top2[:, 0])]
    # 하단 좌/우
    bl, br = bottom2[np.argsort(bottom2[:, 0])]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # CCW 보정(필수 아님이지만 일관성 위해)
    # 다각형 면적이 음수면 시계방향 -> 반시계로 뒤집기
    area = 0.5 * np.linalg.det(np.stack([ordered[1] - ordered[0],
                                         ordered[3] - ordered[0]], axis=0))
    if area < 0:
        ordered = np.array([tl, bl, br, tr], dtype=np.float32)
    return ordered

# -------------------- 1단계: 4점으로 원근변환 --------------------
def four_point_warp_with_H(image: np.ndarray, src_pts: np.ndarray):
    src = order_points_robust(src_pts)
    # 목적지 크기 결정(상·하/좌·우 변 길이 최대)
    w1 = hypot(*(src[1] - src[0]))
    w2 = hypot(*(src[2] - src[3]))
    h1 = hypot(*(src[3] - src[0]))
    h2 = hypot(*(src[2] - src[1]))
    W = int(max(w1, w2)); H = int(max(h1, h2))
    W = max(W, 50); H = max(H, 50)

    dst = np.array([[0, 0],
                    [W - 1, 0],
                    [W - 1, H - 1],
                    [0, H - 1]], dtype=np.float32)
    H1 = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H1, (W, H))
    return warped, H1, (W, H)

# -------------------- 컨투어 분할 유틸(짧은 호 선택) --------------------
def circular_shorter_slice(points: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    points: (N,2), i->j 사이를 '더 짧은' 원호 방향으로 슬라이스.
    """
    N = len(points)
    fwd = (j - i) % N
    bwd = (i - j) % N
    if fwd <= bwd:  # 정방향이 더 짧음
        if i <= j:
            return points[i:j+1]
        else:
            return np.concatenate([points[i:], points[:j+1]], axis=0)
    else:           # 역방향이 더 짧음 (순서를 i->j로 유지하려면 역방향을 뒤집어줌)
        if j <= i:
            seg = points[j:i+1][::-1]
        else:
            seg = np.concatenate([points[j:], points[:i+1]], axis=0)[::-1]
        return seg

def cumulative_lengths(polyline: np.ndarray) -> np.ndarray:
    if len(polyline) <= 1:
        return np.array([0.0], dtype=np.float32)
    d = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)]).astype(np.float32)

def lerp(A: np.ndarray, B: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = A.reshape(1, 2).astype(np.float32)
    B = B.reshape(1, 2).astype(np.float32)
    t = t.reshape(-1, 1).astype(np.float32)
    return (1 - t) * A + t * B

# -------------------- 2단계: 모든 점 기반 정밀 보정 --------------------
def build_correspondences_axis_aligned(cnt_warped: np.ndarray,
                                       target_size=None,
                                       padding: int = 8):
    """
    1단계로 워프된 좌표계의 컨투어(cnt_warped: (N,1,2) or (N,2))에서
    축정렬 바운딩 박스 기준으로 TL,TR,BR,BL 포인트를 찾아
    각 변에 호길이 비율로 매핑할 (src_pts, dst_pts) 생성.
    """
    cnt = cnt_warped.reshape(-1, 2).astype(np.float32)
    N = len(cnt)

    # 축정렬 바운딩 박스
    x_min, y_min = np.min(cnt, axis=0)
    x_max, y_max = np.max(cnt, axis=0)

    # 출력 크기
    W = int(round(x_max - x_min))
    H = int(round(y_max - y_min))
    W = max(W, 10); H = max(H, 10)
    if target_size is not None:
        W, H = target_size
    W += 2 * padding
    H += 2 * padding

    # 타깃 사각형 네 모서리
    TL = np.array([padding, padding], dtype=np.float32)
    TR = np.array([W - 1 - padding, padding], dtype=np.float32)
    BR = np.array([W - 1 - padding, H - 1 - padding], dtype=np.float32)
    BL = np.array([padding, H - 1 - padding], dtype=np.float32)

    # 박스 코너(원 좌표계상)
    boxTL = np.array([x_min, y_min], dtype=np.float32)
    boxTR = np.array([x_max, y_min], dtype=np.float32)
    boxBR = np.array([x_max, y_max], dtype=np.float32)
    boxBL = np.array([x_min, y_max], dtype=np.float32)

    # 각 corner에 가장 가까운 컨투어 인덱스 (중복 피하기)
    corners_src = [boxTL, boxTR, boxBR, boxBL]
    used = set()
    corner_idx = []
    for c in corners_src:
        d2 = np.sum((cnt - c) ** 2, axis=1)
        order = np.argsort(d2)
        # 유니크 인덱스 선택
        idx_sel = next(int(k) for k in order if int(k) not in used)
        used.add(idx_sel)
        corner_idx.append(idx_sel)
    tl_i, tr_i, br_i, bl_i = corner_idx

    # 네 변을 '짧은 경로'로 슬라이스 (항상 해당 변에 대응)
    seg_top    = circular_shorter_slice(cnt, tl_i, tr_i)
    seg_right  = circular_shorter_slice(cnt, tr_i, br_i)
    seg_bottom = circular_shorter_slice(cnt, br_i, bl_i)
    seg_left   = circular_shorter_slice(cnt, bl_i, tl_i)

    # 각 세그먼트에서 누적 길이 비율 -> 직선 변에 보간
    def seg_map(seg, A, B):
        if len(seg) == 1:
            return seg.astype(np.float32), np.array([A], dtype=np.float32)
        cl = cumulative_lengths(seg)
        t = cl / (cl[-1] if cl[-1] > 0 else 1.0)
        dst = lerp(A, B, t)
        return seg.astype(np.float32), dst.astype(np.float32)

    src_top,    dst_top    = seg_map(seg_top,    TL, TR)
    src_right,  dst_right  = seg_map(seg_right,  TR, BR)
    src_bottom, dst_bottom = seg_map(seg_bottom, BR, BL)
    src_left,   dst_left   = seg_map(seg_left,   BL, TL)

    src_pts = np.vstack([src_top, src_right, src_bottom, src_left]).astype(np.float32)
    dst_pts = np.vstack([dst_top, dst_right, dst_bottom, dst_left]).astype(np.float32)

    # 코너 보강(중복 있어도 무방)
    src_pts = np.vstack([src_pts, cnt[[tl_i, tr_i, br_i, bl_i]]])
    dst_pts = np.vstack([dst_pts, np.array([TL, TR, BR, BL], dtype=np.float32)])

    return src_pts, dst_pts, (W, H)

def two_stage_document_warp(image_bgr: np.ndarray,
                            screen_cnt: np.ndarray,
                            smooth_contour: np.ndarray,
                            target_size: tuple[int, int] | None = None,
                            padding: int = 8,
                            ransac_reproj_threshold: float = 3.0):
    """
    1) screen_cnt(4점)로 1차 워프(H1)
    2) smooth_contour를 H1로 투영 → 모든 점-변 대응으로 H2 추정 → 정밀 보정
    return: warped_final, H_total, warped_stage1, H1, H2
    """
    # 1단계
    warped1, H1, _ = four_point_warp_with_H(image_bgr, screen_cnt)

    # smooth_contour를 1단계 좌표계로
    cnt = np.asarray(smooth_contour, dtype=np.float32)
    if cnt.ndim == 2 and cnt.shape[1] == 2:
        cnt = cnt.reshape(-1, 1, 2)
    elif not (cnt.ndim == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2):
        raise ValueError("smooth_contour는 (N,1,2) 또는 (N,2) 형태여야 합니다.")
    cnt_warped = cv2.perspectiveTransform(cnt, H1)  # (N,1,2)

    # 2단계: 축정렬 바운딩 박스로 코너 찾고 모든 점 매칭
    src_pts, dst_pts, (W2, H2out) = build_correspondences_axis_aligned(
        cnt_warped, target_size=target_size, padding=padding
    )
    H2, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold)
    if H2 is None:
        H2, _ = cv2.findHomography(src_pts, dst_pts, 0)
    if H2 is None:
        raise RuntimeError("2단계 호모그래피 추정 실패")

    warped_final = cv2.warpPerspective(warped1, H2, (W2, H2out))
    H_total = H2 @ H1
    return warped_final, H_total, warped1, H1, H2