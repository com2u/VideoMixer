import cv2
import numpy as np

def resize_and_pad(frame, target_size, padding_color='black'):
    if target_size is None:
        return frame
    
    tw, th = target_size
    h, w = frame.shape[:2]
    
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (nw, nh))
    
    color = (0, 0, 0) if padding_color == 'black' else (255, 255, 255)
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    
    x_offset = (tw - nw) // 2
    y_offset = (th - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    
    return canvas

def mix_frames(frame1, frame2, mode, alpha=0.5, threshold=128, gamma=1.0, posterize_levels=4, t=0):
    # Assume frames are already resized to same size
    m1_f = frame1.astype(np.float32)
    m2_f = frame2.astype(np.float32)

    m1, m2 = m1_f / 255.0, m2_f / 255.0
    res = None

    mode = mode.lower()
    if mode == 'add': res = m1_f + m2_f
    elif mode == 'subtract': res = m1_f - m2_f
    elif mode == 'multiply': res = (m1_f * m2_f) / 255.0
    elif mode == 'minimum': res = np.minimum(m1_f, m2_f)
    elif mode == 'maximum': res = np.maximum(m1_f, m2_f)
    elif mode == 'difference': res = cv2.absdiff(m1_f, m2_f)
    elif mode == 'screen': res = 255 * (1 - (1 - m1) * (1 - m2))
    elif mode == 'overlay':
        res = 255 * np.where(m1 < 0.5, 2 * m1 * m2, 1 - 2 * (1 - m1) * (1 - m2))
    elif mode == 'hard_light':
        res = 255 * np.where(m2 < 0.5, 2 * m1 * m2, 1 - 2 * (1 - m1) * (1 - m2))
    elif mode == 'soft_light':
        res = 255 * ((1 - 2 * m2) * m1**2 + 2 * m2 * m1)
    elif mode == 'color_dodge':
        res = 255 * np.divide(m1, 1 - m2 + 1e-6)
    elif mode == 'color_burn':
        res = 255 * (1 - np.divide(1 - m1, m2 + 1e-6))
    elif mode == 'linear_burn': res = m1_f + m2_f - 255
    elif mode == 'exclusion': res = m1_f + m2_f - 2 * (m1_f * m2_f) / 255.0
    elif mode == 'average': res = (m1_f + m2_f) / 2.0
    elif mode == 'negation': res = 255 - np.abs(255 - m1_f - m2_f)
    elif mode == 'divide': res = 255 * np.divide(m1, m2 + 1e-6)
    elif mode == 'power': res = 255 * np.power(m1, m2 + 1e-6)
    elif mode == 'gamma': res = 255 * np.power(m1, 1.0 / (gamma + 1e-6))
    elif mode == 'threshold':
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        res = mask_3d.astype(np.float32)
    elif mode == 'bitwise_and': res = cv2.bitwise_and(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
    elif mode == 'bitwise_or': res = cv2.bitwise_or(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
    elif mode == 'bitwise_xor': res = cv2.bitwise_xor(m1_f.astype(np.uint8), m2_f.astype(np.uint8)).astype(np.float32)
    elif mode == 'alpha_composite':
        res = m1_f * alpha + m2_f * (1 - alpha)
    elif mode == 'weighted_blend':
        res = m1_f * alpha + m2_f * (1 - alpha)
    elif mode == 'crossfade':
        cf_alpha = (np.sin(t) + 1) / 2.0
        res = m1_f * cf_alpha + m2_f * (1 - cf_alpha)
    elif mode == 'luminance_key':
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray2, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        res = m2_f * mask + m1_f * (1 - mask)
    elif mode == 'chroma_key':
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv2, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        res = m2_f * mask + m1_f * (1 - mask)
    elif mode == 'posterize':
        n = max(1, posterize_levels)
        res = np.floor(m1_f / (256.0 / n)) * (256.0 / n)
    elif mode == 'invert': res = 255 - m1_f
    elif mode == 'log': res = 255 * (np.log(1 + m1) / np.log(2))
    elif mode == 'sigmoid':
        res = 255 * (1 / (1 + np.exp(-10 * (m1 - 0.5))))
    elif mode == 'opacity': res = m1_f * alpha
    elif mode == 'normalize':
        res = cv2.normalize(m1_f, None, 0, 255, cv2.NORM_MINMAX)
    elif mode == 'clamp':
        res = np.clip(m1_f, threshold, 255)
    else: res = m1_f

    return np.clip(res, 0, 255).astype(np.uint8)