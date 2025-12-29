import cv2
import numpy as np
import pyvirtualcam
import sys
import argparse
from video_mixer import resize_and_pad, mix_frames

class VideoMixer:
    def __init__(self, source1, source2, mode='add', size_mode='max', output_size=None, padding_color='black', 
                 alpha=0.5, threshold=128, gamma=1.0, posterize_levels=4):
        self.cap1 = self._open_source(source1)
        self.cap2 = self._open_source(source2)
        self.mode = mode.lower()
        self.size_mode = size_mode.lower()
        self.padding_color = padding_color.lower()
        
        # Parameters for filters
        self.alpha = alpha
        self.threshold = threshold
        self.gamma = gamma
        self.posterize_levels = posterize_levels
        
        # Get native sizes
        self.w1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_size:
            self.out_w, self.out_h = output_size
        elif self.size_mode == 's1':
            self.out_w, self.out_h = self.w1, self.h1
        elif self.size_mode == 's2':
            self.out_w, self.out_h = self.w2, self.h2
        elif self.size_mode == 'overlap':
            self.out_w, self.out_h = min(self.w1, self.w2), min(self.h1, self.h2)
        else: # max
            self.out_w, self.out_h = max(self.w1, self.w2), max(self.h1, self.h2)

    def _open_source(self, source):
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(source)


    def mix_frames(self, frame1, frame2, t=0):
        # Resize both to output size
        resized1 = resize_and_pad(frame1, (self.out_w, self.out_h), self.padding_color)
        resized2 = resize_and_pad(frame2, (self.out_w, self.out_h), self.padding_color)

        return mix_frames(resized1, resized2, self.mode, self.alpha, self.threshold, self.gamma, self.posterize_levels, t)

    def run(self, virtual_out=False, save_path=None):
        cam = None
        if virtual_out:
            try:
                cam = pyvirtualcam.Camera(width=self.out_w, height=self.out_h, fps=30)
                print(f"Using virtual camera: {cam.device}")
            except Exception as e:
                print(f"Error starting virtual camera: {e}")
                virtual_out = False

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, 30.0, (self.out_w, self.out_h))
            print(f"Saving to: {save_path}")

        print("Press 'q' to quit")
        t = 0
        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1:
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = self.cap1.read()
            if not ret2:
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                break

            mixed = self.mix_frames(frame1, frame2, t=t)
            t += 0.1

            if virtual_out and cam:
                cam.send(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
            
            if writer:
                writer.write(mixed)
            
            cv2.imshow('Video Mixer', mixed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        if cam: cam.close()
        if writer: writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Mixer Tool')
    parser.add_argument('--s1', type=str, required=True, help='Source 1')
    parser.add_argument('--s2', type=str, required=True, help='Source 2')
    parser.add_argument('--mode', type=str, default='add', help='Mixing mode')
    parser.add_argument('--size-mode', type=str, default='max', choices=['s1', 's2', 'max', 'overlap'], help='Output size mode')
    parser.add_argument('--width', type=int, help='Manual output width')
    parser.add_argument('--height', type=int, help='Manual output height')
    parser.add_argument('--padding', type=str, default='black', choices=['black', 'white'], help='Padding color')
    parser.add_argument('--virtual', action='store_true', help='Output to virtual camera')
    parser.add_argument('--save', type=str, help='Save to video file path')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for blending')
    parser.add_argument('--threshold', type=int, default=128, help='Threshold for keying')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value')
    parser.add_argument('--posterize', type=int, default=4, help='Posterize levels')

    args = parser.parse_args()
    
    output_size = (args.width, args.height) if args.width and args.height else None
    
    mixer = VideoMixer(args.s1, args.s2, mode=args.mode, size_mode=args.size_mode, output_size=output_size, 
                       padding_color=args.padding, alpha=args.alpha, threshold=args.threshold, 
                       gamma=args.gamma, posterize_levels=args.posterize)
    mixer.run(virtual_out=args.virtual, save_path=args.save)