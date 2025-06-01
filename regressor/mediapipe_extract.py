import cv2
import mediapipe as mp
import json
import numpy as np
from typing import List, Dict, Any

import os
import sys

class MediaPipePoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
        # Initialize pose detection only
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def extract_keypoints_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract pose keypoints from an image and return in OpenPose JSON format"""
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Process pose only
        pose_results = self.pose.process(image_rgb)
        
        # Create the exact structure as your JSON
        result = {
            "version": 1.3,
            "people": []
        }
        
        if pose_results.pose_landmarks:
            person = {
                "person_id": [-1],
                "pose_keypoints_2d": self._extract_pose_keypoints_25(pose_results.pose_landmarks, width, height),
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            
            result["people"].append(person)
        
        return result

    def _extract_pose_keypoints_25(self, landmarks, width: int, height: int) -> List[float]:
        """
        Extract 25 pose keypoints in OpenPose BODY_25 format
        OpenPose BODY_25 keypoint order:
        0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
        5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
        10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
        15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe,
        20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel
        """
        keypoints = []
        
        # MediaPipe landmark indices (33 total landmarks)
        mp_landmarks = landmarks.landmark
        
        # OpenPose keypoint mapping from MediaPipe landmarks
        # Some points need to be computed or approximated
        
        # 0: Nose
        nose = mp_landmarks[0]
        keypoints.extend([nose.x * width, nose.y * height, nose.visibility])
        
        # 1: Neck (approximate as midpoint between shoulders)
        left_shoulder = mp_landmarks[11]
        right_shoulder = mp_landmarks[12]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
        neck_y = (left_shoulder.y + right_shoulder.y) / 2 * height
        neck_conf = min(left_shoulder.visibility, right_shoulder.visibility)
        keypoints.extend([neck_x, neck_y, neck_conf])
        
        # 2: Right Shoulder
        keypoints.extend([right_shoulder.x * width, right_shoulder.y * height, right_shoulder.visibility])
        
        # 3: Right Elbow
        right_elbow = mp_landmarks[14]
        keypoints.extend([right_elbow.x * width, right_elbow.y * height, right_elbow.visibility])
        
        # 4: Right Wrist
        right_wrist = mp_landmarks[16]
        keypoints.extend([right_wrist.x * width, right_wrist.y * height, right_wrist.visibility])
        
        # 5: Left Shoulder
        keypoints.extend([left_shoulder.x * width, left_shoulder.y * height, left_shoulder.visibility])
        
        # 6: Left Elbow
        left_elbow = mp_landmarks[13]
        keypoints.extend([left_elbow.x * width, left_elbow.y * height, left_elbow.visibility])
        
        # 7: Left Wrist
        left_wrist = mp_landmarks[15]
        keypoints.extend([left_wrist.x * width, left_wrist.y * height, left_wrist.visibility])
        
        # 8: MidHip (approximate as midpoint between hips)
        left_hip = mp_landmarks[23]
        right_hip = mp_landmarks[24]
        mid_hip_x = (left_hip.x + right_hip.x) / 2 * width
        mid_hip_y = (left_hip.y + right_hip.y) / 2 * height
        mid_hip_conf = min(left_hip.visibility, right_hip.visibility)
        keypoints.extend([mid_hip_x, mid_hip_y, mid_hip_conf])
        
        # 9: Right Hip
        keypoints.extend([right_hip.x * width, right_hip.y * height, right_hip.visibility])
        
        # 10: Right Knee
        right_knee = mp_landmarks[26]
        keypoints.extend([right_knee.x * width, right_knee.y * height, right_knee.visibility])
        
        # 11: Right Ankle
        right_ankle = mp_landmarks[28]
        keypoints.extend([right_ankle.x * width, right_ankle.y * height, right_ankle.visibility])
        
        # 12: Left Hip
        keypoints.extend([left_hip.x * width, left_hip.y * height, left_hip.visibility])
        
        # 13: Left Knee
        left_knee = mp_landmarks[25]
        keypoints.extend([left_knee.x * width, left_knee.y * height, left_knee.visibility])
        
        # 14: Left Ankle
        left_ankle = mp_landmarks[27]
        keypoints.extend([left_ankle.x * width, left_ankle.y * height, left_ankle.visibility])
        
        # 15: Right Eye
        right_eye = mp_landmarks[5]
        keypoints.extend([right_eye.x * width, right_eye.y * height, right_eye.visibility])
        
        # 16: Left Eye
        left_eye = mp_landmarks[2]
        keypoints.extend([left_eye.x * width, left_eye.y * height, left_eye.visibility])
        
        # 17: Right Ear
        right_ear = mp_landmarks[8]
        keypoints.extend([right_ear.x * width, right_ear.y * height, right_ear.visibility])
        
        # 18: Left Ear
        left_ear = mp_landmarks[7]
        keypoints.extend([left_ear.x * width, left_ear.y * height, left_ear.visibility])
        
        # 19: Left Big Toe
        left_big_toe = mp_landmarks[31]
        keypoints.extend([left_big_toe.x * width, left_big_toe.y * height, left_big_toe.visibility])
        
        # 20: Left Small Toe
        left_small_toe = mp_landmarks[29]
        keypoints.extend([left_small_toe.x * width, left_small_toe.y * height, left_small_toe.visibility])
        
        # 21: Left Heel
        left_heel = mp_landmarks[30]
        keypoints.extend([left_heel.x * width, left_heel.y * height, left_heel.visibility])
        
        # 22: Right Big Toe
        right_big_toe = mp_landmarks[32]
        keypoints.extend([right_big_toe.x * width, right_big_toe.y * height, right_big_toe.visibility])
        
        # 23: Right Small Toe
        right_small_toe = mp_landmarks[30]  # MediaPipe doesn't have separate right small toe
        keypoints.extend([right_small_toe.x * width, right_small_toe.y * height, right_small_toe.visibility])
        
        # 24: Right Heel
        right_heel = mp_landmarks[29]  # MediaPipe index for right heel
        keypoints.extend([right_heel.x * width, right_heel.y * height, right_heel.visibility])
        
        return keypoints

    def visualize_keypoints(self, image_path: str, results: Dict[str, Any] = None, output_path: str = None, show_image: bool = True):
        """Visualize the extracted pose keypoints on the image"""
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # If no results provided, extract them
        if results is None:
            results = self.extract_keypoints_from_image(image_path)
        
        # Draw keypoints if person detected
        if results["people"]:
            person = results["people"][0]
            
            # Draw pose keypoints (25 points)
            pose_keypoints = person["pose_keypoints_2d"]
            pose_connections = [
                # Head
                (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                # Body
                (1, 8), (8, 9), (8, 12), (9, 10), (10, 11), (12, 13), (13, 14),
                # Eyes and ears
                (0, 15), (0, 16), (15, 17), (16, 18),
                # Feet
                (11, 22), (11, 24), (14, 19), (14, 21), (19, 20), (22, 23)
            ]
            
            # Draw pose skeleton
            for i in range(0, len(pose_keypoints), 3):
                x, y, conf = pose_keypoints[i:i+3]
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 8, (0, 255, 0), -1)  # Green circles
                    # Add keypoint number
                    cv2.putText(image, str(i//3), (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw pose connections
            for connection in pose_connections:
                pt1_idx, pt2_idx = connection
                x1, y1, conf1 = pose_keypoints[pt1_idx*3:pt1_idx*3+3]
                x2, y2, conf2 = pose_keypoints[pt2_idx*3:pt2_idx*3+3]
                
                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add legend
            cv2.putText(image, "Pose Keypoints (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the image if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to: {output_path}")
        
        # Show the image if requested
        if show_image:
            # Resize image if it's too large for display
            height, width = image.shape[:2]
            if height > 800 or width > 800:
                scale = min(800/height, 800/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            cv2.imshow('Pose Detection Results', image)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def process_and_visualize(self, image_path: str, save_json: bool = True, save_visualization: bool = True, show_image: bool = True):
        """Process image, extract keypoints, save JSON, and visualize results"""
        try:
            print(f"Processing image: {image_path}")
            
            # Extract keypoints
            results = self.extract_keypoints_from_image(image_path)
            
            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_output = f"{base_name}_keypoints.json"
            viz_output = f"{base_name}_visualization.jpg"
            
            # Save JSON if requested
            if save_json:
                self.save_to_json(results, json_output)
                print(f"✓ JSON saved to: {json_output}")
            
            # Show results
            if results["people"]:
                person = results["people"][0]
                print(f"✓ Detected pose keypoints: {len(person['pose_keypoints_2d'])//3} points")
                
                # Visualize keypoints
                self.visualize_keypoints(
                    image_path, 
                    results, 
                    output_path=viz_output if save_visualization else None,
                    show_image=show_image
                )
                
                if save_visualization:
                    print(f"✓ Visualization saved to: {viz_output}")
            else:
                print("⚠ No person detected in the image")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

    def save_to_json(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file with proper formatting"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, separators=(',', ':'))

    def process_image_and_save(self, image_path: str, output_json_path: str):
        """Process an image and save results to JSON file"""
        try:
            results = self.extract_keypoints_from_image(image_path)
            self.save_to_json(results, output_json_path)
            print(f"Successfully processed {image_path}")
            print(f"Results saved to {output_json_path}")
            
            if results["people"]:
                person = results["people"][0]
                print(f"✓ Detected pose keypoints: {len(person['pose_keypoints_2d'])//3} points")
            else:
                print("⚠ No person detected in the image")
                
        except Exception as e:
            print(f"Error processing image: {e}")

# Example usage
def main():
    # Initialize the extractor
    extractor = MediaPipePoseExtractor()
    
    # Get image path from command line argument or prompt user
    image_path = r"c:\Users\tuongquan\Downloads\laudaitinhai.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Please provide a valid image path.")
        return
    
    # Generate output filename based on input
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_json = f"{base_name}.json"
    
    print(f"Processing image: {image_path}")
    print(f"Output will be saved to: {output_json}")
    
    extractor.process_image_and_save(image_path, output_json)
    extractor.visualize_keypoints(image_path, output_path=f"{base_name}_visualization.jpg", show_image=True)

if __name__ == "__main__":
    main()