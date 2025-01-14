import os
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from util.detect_lm68 import detect_68p, load_lm_graph
from util.skin_mask import get_skin_mask
from util.preprocess import align_img
from util.load_mats import load_lm3d
from models import create_model
from util.visualizer import MyVisualizer
from util.generate_list import check_list, write_list


# from options.test_options import TestOptions
from options.facellm_options import TestOptions

def detect_keypoints_and_save(image_path):
    """
    Detects 5 facial keypoints using MTCNN and saves them in a text file.

    Args:
        image_path (str): Path to the face image.

    Returns:
        str: Path to the saved keypoints file.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.asarray(image)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces
    detection = detector.detect_faces(image_array)

    # Check if any face was detected
    if len(detection) == 0:
        raise ValueError("No face detected in the image.")

    # Extract keypoints from the first detected face
    keypoints = detection[0]['keypoints']

    # Format keypoints for saving
    keypoints_list = [
        [keypoints['left_eye'][0], keypoints['left_eye'][1]],
        [keypoints['right_eye'][0], keypoints['right_eye'][1]],
        [keypoints['nose'][0], keypoints['nose'][1]],
        [keypoints['mouth_left'][0], keypoints['mouth_left'][1]],
        [keypoints['mouth_right'][0], keypoints['mouth_right'][1]],
    ]

    # Create the detections folder
    detections_dir = os.path.join(os.path.dirname(image_path), 'detections')
    os.makedirs(detections_dir, exist_ok=True)

    # Save keypoints to a text file
    keypoints_file = os.path.join(detections_dir, os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(keypoints_file, 'w') as f:
        for point in keypoints_list:
            f.write(f"{point[0]:.2f}\t{point[1]:.2f}\n")

    print(f"Keypoints saved to: {keypoints_file}")
    return keypoints_file

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def process_image(rank, opt, image_path='examples', temp_dir='./temp'):
    """
    Takes a single image and runs face reconstruction to return 3DMM coefficients and landmarks.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: A dictionary containing the 3DMM coefficients and landmarks.
    """
    # Step 1: Set up temporary directory
    temp_dir = tempfile.mkdtemp()
    # temp_dir = temp_dir
    temp_image_path = os.path.join(temp_dir, os.path.basename(image_path))
    print("TEMP IMAGE PATH:", temp_image_path)
    shutil.copy(image_path, temp_image_path)
    if not os.path.isfile(temp_image_path):
        raise FileNotFoundError(f"Temp image file not found: {temp_image_path}")
    img_name = temp_image_path.split(os.path.sep)[-1].replace('.png','').replace('.jpg','')

    # Step 2: Detect keypoints and save them
    landmark_path = detect_keypoints_and_save(temp_image_path)
    if not os.path.isfile(landmark_path):
        raise FileNotFoundError(f"Temp landmark file not found: {landmark_path}")

    opt.img_folder = temp_dir
    name = temp_image_path

    # Step 4: Initialize face reconstruction model
    opt = TestOptions().parse()
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)


    # Step 5: Prepare input for reconstruction
    lm3d_std = load_lm3d(opt.bfm_folder)
    im_tensor, lm_tensor = read_data(temp_image_path, landmark_path, lm3d_std)

    # Step 6: Run face reconstruction
    data = {'imgs': im_tensor.to(device), 'lms': lm_tensor.to(device)}
    model.set_input(data)
    model.test()

    # Save intermediate files
    visuals = model.get_current_visuals()  # get image results
    visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
        save_results=True, count=0, name=name.split(os.path.sep)[-1], add_image=False)

    model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
    model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients


    coeff, landmarks = model.get_coeff()  # Replace with the correct method to retrieve coefficients

    # Step 7: Clean up temporary files
    shutil.rmtree(temp_dir)

    # Step 8: Return results
    return {
        "coefficients": coeff,
        "landmarks": landmarks
    }

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    result = process_image(0, opt, opt.img_path)
    print("Coefficients:", len(result['coefficients']))
    print("Landmarks:", len(result['landmarks']))
  
    