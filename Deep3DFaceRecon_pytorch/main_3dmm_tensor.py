import os
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import tensor2im
from models import create_model
from util.visualizer import MyVisualizer
from options.facellm_options import TestOptions

# from options.test_options import TestOptions
from options.facellm_options import TestOptions

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_INTERMEDIATES = True

def detect_keypoints(img_tensor):
    """
    Detects 5 facial keypoints using MTCNN.

    Args:
        img_tensor (torch.Tensor): Image tensor.

    Returns:
        list: List of detected keypoints.
    """
    # Convert tensor to PIL image
    img = tensor2im(img_tensor)
    img = Image.fromarray(img)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces
    detection, _ = detector.detect(img)

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

    return keypoints_list

def process_image(rank, opt, img_tensor):
    # Create model
    model = create_model(opt)
    model.setup(opt)
    model.device = torch.device(f'cuda:{rank}')
    model.parallelize()
    model.eval()

    # Load landmarks
    lm3d_std = load_lm3d(opt.bfm_folder)

    # Convert image tensor to appropriate format
    img_tensor = img_tensor.unsqueeze(0).to(model.device)  # Add batch dimension and move to device

    # Detect keypoints
    keypoints = detect_keypoints(img_tensor)

    # Align image based on keypoints and standard landmarks
    img_aligned, lm_aligned = align_img(img_tensor, keypoints, lm3d_std)

    # Convert aligned image and landmarks to tensors
    im_tensor = torch.tensor(np.array(img_aligned)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    lm_tensor = torch.tensor(lm_aligned).unsqueeze(0)

    # Generate 3DMM parameters
    data = {'imgs': im_tensor.to(model.device), 'lms': lm_tensor.to(model.device)}
    model.set_input(data)
    model.test()

    # Save intermediate files if needed
    if SAVE_INTERMEDIATES:
        visualizer = MyVisualizer(opt)
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, 0, opt.epoch, dataset="aligned_image", save_results=True, count=0, name="aligned_image", add_image=False)
        model.save_mesh(os.path.join(visualizer.img_dir, 'aligned_image.obj'))  # save reconstruction meshes
        model.save_coeff(os.path.join(visualizer.img_dir, 'aligned_image.mat'))  # save predicted coefficients

    coefficients, landmarks = model.get_coeff()  # Replace with the correct method to retrieve coefficients

    return {
        "coefficients": coefficients,
        "landmarks": landmarks
    }

    
def get_3dmm(img_tensor: torch.Tensor):
    # Get cuda device
    rank = torch.cuda.current_device()

    # Initialize model params
    opt = TestOptions().parse()
    opt.name = "face_recon_v0"
    opt.epoch = 20
    opt.use_opengl = False

    # Set checkpoints dir
    opt.checkpoints_dir = os.path.join(PACKAGE_DIR, 'checkpoints')
    opt.bfm_folder = os.path.join(PACKAGE_DIR, 'BFM')

    # Generate 3dmm params
    return process_image(rank, opt, img_tensor)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    result = process_image(0, opt, opt.img_path)
    print("Coefficients:", len(result['coefficients']))
    print("Landmarks:", len(result['landmarks']))
  
    