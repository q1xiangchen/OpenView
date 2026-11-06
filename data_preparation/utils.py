import cv2, io, os
from PIL import Image

from pinhole_projector import rotation_matrix, extract_pinhole_view

def uv2pose(uv):
    """
    Convert normalized uv coordinates to yaw, pitch, roll in degrees.
    center of the panorama (0.5, 0.5) -> (0, 0, 0) no rotation.
    Input:
        uv: normalized uv coordinates (u_norm, v_norm)
    Output:
        pose: (yaw, pitch, roll) in degrees
    """
    u_norm = uv[0]
    v_norm = uv[1]
    yaw = u_norm * 360.0 - 180.0
    pitch = 90.0 - v_norm * 180.0
    if yaw >= 180.0: yaw -= 360.0
    if yaw < -180.0: yaw += 360.0
    return (yaw, pitch, 0.0)

def project(img_pano, pose, fov_deg, out_size, fov_type="horizontal"):
    """
    Given the panorama image, the pose of the pinhole view,
    and the fov of the pinhole view, return the projected pinhole view.
    Input:
        img_pano: panorama image
        pose: (yaw, pitch, roll) in degrees
        fov_deg: horizontal fov of the pinhole view in degrees
        out_size: height, width of the output pinhole view
    """
    yaw, pitch, roll = pose
    R = rotation_matrix(yaw, pitch, roll)
    img_pinhole = extract_pinhole_view(img_pano, fov_deg=fov_deg, out_size=out_size, cam_rot=R, fov_type=fov_type)
    return img_pinhole

def extract_view(proposal, save_dir="../data/temp"):
    """
    Extract the view from the panorama image.
    Input:
        pano_path: path to the panorama image
        proposal: proposal dictionary
    Output:
        image_object: image object
    """
    assert "u_norm" in proposal and "v_norm" in proposal and "diag_fov" in proposal and "image_size" in proposal
    os.makedirs(save_dir, exist_ok=True)

    # check if the image exists
    image_path = os.path.join(save_dir, f"{proposal['id']}_question_view.png")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if os.path.exists(image_path):
        return image_path

    # pano_path may be given as "<source>/<file>" or "./data/<source>/<file>"; resolve it under ../data
    rel_path = os.path.normpath(proposal["pano_path"])
    if rel_path.startswith("data" + os.sep):
        rel_path = rel_path[len("data" + os.sep):]
    pano_path = os.path.join("../data", rel_path)
    if not os.path.exists(pano_path):
        raise FileNotFoundError(f"Panorama not found: {pano_path}")
    u_norm = float(proposal["u_norm"])
    v_norm = float(proposal["v_norm"])
    fov = float(proposal["diag_fov"])
    image_size = proposal["image_size"]

    img = project(
        cv2.imread(pano_path, cv2.IMREAD_COLOR),
        uv2pose((u_norm, v_norm)),
        fov, image_size,
        fov_type="diagonal"
    )
    _, buf = cv2.imencode(".png", img)
    image_object = Image.open(io.BytesIO(buf.tobytes()))

    # save the image to file
    image_object.save(image_path)
    return image_path