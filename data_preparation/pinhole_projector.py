import cv2
import numpy as np

def get_pinhole_rays(fx, fy, cx, cy, width, height):
    """
    Get the pinhole rays from the pinhole view.
    Input:
        fx, fy: focal length in x and y direction
        cx, cy: center of the pinhole view
        width, height: width and height of the pinhole view
    Output:
        rays: pinhole rays
    """
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / norms


def rays_to_equirectangular_coords(rays, eq_width, eq_height):
    """
    Convert the pinhole rays to the equirectangular coordinates.
    Input:
        rays: pinhole rays
        eq_width, eq_height: width and height of the equirectangular image
    Output:
        u, v: equirectangular coordinates
    """
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(-y, np.sqrt(x**2 + z**2))

    u = (lon / (2 * np.pi) + 0.5) * eq_width
    v = (lat / np.pi + 0.5) * eq_height

    return u.astype(np.float32), v.astype(np.float32)


def extract_pinhole_view(equirect_img, fov_deg, out_size, cam_rot=np.eye(3), flip_y=True, fov_type="horizontal"):
    """
    Extract the pinhole view from the equirectangular image.
    Input:
        equirect_img: equirectangular image
        fov_deg: horizontal/diagonal field of view in degrees
        out_size: height, width of the pinhole view
        cam_rot: rotation matrix of the camera
        flip_y: whether to flip the pinhole view
    Output:
        pinhole_img: pinhole view
    """
    eq_height, eq_width = equirect_img.shape[:2]
    height, width  = out_size

    if fov_type == "horizontal":
        fov_x_rad = np.deg2rad(fov_deg)
    elif fov_type == "diagonal":
        aspect_ratio = out_size[1] / out_size[0]
        fov_d_rad = np.deg2rad(fov_deg)
        fov_x_rad = 2 * np.arctan(
            np.tan(fov_d_rad / 2) * aspect_ratio / np.sqrt(aspect_ratio**2 + 1)
        )
    else:
        raise ValueError(f"Invalid fov_type: {fov_type}")

    fx = fy = 0.5 * width / np.tan(fov_x_rad / 2)
    cx, cy = width / 2, height / 2

    rays = get_pinhole_rays(fx, fy, cx, cy, width, height)
    rays = rays @ cam_rot.T

    u_map, v_map = rays_to_equirectangular_coords(rays, eq_width, eq_height)

    # clip to range
    u_map = np.clip(u_map, 0, eq_width - 1.001) 
    v_map = np.clip(v_map, 0, eq_height - 1.001)  

    pinhole_img = cv2.remap(equirect_img, u_map, v_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,255])

    if flip_y:
        pinhole_img = cv2.flip(pinhole_img, 0)

    return pinhole_img


def rotation_matrix(yaw, pitch, roll):
    """
    Get the rotation matrix of the camera.
    Input:
        yaw, pitch, roll: yaw, pitch, roll in degrees
    Output:
        R: rotation matrix
    """
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx