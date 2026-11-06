import os, json, cv2, io
from PIL import Image
import numpy as np
from pathlib import Path

from .pinhole_projector import rotation_matrix, extract_pinhole_view
from .variables import ViewCaption, PanoSummary


def default_attn_implementation():
    """Use FlashAttention-2 if installed, otherwise PyTorch SDPA."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


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

def pose2uv(yaw_deg, pitch_deg):
    """
    Convert yaw, pitch, roll in degrees to normalized uv coordinates.
    (0, 0, 0) -> (0.5, 0.5) center of the panorama.
    Input:
        yaw_deg: yaw in degrees
        pitch_deg: pitch in degrees
    Output:
        uv: normalized uv coordinates (u_norm, v_norm)
    """
    u = (yaw_deg + 180.0) / 360.0
    u = np.round(u - np.floor(u), 4)
    v = (90.0 - pitch_deg) / 180.0
    v = np.round(max(0.0, min(1.0, v)), 4)
    return (u, v)

def poses_from_patchify(cols, rows, padding_deg=0.0):
    """
    Given the number of columns and rows of the patchified panorama,
    return the poses of the pinhole views.
    Input:
        cols: number of columns of the patchified panorama
        rows: number of rows of the patchified panorama
        padding_deg: padding degree of the patchified panorama
    Output:
        poses: list of poses of the pinhole views (yaw, pitch, roll) in degrees
        fov_deg: fov of the pinhole views in degrees (horizontal, vertical, diagonal)
    """
    assert cols >= 1 and rows >= 1
    base_fov_x = round(360.0 / cols, 3)
    base_fov_y = round(180.0 / rows, 3)

    # tiny floor to avoid zero
    fov_x_deg = max(1e-3, base_fov_x + padding_deg)  
    fov_y_deg = max(1e-3, base_fov_y + padding_deg)

    # Diagonal FOV
    fx = np.radians(fov_x_deg)
    fy = np.radians(fov_y_deg)
    fov_diag_deg = round(np.degrees(2.0 * np.arctan(np.sqrt(np.tan(fx/2.0)**2 + np.tan(fy/2.0)**2))), 3)

    # yaws: longitudes (0, 360)
    # pitches: latitudes (-90, 90)
    if rows == 1:
        pitches = [0.0]
    else:
        pitches = [((j + 0.5) * 180.0 / rows) - 90.0 for j in range(rows)]

    poses = []
    # iterate horizontally first, then next row
    for pj in pitches:
        yaw_step = 360.0 / cols
        yaws = [yaw_step * i for i in range(cols)]
        for yi in yaws:
            poses.append((yi, pj, 0.0))

    return poses, (fov_x_deg, fov_y_deg, fov_diag_deg)

def out_size_from_fov(panorama_size, fov_degs, aspect_ratio= 4/3, oversample=1.0):
    """
    Given the size of the panorama, the fov of the pinhole views,
    and the aspect ratio of the pinhole views,
    return the width and height of the output pinhole views.
    Input:
        panorama_size: height, width of the panorama image
        fov_deg: fov of the pinhole views in degrees (horizontal, vertical, diagonal)
        aspect_ratio: aspect ratio of the pinhole views (default: 4/3 width/height) or "W:H" string
        oversample: oversample factor (default: 1.0)
    Output:
        out_size: height, width of the output pinhole views
        fov_x: horizontal fov in degrees (for projection)
    """
    _, width = panorama_size
    fov_x_deg, fov_y_deg, fov_diag_deg = fov_degs

    if type(aspect_ratio) == str:
        aspect_ratio = string_to_aspect(aspect_ratio)

    if fov_x_deg is None and fov_y_deg is None:
        td = np.tan(np.deg2rad(fov_diag_deg) / 2.0)
        tx = (aspect_ratio / np.sqrt(1.0 + aspect_ratio ** 2)) * td
        ty = (1.0 / np.sqrt(1.0 + aspect_ratio ** 2)) * td
        fov_x_deg = np.rad2deg(2.0 * np.arctan(tx))
        fov_y_deg = np.rad2deg(2.0 * np.arctan(ty))

    # Pixels-per-radian horizontally from the pano (uses pano width)
    # (works even if pano isn't exactly 2:1)
    ppr_h = width / (2.0 * np.pi)
    f = oversample * ppr_h

    # Convert FOVs to radians
    thx = np.deg2rad(fov_x_deg)
    thy = np.deg2rad(fov_y_deg)

    # Rectilinear image spans: size = 2 * f * tan(FOV/2)
    W_out = int(round(2.0 * f * np.tan(thx / 2.0)))
    H_out = int(round(2.0 * f * np.tan(thy / 2.0)))

    return (H_out,W_out), fov_x_deg

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

def extract_view(pano_path, proposal):
    """
    Extract the view from the panorama image.
    Input:
        pano_path: path to the panorama image
        proposal: proposal dictionary
    Output:
        image_object: image object
    """
    assert "u_norm" in proposal and "v_norm" in proposal and "diag_fov" in proposal and "image_size" in proposal
    
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
    return image_object

def extract_views(img_pano, cols, rows, save_dir=Path("./tmp"), padding_deg=0.0, oversample=1.0, resize=1.0):
    """
    Given the panorama image, the number of columns and rows of the patchified panorama,
    return the projected pinhole views.
    Input:
        img_pano: panorama image
        cols: number of columns of the patchified panorama
        rows: number of rows of the patchified panorama
        save_dir: directory to save the projected pinhole views
        padding_deg: padding degree of the patchified panorama
        oversample: oversample factor (default: 1.0)
    Output:
        views: list of projected pinhole views
        meta: metadata of the patchified panorama
    """
    panorama_size = img_pano.shape[:2]
    poses, fov_degs = poses_from_patchify(cols, rows, padding_deg=padding_deg)
    out_size, fov_x_deg = out_size_from_fov(panorama_size, fov_degs, oversample=oversample)
    out_size = (int(out_size[0] * resize), int(out_size[1] * resize))
    
    views = []
    for idx, pose in enumerate(poses):
        view_path = save_dir / f"view_{idx+1}.jpg"
        if view_path.exists():
            img = cv2.imread(str(view_path), cv2.IMREAD_COLOR)
            # print(f"View {idx+1} already exists, skipping")
        else:
            img = project(
                img_pano, pose=pose,
                fov_deg=fov_x_deg, out_size=out_size
            )
            if save_dir != Path("./tmp"):
                # print(f"Saving view {idx+1} to {view_path}")
                cv2.imwrite(str(view_path), img)
        views.append(img)

    meta = {
        "cols": cols,
        "rows": rows,
        "fov_degs": fov_degs,
        "out_size": out_size,
        "pano_size": panorama_size,
        "poses": [{
            "yaw": round(y, 2), "pitch": round(p, 2), "roll": round(r, 2), 
            "image_path": save_dir / f"view_{idx+1}.jpg",
            "left_neighbor_view":   f"view_{idx}" if idx % cols > 0 else f"view_{idx + cols}",
            "right_neighbor_view":  f"view_{idx + 2}" if idx % cols < cols - 1 else f"view_{idx - cols + 2}",
            "top_neighbor_view":    f"view_{idx - cols + 1}" if idx >= cols else "Top Edge",
            "bottom_neighbor_view": f"view_{idx + cols + 1}" if idx < cols * (rows - 1) else "BottomEdge",
            } for idx, (y, p, r) in enumerate(poses)],
    }
    
    if save_dir:
        return meta
    else:
        return views, meta

def views_block_str(captions):
    """Format the list of ViewCaption objects into a multi-line string."""
    lines = []
    aspect_ratio = captions[0].uv_meta['aspect_ratio']
    lines.append(f"Aspect ratio of the set: {aspect_to_string(aspect_ratio)}")
    for vc in captions:
        uv = vc.uv_meta
        lines.append(
            f"- view_id={vc.view_id}; u_coordinate={uv['u_norm']:.3f}, "
            f"v_coordinate={uv['v_norm']:.3f}, diag_fov={uv['fov_diag_deg']:.2f}; "
            f"left_neighbor_view={uv['left_neighbor_view']}, right_neighbor_view={uv['right_neighbor_view']}, "
            f"top_neighbor_view={uv['top_neighbor_view']}, bottom_neighbor_view={uv['bottom_neighbor_view']}; "
            f"caption={vc.caption}; objects with location in the view={vc.objects}; "
            f"spatial_facts={vc.spatial_facts}"
        )
    return "\n".join(lines)

def aspect_to_string(aspect_ratio, max_den=100):
    """
    Convert aspect ratio (width/height) into a simplified "W:H" string.
    Example: 1.7777 -> "16:9"
    Input:
        aspect_ratio: aspect ratio (width/height)
        max_den: maximum denominator (default: 100)
    Output:
        aspect_string: "W:H" string
    """
    # scale to integers
    denom = max_den
    num = round(aspect_ratio * denom)

    # reduce fraction with gcd
    g = np.gcd(num, denom)
    num //= g
    denom //= g

    return f"{num}:{denom}"

def string_to_aspect(aspect_string):
    """
    Convert "W:H" string to aspect ratio (width/height).
    Example: "16:9" -> 16/9
    Input:
        aspect_string: "W:H" string
    Output:
        aspect_ratio: aspect ratio (width/height)
    """
    num, denom = map(int, aspect_string.split(":"))
    return num / denom

def read_variable_saves(out_dir, save_name, cache_dir=None):
    """
    Read the variable saves from the output directory.
    Input:
        out_dir: output directory
        save_name: name of the variable to read
    Output:
        variable: variable object or list of variable objects
    """
    if cache_dir:
        out_dir = Path(cache_dir)
        
    if not os.path.exists(out_dir / f"{save_name}.json"):
        # print(f"{save_name} not found in {out_dir / save_name}.json")
        return None

    # print(f"Loading {save_name} from {out_dir / save_name}.json")
    variable_type = {
        "captions": ViewCaption,
        "summary": PanoSummary,
    }
    with open(out_dir / f"{save_name}.json", "r") as f:
        if save_name == "captions":
            return [variable_type[save_name](**c) for c in json.load(f)]
        elif save_name == "summary":
            return variable_type[save_name](**json.load(f))
        else:
            return json.load(f)

def save_variable(input, out_dir, save_name):
    """
    Save the variable to the output directory.
    Input:
        input: input variable to save
        out_dir: output directory
        save_name: name of the variable to save
    """
    os.makedirs(out_dir, exist_ok=True)
    if type(out_dir) == str:
        out_dir = Path(out_dir)
        
    if isinstance(input, str):
        with open(out_dir / f"{save_name}.txt", "w+") as f:
            f.write(input)
    else:
        with open(out_dir / f"{save_name}.json", "w+") as f:
            if isinstance(input, list):
                try:
                    json.dump([c.model_dump(mode="json", exclude_none=True) for c in input], f, ensure_ascii=False, indent=2)
                except:
                    json.dump(input, f, ensure_ascii=False, indent=2)
            elif isinstance(input, dict):
                json.dump(input, f, ensure_ascii=False, indent=2)
            else:
                json.dump(input.model_dump(mode="json", exclude_none=True), f, ensure_ascii=False, indent=2)

def spec_check(proposal):
    """
    Check if the proposal specification is valid.
    Input:
        proposal: proposal dict
    Output:
        valid: whether the proposal specification is valid
    """
    if type(proposal) != dict:
        return False
    inspect_list = [
        "u_norm", "v_norm", "diag_fov", 
        "aspect_ratio", "question",
        "option_a", "option_b", 
        "option_c", "option_d", "option_e",
        "option_a_reasoning", "option_b_reasoning", 
        "option_c_reasoning", "option_d_reasoning", "option_e_reasoning",
        "answer", "answer_reasoning", "confidence_score"
        ]
    for key in inspect_list:
        if key not in proposal:
            return False
        if isinstance(proposal[key], str) and len(proposal[key]) == 0:
            return False
    
    u_norm = float(proposal['u_norm'])
    v_norm = float(proposal['v_norm'])

    if u_norm < 0 or u_norm >= 1 or v_norm < 0 or v_norm >= 1:
        return False

    fov = float(proposal['diag_fov'])
    if fov < 40 or fov > 100:
        return False

    aspect_ratio = proposal['aspect_ratio']
    AR_list = ["4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "1:1"]
    if aspect_ratio not in AR_list:
        return False

    return True

def merge_proposal(origin, new):
    """
    Merge the new proposal into the origin proposal.
    Only merge the keys that are not in the new proposal.
    Input:
        origin: origin proposal
        new: new proposal
    Output:
        merged: merged proposal
    """
    merged = new.copy()
    for k, v in origin.items():
        if k not in merged:
            merged[k] = v
    
    return merged

def logging(cfg, string):
    if cfg["token_logging"]:
        os.makedirs("/".join(cfg["log_path"].split("/")[:-1]), exist_ok=True)
        with open(cfg["log_path"], "a") as f:
            f.write(string)