import cv2.aruco as aruco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2, shutil, math, re, random, functools, time, os, errno
from sklearn.metrics import mean_absolute_error
from glob import glob
from shapely.geometry import Polygon
from PIL import Image

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def create_empty_df_dict(type='img'):
    if type=='img':
        return {"original_fp": [], 'corners': [], "pix_per_len": [], 'units': []}
    elif type == 'crop':
        return {"original_fp": [], 'file': [], "pix_per_len": [], 'units': [], 'marker_corners':[]}
    elif type == 'crop_zoom':
        return {
            'original_fp': [],
            's_i': [],
            'file': [],
            'pix_per_len': [],
            'units': [],
            'crop_corners': [],
            'n': [],
            'marker_corners': []
        }
    else:
        raise ValueError(f"Type {type} is not implemented.")


def get_intermittent_file_path(original_fp, data_folder):
    f_list = original_fp.split('/')
    idx = f_list.index(data_folder)
    return f_list[idx - len(f_list) + 1:-1]


def extract_directory(files, data_folder, out_folder):
    """
    Given a list of all the images, extracts all sub-directories in order
    :param files: list of strs, containing all the image paths
    :param data_folder: str, folder name containing the dataset
    :param out_folder: str, folder name that will contain the newly created montage training set
    :return: all_dirs: set of str, containing unique sequence of sub-directories that need to be created.
    """
    all_dirs = set()
    for f in files:
        f_list = get_intermittent_file_path(f, data_folder)
        all_dirs.add(os.path.join(out_folder, 'cropped', '/'.join(f_list)))
        all_dirs.add(os.path.join(out_folder, 'detected_img', '/'.join(f_list)))
    return all_dirs


def make_dir(pths):
    """
    Creates all the directories listed in pths.
    Warning: if directories and subdirectories are both given, directories must be made first or else,
    some subdirectories will be deleted when the directory is made. This is because when each folder is made,
    the algorithm checks if it exists, and if it does,
    it proceeds to delete all its contents before remaking the folder.
    :param pths: List of str, or str.
    :return: None
    """
    if isinstance(pths, list) or isinstance(pths, set):
        for pth in pths:
            if os.path.exists(pth):
                shutil.rmtree(pth)
            os.makedirs(pth)
    else:
        if os.path.exists(pths):
            shutil.rmtree(pths)
        os.makedirs(pths)


def detect_marker(image, img, aruco_dict, parameters, skip_manual_marker_selection):
    """
    Using aruco algorithm, automatically find the marker.
    Assumption: There is always a marker of interest in the image
    :param image: numpy array of the image
    :param img: image path
    :param aruco_dict: aruco dictionary
    :param parameters: aruco dictionary parameters
    :return: corners of the aruco marker
    """
    dummy_corners = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initial try to find marker
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) != 0:
        found_marker = True
    else:
        found_marker = False

    # Try rotating image to find marker
    if not found_marker:
        for i in range(0, 360, 90):
            # print(i)
            gray_rotated, dummy_corners = rotate_image(gray, dummy_corners, i)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_rotated, aruco_dict, parameters=parameters)
            if len(corners) != 0:
                print("Found markers!")
                gray, corners = rotate_image(gray_rotated, corners, -i)
                found_marker = True
                break
    # Check that there is only one marker detected
    if len(corners) == 0:
        print("Found no markers in file: {}".format(img))
        # Manually find marker
        if skip_manual_marker_selection:
            return None, False
        else:
            # Manually find marker
            corners = manually_select_marker(image)
        # corners = False
    elif len(corners) > 1:
        print("Found more than 1 markers in file: {}".format(img))
        # Manually find marker
        if skip_manual_marker_selection:
            return None, False
        else:
            # Manually find marker
            corners = manually_select_marker(image)
        # corners = False
    else:
        corners = corners[0][0]
    return corners


def cmanually_select_marker(image):
    """
    https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    Get user to automatically select 4 points
    :param image: numpy image
    :return: 4 points of the marker
    """
    refPt = []

    def click_and_crop(event, x, y, flags, param):
        '''
        Mouse event connected to window
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        '''
        # if the left mouse button was clicked, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            # draw a rectangle around the region of interest
            cv2.circle(clone, tuple(refPt[-1]), 25, (0, 255, 0), -1)
            cv2.imshow("image", clone)

    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)
    cv2.resizeWindow('image', 2400, 1200)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF

        # if the 's' key is pressed, show current points
        if key == ord("s"):
            print(refPt)
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            clone = image.copy()
            refPt = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            # if there are four reference points, then crop the region of interest from the image and display it
            if len(refPt) == 4:
                break
            print("You do not have exactly 4 points.")
            print(refPt)
    # close all open windows
    cv2.destroyAllWindows()
    return np.array(refPt)


def rotate_image(image, restricted_area, angle):
    """
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Obtain the rotated coordinates of the restricted corners
    restricted_area = np.array([
        np.dot(affine_mat, np.append(restricted_area[0, :], 1)).A[0],
        np.dot(affine_mat, np.append(restricted_area[1, :], 1)).A[0],
        np.dot(affine_mat, np.append(restricted_area[2, :], 1)).A[0],
        np.dot(affine_mat, np.append(restricted_area[3, :], 1)).A[0]
    ])

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result, restricted_area


def largest_rotated_rect(w, h, angle):
    """
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, rotated_restricted_area, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """
    cropped_and_rotated_restricted_area = rotated_restricted_area.copy()
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    # Shift the marker corners by the cropping amount
    cropped_and_rotated_restricted_area[:, 0] = cropped_and_rotated_restricted_area[:, 0] - y1
    cropped_and_rotated_restricted_area[:, 1] = cropped_and_rotated_restricted_area[:, 1] - x1

    return image[y1:y2, x1:x2], cropped_and_rotated_restricted_area


def get_crop(img, cnt, width=299, height=299):
    # TODO: See img file sent by Juan, 25/26-02-2021 STEP 4
    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    if abs(int(rect[1][0])-width) > 10 or abs(int(rect[1][1])-height) > 10:
        raise ValueError("Your crop image size and desired image size are very different.")

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    return cv2.warpPerspective(img, H, (width, height))


def RotM(alpha):
    """ Rotation Matrix for angle ``alpha`` """
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa],
                     [sa,  ca]])


def getRandomSquareVertices(center, point_0, phi):
    '''
    center: tuple
    point_0: tuple from origin
    phi: angle
    '''
    vv = [[np.asarray(center) + functools.reduce(np.dot, [RotM(phi), RotM(np.pi / 2 * c), point_0])] for c in range(4)]
    return np.array(vv).astype(np.float32)


def get_random_crops(image, crop_height, crop_width, restricted_area, n_crops=4, max_angle=360, seed=None, width=299, height=299, n_channels=1, m_patches=10, margin=75):
    """
    Randomly rotate and retrieve crops from image to generate montages
    :param image: numpy array, contains the pixel value of images
    :param crop_height: int, crop height
    :param crop_width: int, crop width
    :param restricted_area: numpy array size 4-by-2, containing coordinates of the marker
    :param n_crops: int, Number of crops in the montage
    :param m_patches: int, Number of montages to generate
    :param max_angle: int, Angle by which the image is rotated
    :param seed: random number generator seed
    :return:
    """
    # Initialize parameters
    np.random.seed(seed=seed)
    image_height, image_width = image.shape[0:2]
    crops = []
    for i in range(m_patches):
        crops.append(get_crops(restricted_area, n_crops, image, crop_width, crop_height, n_channels, margin))
        # crops.extend(get_crops(restricted_area, n_crops, image, crop_width, crop_height, n_channels, margin))
    return crops


def montage_crops(n_crops, crop_width, crop_height, n_channels, crops):
    rows = int(np.ceil(n_crops ** 0.5))
    tmp_image = np.zeros([crop_width * rows, crop_height * rows, n_channels], dtype='uint8')
    for i in range(rows):
        for j in range(rows):
            if n_channels == 1:
                if len(crops) > i * rows + j:
                    tmp_image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width, 0] = crops[
                        i * rows + j]
            else:
                if len(crops) > i * rows + j:
                    tmp_image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width, :] = crops[
                        i * rows + j]
    return tmp_image

def attempt_find_valid_crops(restricted_area, margin, m_patches, n, original_fp, attempts):
    crop_corners = []
    img = Image.open(original_fp)  # Loads image without actually loading image.
    width, height = img.size

    # Create polygon to check if randomly generated points are inside polygon
    marker_polygon = Polygon(restricted_area)
    # Added margin to avoid whitespace on marker
    marker_polygon = marker_polygon.buffer(margin)
    is_there_valid_regions = False
    for m in range(m_patches):
        # Generate crops
        num_attempts = 1
        while num_attempts <= attempts or is_there_valid_regions:
            forbid_border = math.ceil((n ** 2 + n ** 2) ** (1 / 2)) / 2
            max_x = width - forbid_border
            max_y = height - forbid_border  # TODO: check if height and width need to be flipped
            if max_x < forbid_border or max_y < forbid_border:
                num_attempts += 1
                continue
            x = np.random.randint(forbid_border, max_x)
            y = np.random.randint(forbid_border, max_y)
            rotation_angle = random.random() * np.pi

            crop_vertices = getRandomSquareVertices((x, y), (n / 2, n / 2), rotation_angle)
            tmp_crop_corners = [(crop_vertices[0][0][0], crop_vertices[0][0][1]),
                 (crop_vertices[1][0][0], crop_vertices[1][0][1]),
                 (crop_vertices[2][0][0], crop_vertices[2][0][1]),
                 (crop_vertices[3][0][0], crop_vertices[3][0][1])]
            crop_polygon = Polygon(tmp_crop_corners)

            found_crop = not marker_polygon.intersects(crop_polygon)
            if found_crop:
                crop_corners.append(tmp_crop_corners)
                is_there_valid_regions = True
                break
            num_attempts += 1
        # Check if we found a crop
        if not is_there_valid_regions:
            return False
    return crop_corners



def get_crops(restricted_area, n_crops, image, crop_width, crop_height, n_channels, margin):
    crops = []
    # Create polygon to check if randomly generated points are inside polygon
    marker_polygon = Polygon(restricted_area)
    # Added margin to avoid whitespace on marker
    marker_polygon = marker_polygon.buffer(margin)
    for n in range(n_crops):
        # Generate crops
        found_crop = False
        # attempt = 0
        while not found_crop:
            forbid_border = math.ceil((crop_width**2+crop_height**2)**(1/2))/2
            max_x = image.shape[1] - forbid_border
            max_y = image.shape[0] - forbid_border
            x = np.random.randint(forbid_border, max_x)
            y = np.random.randint(forbid_border, max_y)
            rotation_angle = random.random()*np.pi

            crop_vertices = getRandomSquareVertices((x,y), (crop_width/2, crop_height/2), rotation_angle)
            crop_polygon = Polygon(
                [(crop_vertices[0][0][0], crop_vertices[0][0][1]),
                 (crop_vertices[1][0][0], crop_vertices[1][0][1]),
                 (crop_vertices[2][0][0], crop_vertices[2][0][1]),
                 (crop_vertices[3][0][0], crop_vertices[3][0][1])])

            found_crop = not marker_polygon.intersects(crop_polygon)
            if found_crop:
                if n_channels == 1:
                    crops.append(get_crop(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), crop_vertices, crop_width, crop_height))
                    # for cv, wid in zip(crop_vertices,[1000, 800, 500]):
                    #     crops.append(get_crop(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv, wid, wid))
                else:
                    crops.append(get_crop(image, crop_vertices, crop_width, crop_height))
                break
            # attempt += 1
    return montage_crops(n_crops, crop_width, crop_height, n_channels, crops)


def get_pix_per_len(corners, marker_len):
    dist = []
    for c in corners:
        tmp_dist = []
        for c_2 in corners:
            tmp_dist.append(((c_2[0] - c[0]) ** 2 + (c_2[1] - c[1]) ** 2) ** 0.5)
        tmp_dist = sorted(tmp_dist)[1:-1]
        dist.extend(tmp_dist)
    return np.average(dist) / marker_len


def read_img_or_crop_df_and_return_img_df(img_or_crop_df_pth):
    df = pd.read_csv(img_or_crop_df_pth)
    cols = df.columns.tolist()
    img_df_cols = ["original_fp", "corners", "pix_per_len", "units"]
    # This is a crop dataset. Convert back to img df
    if 'file' in cols:
        df['pix_per_len'] = df['s_i']
        df['corners'] = df['marker_corners']
        df = df[img_df_cols]
        df.drop_duplicates(subset=['original_fp'], inplace=True)
    return df

def create_df_img(files, img_or_crop_df_pth, skip_manual_marker_selection, marker_len, out_folder, units):
    # If path to a dataframe containing marker information is provided, read the csv.
    if img_or_crop_df_pth is not None:
        df_img = read_img_or_crop_df_and_return_img_df(img_or_crop_df_pth)
        if isinstance(df_img['corners'][0], np.ndarray):
            pass
        else:
            tmp_corners = [re.sub(' +', ' ', corner.replace('\n', ' ').replace('[', ' ').replace(']', ' ')).strip().split(' ') for corner in df_img['corners'].tolist()]
            tmp_corners = [np.array([(float(corners[0]), float(corners[1])),
                                (float(corners[2]), float(corners[3])),
                                (float(corners[4]), float(corners[5])),
                                (float(corners[6]), float(corners[7]))]) for corners  in tmp_corners]
            df_img['corners'] = tmp_corners
    # Otherwise, loop through the images to detect the markers to form the dataframe
    else:
        # Container to hold the dataframe
        df_img = create_empty_df_dict('img')
        # Initialize aruco variables
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5

        # Detect marker and compute pix_per_len for each image
        print("Running detection algorithm...\n\n")
        for img in tqdm(files):
            # Read image
            image = cv2.imread(img)
            # Detect marker
            corners, found_corners = detect_marker(image, img, aruco_dict, parameters, skip_manual_marker_selection)
            if not found_corners:
                continue
            # If the marker detection algorithm didn't work, manually select the marker
            if isinstance(corners, bool):
                if skip_manual_marker_selection:
                    continue
                else:
                    corners = manually_select_marker(image)
            else:
                cont = False
                for c in corners:
                    if c[0] > 5000 or c[1] > 5000 or c[0] < 0 or c[1] < 0:
                        cont = True
                        break
                if cont:
                    if skip_manual_marker_selection:
                        continue
                    else:
                        corners = manually_select_marker(image)

            # Get pix_per_len
            pix_per_len = get_pix_per_len(corners, marker_len)
            # Check if marker was detected wierd. If true, then reselect marker corners
            if pix_per_len < 10 or pix_per_len > 500:
                if img_or_crop_df_pth is not None:
                    print(f"Skipped {img}, pix_per_len = {pix_per_len}")
                    continue
                else:
                    if skip_manual_marker_selection:
                        continue
                    else:
                        corners = manually_select_marker(image)
                    pix_per_len = get_pix_per_len(corners, marker_len)

            # Save marker information
            df_img['original_fp'].append(img)
            df_img['corners'].append(corners)
            df_img['pix_per_len'].append(pix_per_len)
            df_img['units'].append(units)

        # Write image dataframe
        df_img = pd.DataFrame(df_img)
        df_img.to_csv(f'{out_folder}/img_dataset.csv', index=False)
    return df_img

def extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution, skip_manual_marker_selection):
    # Initialize container for holding patch-wise information
    df_crop = create_empty_df_dict('crop')

    df_img = create_df_img(files, img_df, skip_manual_marker_selection, marker_len, out_folder, units)

    # If equalize distribution is on, then calculate historgram of image scales
    if equalize_distribution:
        # Calculate histogram
        counts, bins = np.histogram(df_img['pix_per_len'],
                                    range=(df_img['pix_per_len'].min() - 1, df_img['pix_per_len'].max() + 1))
        # Calculate m_image multiplication factor
        multiplier = np.floor(max(counts) / counts)
        plt.bar(bins[:-1], multiplier)
        plt.show()
        print(counts)
        print(multiplier)

    # Extract m patches for each image
    print(f"Extracting patches from images...\n\n")
    for idx, row in tqdm(df_img.iterrows(), total=len(df_img)):
        img = row['original_fp']
        pix_per_len = row['pix_per_len']
        # Read image
        image = cv2.imread(img)

        # Create flexible subdirectory path
        f_list = get_intermittent_file_path(img, data_folder)

        # Get marker corners
        corners = row['corners']

        # Get Crops
        if equalize_distribution:
            multiplier_idx = -1
            for j in range(len(multiplier)):
                if pix_per_len >= bins[j] and pix_per_len < bins[j + 1]:
                    multiplier_idx = j
                    break
            if multiplier_idx == -1:
                raise ValueError("WRONG")
            new_crops = get_random_crops(image, crop_height, crop_width, corners, n_crops, m_patches=int(m_patches * multiplier[multiplier_idx]))
        else:
            new_crops = get_random_crops(image, crop_height, crop_width, corners, n_crops, m_patches=m_patches)

        # Save crop and crop information
        for c in range(len(new_crops)):
            # Crop
            new_cropped_fp = os.path.join(out_folder, 'cropped', '/'.join(f_list),
                                          f"{img.split('.')[-2].split('/')[-1]}_crop_{c + len(new_crops)}.JPG")
            cv2.imwrite(new_cropped_fp, new_crops[c])
            # Crop information
            df_crop['original_fp'].append(img)
            df_crop['file'].append(new_cropped_fp)
            df_crop['pix_per_len'].append(pix_per_len)
            df_crop['units'].append(units)
            df_crop['marker_corners'].append(corners)

        # Print images with recognized markers
        print_image = image.copy()
        for corner in corners:
            cv2.polylines(print_image, [corner.reshape((-1, 1, 2)).astype('int32')], True, (0, 0, 255), 20)
        cv2.imwrite(f"{out_folder}/detected_img/{'/'.join(f_list)}/{img.split('/')[-1]}", print_image)

    if img_df is not None:
        return df_crop
    else:
        return df_crop, df_img


def create_n_by_n_markers(crop_width=299,
                          crop_height=299,
                          n_crops=25,
                          m_patches=10,
                          marker_len=10.0,  # 10
                          units='cm',
                          overall_folder='datasets/9_all_data_compiled/',
                          data_folder='1_data',
                          out_folder='datasets/9_all_data_compiled/5by5/',
                          img_df=None,
                          equalize_distribution=False,
                          skip_manual_marker_selection=True):
    """
    From a set of collected images categorized into scenes (e.g., 1 folder contains images pertaining to 1 unique scene), create a patch-image scale dataset by:
        Option 1) Running a marker detection algorithm to detect markers, compute the pixel per length metric, and extract m patches per image.
        Option 2) Reading a dataframe containing marker corner coordinates and scales for each image to extract m patches per image.

        Optionally, the user can turn on "equalize_distribution" to extract different number of patches per image to equalize the histogram of patch scales.
        This would allow loss to be equally distributed across all values of scales.

    If an image of with a marker pops up on screen, that means the algorithm was not able to detect the marker.
    Proceed to click the four corners of the marker and then when you're done, press "c".
    If you mess up, press "r" to reset.

    TODO: equalize frequency doesn't work properly. Feel free to try and fix it :)

    :param crop_width (int): Width of crop to extract from the images
    :param crop_height (int): Height of crop to extract from the images
    :param n_crops (int): Number of crops to include in a patch. Use n^2 (1, 4, 9, 25, etc.)
    :param m_patches (int): Number of patches to extract per image
    :param marker_len (float): The physical length of the marker for a given dataset.
    :param units (str): The units of the marker_len (cm? mm?)
    :param overall_folder (str): Path to the folder containing all data, the raw data, detected images, raw data etc.
    :param data_folder (str): Name of the folder in the overall_folder that contains the raw data
    :param out_folder (str): Path to the folder that will contain the crop dataset
    :param img_df (str): Path to the csv that contains marker corner coordinates and scales for each image. If supplied, the algorithm will run option 2, if not, it will run option 1.
    :param equalize_distribution (boolean): Set true to equalize the distribution of patch scales.
    """
    # Retrieve all images in data folder
    files_glob = os.path.join(overall_folder, data_folder, "**/*.[jJ][pP][gG]") # extracts files which end with jpg (not case sensitive)
    files = glob(files_glob, recursive=True)    # returns the list of files
    # Extract directories from image paths and make directories
    all_dirs = extract_directory(files, data_folder, out_folder)
    make_dir(all_dirs)
    if os.path.exists(os.path.join(out_folder, 'code')):
        shutil.rmtree(os.path.join(out_folder, 'code'))
    copytree('../code', os.path.join(out_folder, 'code'))
    # Extract crops to form the patch dataset
    if img_df is None:
        df_crop, df_img = extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution, skip_manual_marker_selection)
    else:
        df_crop = extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution,  skip_manual_marker_selection)

    # Save crop dataframes
    df_crop = pd.DataFrame(df_crop)
    df_crop.to_csv(f'{out_folder}/crop_dataset.csv', index=False)

def create_zoom_dataset(crop_length=550,
                        m_patches=10,
                        marker_len=9.4,  # 10
                        units='cm',
                        mode='training_and_validation',
                        overall_folder='datasets/9_all_data_compiled/',
                        data_folder='1_data',
                        out_folder='datasets/9_all_data_compiled/5by5/',
                        img_or_crop_df_pth=None,
                        skip_manual_marker_selection=True,
                        n_bins=10,
                        margin=0,  # in units of pixels
                        patches_per_bin=300,
                        attempts=20,
                        st_type='smooth',  # "smooth" for randomly selecting the st between the bin range. "bin" for using st=bin value (constant value),
                        img_sampling_type='normal',  # 'normal' to use a "normal"-like distribution to sample images close to st. 'uniform' to randomly select any images (uniform distribution)
                        minimum_n=550,
                        testing_step_n=10,  # increment of pixels from crop_length
                        testing_s_steps=10,  # no of times it increments
                        ):
    """

    Args:
        crop_length: The length of the base crop length
        m_patches: Number of patches to extract for each image at scale st
        marker_len: Physical length of the marker
        units: Physical length units
        mode: one of 'training_and_validation' or 'testing_implementation'. If training_and_validation, generate the regular zoom dataset to be used to train and validate the model.
              if 'testing_implementation', the dataset will extract m_patches of different patch sizes for each image, which will be used to aggregate scales.
        overall_folder: Path to the folder containing raw images, crop datasets, etc.
        data_folder: Name of the folder in the 'overall' folder containing the raw image dataset
        out_folder: Path to the output crop folder
        img_or_crop_df_pth: (Optional) Path to the dataframe containing images, the marker corners, and image scale (s_i)
        skip_manual_marker_selection: True to skip images that fail marker detection. False to manually select the marker in the image.
        n_bins: number of bins for the image scale
        margin: In units of pixels. sets a margin around the marker to avoid during crop extraction
        patches_per_bin: Number of patches to extract for each bin
        attempts: NUmber of attempts to find a valid crop size before giving up and trying another image
        st_type: 'smooth' or 'bin'.
            smooth: randomly choose a st value in the bin range each time when extract crops from a image
            bin: set st=bin value.
        img_sampling_type: 'normal' or 'uniform'
            'normal' for weighting the image sampling algorithm for selecting images with mean=st and std.dev.=std.dev.(dataset)/2
            'uniform' for uniformly weighted sampling of images
        minimum_n: the minimum size to extract patches. (Typically set to the model input size)
        testing_step_n: Used for mode 'testing', specifies the amount of change (in pixels) in n for each iteration
            (i.e. given crop_length 550 and testing_step_n of 10 and testing_s_steps of, we extract m patches of size 520, 530, 540, 550, 560, 570, 580)
        testing_s_steps: Used for mode 'testing', specifies number of times to step n.
    """
    # Retrieve all images in data folder
    files_glob = os.path.join(overall_folder, data_folder,
                              "**/*.[jJ][pP][gG]")  # extracts files which end with jpg (not case sensitive)
    files = glob(files_glob, recursive=True)  # returns the list of files
    # Extract directories from image paths and make directories
    all_dirs = extract_directory(files, data_folder, out_folder)
    make_dir(all_dirs)
    if os.path.exists(os.path.join(out_folder, 'code')):
        shutil.rmtree(os.path.join(out_folder, 'code'))
    copytree('../code', os.path.join(out_folder, 'code'))

    # Read or create df_img dataframe
    df_img = create_df_img(files, img_or_crop_df_pth, skip_manual_marker_selection, marker_len, out_folder, units)

    # Find min and max scale
    s_min = df_img['pix_per_len'].min()
    s_max = df_img['pix_per_len'].max()
    # # Specify the number of bins b/w Smin and Smax
    # n_bins = 10
    # Find range of each bin
    interval = (s_max - s_min) / n_bins
    # bins = np.arange(s_min, s_max + 1, interval)
    s_t = [s_min+(i+1)*interval for i in range(n_bins)]


    # Build crop dataframe
    print(f"Building {mode} crop dataframe...")
    crop_df = []
    # If mode is training create the zoom training dataset, which sets up bins and extracts m crops from a randomly selected image for each bin for all the bins
    if mode == 'training_and_validation':
        for st_bin in tqdm(s_t):
            crop_df_for_st = create_empty_df_dict('crop_zoom')

            while patches_per_bin >= len(crop_df_for_st['original_fp']):
                if st_type == 'bin':
                    st = st_bin
                elif st_type == 'smooth':
                    st = random.uniform(st_bin, st_bin-interval)  # Get a random float number in the st bin range
                else:
                    raise ValueError(f"{st_type} not implemented. Please choose between 'bin' and 'smooth'.")
                """"""""""""""""""
                if img_sampling_type == 'uniform':
                    row = df_img.sample()
                elif img_sampling_type == 'normal':
                    # define gaussian dist with mean s_t and std dev as original std dev/2
                    # for each row, plug each row in the gaussian equation (returns probability for that rows)
                    # this way, get the probability distribution using s_i, this will be the weights
                    mu=st
                    sigma = df_img['pix_per_len'].std()/2
                    prob_func = lambda x: 1/(sigma*(2*np.pi)**0.5)*(np.exp(-0.5*((x-mu)/sigma)**2))

                    # for each row, plug in s_i to get the corresponding weight
                    weights = prob_func(df_img['pix_per_len'])
                    row = df_img.sample(weights=weights)  # TODO: Check if this is working properly
                else:
                    raise ValueError(f"{img_sampling_type} not implemented. Please choose between 'random' and 'weighted'.")
                """"""""""""""""""

                s_i = row['pix_per_len'].tolist()[0]
                marker_corners = row['corners'].tolist()[0]
                original_fp = row['original_fp'].tolist()[0]
                n = int(s_i * crop_length / st)
                # Recalculate the st based on n.
                st = s_i * crop_length / n
                # If the patch size (n) is less than the minimum_n, reselect and retry algorithm
                if n < minimum_n:
                    continue

                f_list = get_intermittent_file_path(original_fp, data_folder)
                crops = attempt_find_valid_crops(marker_corners, margin, m_patches, n, original_fp, attempts)
                if crops: # If we found crops
                    for idx, crop in enumerate(crops):
                        crop_df_for_st['original_fp'].append(original_fp)
                        crop_df_for_st['s_i'].append(s_i)
                        crop_df_for_st['file'].append(os.path.join(out_folder, 'cropped', '/'.join(f_list),
                                              f"st_{np.round(st, 1)}_si_{np.round(s_i, 1)}_n_{n}{original_fp.split('.')[-2].split('/')[-1]}_crop_{idx}.JPG")) # TODO: Check for correct function
                        crop_df_for_st['pix_per_len'].append(st)
                        crop_df_for_st['units'].append(units)
                        crop_df_for_st['crop_corners'].append(crop)
                        crop_df_for_st['marker_corners'].append(marker_corners)
                        crop_df_for_st['n'].append(n)
            crop_df.append(pd.DataFrame(crop_df_for_st))
    # Testing dataset
    elif mode == 'testing_implementation':
        n_arr = np.arange(crop_length-testing_step_n*testing_s_steps, crop_length+testing_step_n*testing_s_steps, testing_s_steps)
        for idx, row in tqdm(df_img.iterrows(), total=len(df_img)):
            crop_df_for_image = create_empty_df_dict('crop_zoom')
            original_fp = row['original_fp']
            s_i = row['pix_per_len']
            marker_corners = row['corners']
            f_list = get_intermittent_file_path(original_fp, data_folder)
            for n in n_arr:
                st = s_i * crop_length / n
                crops = attempt_find_valid_crops(marker_corners, margin, m_patches, n, original_fp, attempts)
                if crops:  # If we found crops
                    for idx, crop in enumerate(crops):
                        crop_df_for_image['original_fp'].append(original_fp)
                        crop_df_for_image['s_i'].append(s_i)
                        crop_df_for_image['file'].append(os.path.join(out_folder, 'cropped', '/'.join(f_list),
                                                                   f"st_{np.round(st, 1)}_si_{np.round(s_i, 1)}_{original_fp.split('.')[-2].split('/')[-1]}_crop_{idx}.JPG"))  # TODO: Check for correct function
                        crop_df_for_image['pix_per_len'].append(st)
                        crop_df_for_image['units'].append(units)
                        crop_df_for_image['crop_corners'].append(crop)
                        crop_df_for_image['marker_corners'].append(marker_corners)
                        crop_df_for_image['n'].append(n)
                else:
                    print(f"WARNING: patches were not able to be extracted for n: {n} and file {original_fp}.\n"
                          f"This can result in a incomplete assessment when testing the zoom-based scale estimation approach.")
            crop_df.append(pd.DataFrame(crop_df_for_image))
    else:
        raise ValueError(f"Mode {mode} is not implmented.")
    crop_df = pd.concat(crop_df)
    crop_df.to_csv(f'{out_folder}/crop_dataset_base_crop_length_{crop_length}.csv', index=False)

    print("Extracting Patches...")
    for idx, row in tqdm(crop_df.iterrows(), total=len(crop_df)):
        crop_corners = row['crop_corners']
        image = cv2.imread(row['original_fp'])
        new_crop = get_crop(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.expand_dims(np.array(crop_corners), axis=1), row['n'], row['n'])
        cv2.imwrite(row['file'], new_crop)

    # Ideas for later:
    # Extract m patches from each image for as any bins as possible and then equalize the histogram using the random image selection algorithm
    # Extract patches from s_i closest to s_t first, and then extract using randomly selected images of which can have significantly different s_i (this is because s_i closest to s_t is the "best" representative samples for that bin)

if __name__ == '__main__':

    # training image dataset


    create_zoom_dataset(overall_folder='../datasets/homography_test/',  # folder that contains raw data
                        out_folder='../datasets/homography_test/case_2_3_train',
                        # contains cropped dataset
                        img_or_crop_df_pth=None,
                        img_sampling_type='uniform',
                        skip_manual_marker_selection=False,
                        n_bins=1,
                        m_patches=1, patches_per_bin=10, crop_length=850,
                        )
    # create_n_by_n_markers(n_crops=2,
    #                       m_patches=5,
    #                       overall_folder='../datasets/homography_test/',
    #                       out_folder='../datasets/homography_test/trial',
    #                       crop_height=100,
    #                       crop_width=100,
    #                       marker_len=9.4)

    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/case_2_3_train',
    #                     # contains cropped dataset
    #                     img_or_crop_df_pth='../datasets/PED_V2/2_processed/train_img_dataset.csv',
    #                     img_sampling_type='uniform',
    #                     m_patches=1, patches_per_bin=900, crop_length=850,
    #                     )
    #
    # # test image dataset
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/case_2_3_test',  # contains cropped dataset
    #                     img_or_crop_df_pth='../datasets/PED_V2/2_processed/test_img_dataset.csv',
    #                     img_sampling_type='uniform',
    #                     m_patches=1, patches_per_bin=300, crop_length=850,
    #                     )

    # To run the detection and output the dataframe containing corners of detected markers, run this code:
    # Test zoom aggreagation implementation
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_zoom_agg_implementation',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', mode='testing_implementation'
    #                     )
    # Testing case 1
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_1_train_set',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv'
    #                     )
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_1_test_set',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv',
    #                     patches_per_bin=100
    #                     )
    # # Testing case 2
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_2_train_set',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', img_sampling_type='uniform',
    #                     )
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_2_test_set',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', img_sampling_type='uniform',
    #                     patches_per_bin=100
    #                     )
    # Testing case 3
    # Take the best case from 1 or 2 and change patches per bin to 600
    # create_zoom_dataset(overall_folder='../datasets/PED_V2/',  # folder that contains raw data
    #                     out_folder='../datasets/PED_V2/testing_minimum550_m_patch_1_ppb_900_crop_length_850',  # contains cropped dataset
    #                     img_df='../datasets/PED_V2/2_processed/img_dataset.csv', img_sampling_type='uniform',
    #                     m_patches=1, patches_per_bin=900, crop_length=850
    #                     )
    # Testing case 4







    # create_zoom_dataset(overall_folder='../datasets/BW/',  # folder that contains raw data
    #                     out_folder='../datasets/BW/test_zoom_generator',  # contains cropped dataset
    #                     img_df='../datasets/BW/2_processed/train_img_dataset.csv'
    #                     )

    # # PED_V2 BRIDGE 100
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_test_100_final', img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', crop_height=100, crop_width=100, marker_len=9.4)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_train_100_final', img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', crop_height=100, crop_width=100, marker_len=9.4)
    # # PED_V2 BRIDGE 350
    # create_n_by_n_markers_from_df(n_crops=1, m_p atches=50, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_test_350_final', img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', crop_height=350, crop_width=350, marker_len=9.4)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_train_350_final', img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', crop_height=350, crop_width=350, marker_len=9.4)


    # # PED_V2 BRIDGE 850
    # create_n_by_n_markers(n_crops=1, m_patches=50,  # 50 patches per image
    #                       overall_folder='../datasets/PED_V2/', # folder that contains raw data
    #                       out_folder='../datasets/PED_V2/testing',  # contains cropped dataset
    #                       img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv',
    #                       crop_height=850, crop_width=850, marker_len=9.4)


    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_train_850_final', img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', crop_height=850, crop_width=850, marker_len=9.4)
    # # ASH
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/ASH/', out_folder='../datasets/ASH/3_test_final', img_df='../datasets/ASH/2_processed/test_img_dataset.csv', crop_height=850, crop_width=850)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/ASH/', out_folder='../datasets/ASH/3_train_final', img_df='../datasets/ASH/2_processed/train_img_dataset.csv', crop_height=850, crop_width=850)

    # # ASH_V2
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/ASH_V2/', out_folder='../datasets/ASH_V2/3_test_final', img_df='../datasets/ASH_V2/2_processed/test_img_dataset.csv', crop_height=850, crop_width=850)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/ASH_V2/', out_folder='../datasets/ASH_V2/3_train_final', img_df='../datasets/ASH_V2/2_processed/train_img_dataset.csv', crop_height=850, crop_width=850)

    # # DIFF
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/DIFF/', out_folder='../datasets/DIFF/3_test_final', img_df='../datasets/DIFF/2_processed/img_dataset.csv', crop_height=850, crop_width=850, marker_len=9.2)
    # # ZOOM
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/ZOOM/', out_folder='../datasets/ZOOM/3_test_final', img_df='../datasets/ZOOM/2_processed/img_dataset.csv', crop_height=850, crop_width=850)


    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=20, overall_folder='../datasets/13_ped_bridge_new_dataset/', out_folder='../datasets/13_ped_bridge_new_dataset/paper_100/', img_df='../datasets/13_ped_bridge_new_dataset/1_processed/train_high_light_img_dataset.csv', crop_height=100, crop_width=100)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=20, overall_folder='../datasets/13_ped_bridge_new_dataset/', out_folder='../datasets/13_ped_bridge_new_dataset/paper_350/', img_df='../datasets/13_ped_bridge_new_dataset/1_processed/train_high_light_img_dataset.csv', crop_height=350, crop_width=350)
