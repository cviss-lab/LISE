import cv2.aruco as aruco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2, shutil, math, re, random, functools, time, os, errno
from sklearn.metrics import mean_absolute_error
from glob import glob
from shapely.geometry import Polygon


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
        f_list = f.split('/')
        idx = f_list.index(data_folder)
        f_list = f_list[idx - len(f_list) + 1:-1]
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


def detect_marker(image, img, aruco_dict, parameters):
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
        corners = manually_select_marker(image)
        # corners = False
    elif len(corners) > 1:
        print("Found more than 1 markers in file: {}".format(img))
        # Manually find marker
        corners = manually_select_marker(image)
        # corners = False
    else:
        corners = corners[0][0]
    return corners


def manually_select_marker(image):
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
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    return cv2.warpPerspective(img, M, (width, height))


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
    # with _log_time_usage('regex: '):
    #     restricted_area = re.sub(' +', ' ', restricted_area.replace('\n', ' ').replace('[', ' ').replace(']', ' ')).strip().split(' ')
    #     restricted_area = np.array([(float(restricted_area[0]), float(restricted_area[1])),
    #                         (float(restricted_area[2]), float(restricted_area[3])),
    #                         (float(restricted_area[4]), float(restricted_area[5])),
    #                         (float(restricted_area[6]), float(restricted_area[7]))])
    crops = []
    for i in range(m_patches):
        crops.append(get_crops(restricted_area, n_crops, image, crop_width, crop_height, n_channels, margin))
        # crops.extend(get_crops(restricted_area, n_crops, image, crop_width, crop_height, n_channels, margin))
    return crops


def montage_crops(n_crops, crop_width, crop_height, n_channels, crops):
    rows = int(n_crops ** 0.5)
    tmp_image = np.zeros([crop_width * rows, crop_height * rows, n_channels], dtype='uint8')
    for i in range(rows):
        for j in range(rows):
            if n_channels == 1:
                tmp_image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width, 0] = crops[
                    i * rows + j]
            else:
                tmp_image[i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width, :] = crops[
                    i * rows + j]
    return tmp_image


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
            with _log_time_usage('getRandomSquareVertices: '):
                crop_vertices = getRandomSquareVertices((x,y), (crop_width/2, crop_height/2), rotation_angle)###############################################################################################
                # crop_vertices = [getRandomSquareVertices((x, y), (1000 / 2, 1000 / 2), rotation_angle),
                #                  getRandomSquareVertices((x, y), (800 / 2, 800 / 2), rotation_angle),
                #                  getRandomSquareVertices((x, y), (500 / 2, 500 / 2), rotation_angle)]
            with _log_time_usage('Polygon: '):
                crop_polygon = Polygon(
                    [(crop_vertices[0][0][0], crop_vertices[0][0][1]),
                     (crop_vertices[1][0][0], crop_vertices[1][0][1]),
                     (crop_vertices[2][0][0], crop_vertices[2][0][1]),
                     (crop_vertices[3][0][0], crop_vertices[3][0][1])])

                found_crop = not marker_polygon.intersects(crop_polygon)
            # # Show plots
            # x_plot, y_plot = crop_polygon.exterior.xy
            # plt.plot(x_plot, y_plot)
            # x_plot, y_plot = marker_polygon.exterior.xy
            # plt.plot(x_plot, y_plot)
            # plt.text(0, 0, found_crop, fontsize=12)
            # plt.show()
            if found_crop:
                with _log_time_usage('get_crop: '):
                    if n_channels == 1:
                        crops.append(get_crop(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), crop_vertices, crop_width, crop_height))###############################################################################################
                        # for cv, wid in zip(crop_vertices,[1000, 800, 500]):
                        #     crops.append(get_crop(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv, wid, wid))
                    else:
                        crops.append(get_crop(image, crop_vertices, crop_width, crop_height))
                break
            # attempt += 1
    with _log_time_usage('montage_crop: '):
        return montage_crops(n_crops, crop_width, crop_height, n_channels, crops) ############################################################################################### crops


def get_pix_per_len(corners, marker_len):
    dist = []
    for c in corners:
        tmp_dist = []
        for c_2 in corners:
            tmp_dist.append(((c_2[0] - c[0]) ** 2 + (c_2[1] - c[1]) ** 2) ** 0.5)
        tmp_dist = sorted(tmp_dist)[1:-1]
        dist.extend(tmp_dist)
    return np.average(dist) / marker_len


def extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution):
    # Initialize container for holding patch-wise information
    df_crop = {"original_fp": [], 'file': [], "pix_per_len": [], 'units': []}

    # If path to a dataframe containing marker information is provided, read the csv.
    if img_df is not None:
        df_img = pd.read_csv(img_df)
    # Otherwise, loop through the images to detect the markers to form the dataframe
    else:
        # Container to hold the dataframe
        df_img = {"original_fp": [], 'corners': [], "pix_per_len": [], 'units': []}
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
            corners = detect_marker(image, img, aruco_dict, parameters)
            # If the marker detection algorithm didn't work, manually select the marker
            if isinstance(corners, bool):
                corners = manually_select_marker(image)
            else:
                cont = False
                for c in corners:
                    if c[0] > 5000 or c[1] > 5000 or c[0] < 0 or c[1] < 0:
                        cont = True
                        break
                if cont:
                    corners = manually_select_marker(image)

            # Get pix_per_len
            pix_per_len = get_pix_per_len(corners, marker_len)
            # Check if marker was detected wierd. If true, then reselect marker corners
            if pix_per_len < 10 or pix_per_len > 500:
                if img_df is not None:
                    print(f"Skipped {img}, pix_per_len = {pix_per_len}")
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
        f_list = img.split('/')
        idx = f_list.index(data_folder)
        f_list = f_list[idx - len(f_list) + 1:-1]


        # Get marker corners
        if isinstance(row['corners'], np.ndarray):
            corners = row['corners']
        else:
            corners = re.sub(' +', ' ',
                             row['corners'].replace('\n', ' ').replace('[', ' ').replace(']', ' ')).strip().split(' ')
            corners = np.array([(float(corners[0]), float(corners[1])),
                                (float(corners[2]), float(corners[3])),
                                (float(corners[4]), float(corners[5])),
                                (float(corners[6]), float(corners[7]))])


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
                          equalize_distribution=False):
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

    # Extract crops to form the patch dataset
    if img_df is None:
        df_crop, df_img = extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution)
    else:
        df_crop = extract_crops_from_df_or_img(files, img_df, data_folder, out_folder, marker_len, units, crop_height, crop_width, n_crops, m_patches, equalize_distribution)

    # Save crop dataframes
    df_crop = pd.DataFrame(df_crop)
    df_crop.to_csv(f'{out_folder}/crop_dataset.csv', index=False)


if __name__ == '__main__':
    # To run the detection and output the dataframe containing corners of detected markers, run this code:
    # BW
    # create_n_by_n_markers(n_crops=1, m_patches=50, overall_folder='../datasets/BW/', out_folder='../datasets/BW/testing',
    #                       # img_df='../datasets/BW/2_processed/test_img_dataset.csv',
    #                       crop_height=850, crop_width=850,
    #                       # equalize_distribution=True
    #                       )


    # # PED_V2 BRIDGE 100
    # create_n_by_n_markers_from_df(n_crops=1, m_patches=50, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_test_100_final', img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', crop_height=100, crop_width=100, marker_len=9.4)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_train_100_final', img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', crop_height=100, crop_width=100, marker_len=9.4)
    # # PED_V2 BRIDGE 350
    # create_n_by_n_markers_from_df(n_crops=1, m_p atches=50, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_test_350_final', img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv', crop_height=350, crop_width=350, marker_len=9.4)
    # create_n_by_n_markers_from_df_equalize_distribution(n_crops=1, m_patches=15, overall_folder='../datasets/PED_V2/', out_folder='../datasets/PED_V2/3_train_350_final', img_df='../datasets/PED_V2/2_processed/train_img_dataset.csv', crop_height=350, crop_width=350, marker_len=9.4)
    # # PED_V2 BRIDGE 850
    create_n_by_n_markers(n_crops=1, m_patches=50,  # 50 patches per image
                          overall_folder='../datasets/PED_V2/', # folder that contains raw data
                          out_folder='../datasets/PED_V2/testing',  # contains cropped dataset
                          # img_df='../datasets/PED_V2/2_processed/test_img_dataset.csv',
                          crop_height=850, crop_width=850, marker_len=9.4)
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
