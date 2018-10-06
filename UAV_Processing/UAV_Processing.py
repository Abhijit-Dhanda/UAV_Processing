# UAV IMAGE PROCESSING
# This script uses metadata to analyze the overlap patterns of grid based UAV flights for photogrammetry

# IMPORT PACKAGES
import os
import shutil
import exiftool
import math
from datetime import datetime as dt
import operator
import geopy
import geopy.distance as gd
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from collections import defaultdict
import numpy as np
from sympy import Plane, Point3D, Ray3D
import transforms3d as td
from gooey import Gooey
from gooey import GooeyParser


# FUNCTIONS

@Gooey(program_name="UAV Processing", default_size=(1000, 600))
def parse_args():
    # Parse Args

    UAV_parser = GooeyParser(description="This script uses metadata to analyze the overlap patterns of grid based UAV "
                            "flights for photogrammetry")
    UAV_parser.add_argument('Image_Directory', type=str, help="File path to the dataset of images", widget='DirChooser')
    UAV_parser.add_argument('Print_Metadata', type=str, choices = ['True', 'False'], help='Print metadata to a file '
                            'in the image directory')
    UAV_parser.add_argument('Sensor_Width', help="Width of the UAV camera sensor in mm")
    UAV_parser.add_argument('Sensor_Height', help="Height of the UAV camera sensor in mm")
    UAV_parser.add_argument('End_Overlap', type=int, default=80, help='Percent of end overlap desired.')
    UAV_parser.add_argument('Side_Overlap', type=int, default=60, help='Percent of side overlap desired.')
    UAV_parser.add_argument('Metadata_Image_Width', type=str, default='EXIF:ExifImageWidth', help='The metadata tag '
                            'for the image width.')
    UAV_parser.add_argument('Metadata_Image_Height', type=str, default='EXIF:ExifImageHeight', help='The metadata tag '
                            'for the image height.')
    UAV_parser.add_argument('Metadata_Focal_Length', type=str, default='EXIF:FocalLength', help='The metadata tag '
                            'for the focal length.')
    UAV_parser.add_argument('Metadata_GPS_Latitude', type=str, default='EXIF:GPSLatitude', help='The metadata tag '
                            'for the GPS Latitude.')
    UAV_parser.add_argument('Metadata_GPS_Longitude', type=str, default='EXIF:GPSLongitude', help='The metadata tag '
                            'for the GPS Longitude.')
    UAV_parser.add_argument('Metadata_Relative_Altitude', type=str, default='XMP:RelativeAltitude',
                            help='The metadata tag for the Relative Altitude.')
    UAV_parser.add_argument('Metadata_Flight_Yaw', type=str, default='XMP:FlightYawDegree', help='The metadata '
                            'tag for the Flight Yaw Angle.')
    UAV_parser.add_argument('Metadata_Gimbal_Yaw', type=str, default='XMP:GimbalYawDegree', help='The metadata '
                            'tag for the Gimbal Yaw Angle.')
    UAV_parser.add_argument('Metadata_Gimbal_Pitch', type=str, default='XMP:GimbalPitchDegree', help='The metadata '
                            'tag for the Gimbal Pitch Angle.')
    UAV_parser.add_argument('Metadata_Gimbal_Roll', type=str, default='XMP:GimbalRollDegree', help='The metadata '
                            'tag for the Gimbal Roll Angle.')
    args = UAV_parser.parse_args()
    return args

def exifread(image_loc, width, height, focal, latitude, longitude, gim_yaw, gim_pitch, gim_roll, uav_yaw, altitude ):
    # EXIF READ: Read and store the specific EXIF tags of an image
    img_width, img_height, cam_yaw, cam_pitch, cam_roll, flight_yaw, rel_alt, \
        foc_len, lat, lon = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    with exiftool.ExifTool() as et:
        try:
            img_width = float(et.get_tag(width, image_loc))
            img_height = float(et.get_tag(height, image_loc))
            foc_len = float(et.get_tag(focal, image_loc))
            lat = (et.get_tag(latitude, image_loc))
            lon = (et.get_tag(longitude, image_loc))
            cam_yaw = float(str(et.get_tag(gim_yaw, image_loc)))
            cam_pitch = float(str(et.get_tag(gim_pitch, image_loc)))
            cam_roll = float(str(et.get_tag(gim_roll, image_loc)))
            flight_yaw = float(str(et.get_tag(uav_yaw, image_loc)))
            rel_alt = float(str(et.get_tag(altitude, image_loc)))

            return img_width, img_height, cam_yaw, cam_pitch, cam_roll, flight_yaw, rel_alt, \
                foc_len, lat, lon
        except:
            return img_width, img_height, cam_yaw, cam_pitch, cam_roll, flight_yaw, rel_alt, \
                foc_len, lat, lon

def bearing(lat_01, lon_01, lat_02, lon_02):
    lat_01 = math.radians(lat_01); lon_01 = math.radians(lon_01); lat_02 = math.radians(lat_02)
    lon_02 = math.radians(lon_02)

    dist_lon = lon_02 - lon_01

    d_phi = math.log(math.tan(lat_02/2.0+math.pi/4.0)/math.tan(lat_01/2.0+math.pi/4.0))
    if abs(dist_lon) > math.pi:
        if dist_lon > 0.0:
            dist_lon = -(2.0 * math.pi - dist_lon)
        else:
            dist_lon = (2.0 * math.pi + dist_lon)

    print(math.degrees(math.atan2(dist_lon, d_phi)) + 360.0) % 360.0
    return (math.degrees(math.atan2(dist_lon, d_phi)) + 360.0) % 360.0


def grndpolygon(rel_alt, flight_yaw, cam_yaw, cam_pitch, cam_roll, sens_w, sens_h, foc_len, origin_01, lat_01,
                lon_01, lat_02, lon_02, strip_info):
    # Calculate the ground polygon of the image

    # Calculate vertical and horizontal Field of View and convert for corner calculations
    h_fov = 2 * math.atan(sens_w/(2*foc_len))
    th_fov = math.tan(h_fov/2)
    v_fov = 2 * math.atan(sens_h/(2*foc_len))
    tv_fov = math.tan(v_fov/2)

    # Create rotation matrix
    roll2 = math.radians(cam_roll); pitch2 = math.radians(cam_pitch); yaw2 = math.radians(cam_yaw+180)
    rm = td.euler.euler2mat(yaw2, pitch2, roll2, axes='rzyx')

    # Calculate ground plane based on relative altitude
    plane_normal = rel_alt - 1
    ground_plane = Plane(Point3D(0, 0, rel_alt), normal_vector=(0, 0, plane_normal))

    # Calculate position of the camera in Cartesian coordinates.
    # The first image is the origin (0,0)
    if strip_info == 0:
        # First image in the first strip
        cam_x = 0; cam_y = 0
    elif strip_info == 1:
        # 2nd to last images in the stip
        gps_01 = geopy.Point(lat_01, lon_01)
        gps_02 = geopy.Point(lat_02, lon_02)
        dist = gd.vincenty(gps_01, gps_02).meters
        disp_x = math.cos(math.radians(flight_yaw)) * dist
        disp_y = math.sin(math.radians(flight_yaw)) * dist
        cam_x = origin_01[0] + disp_x
        cam_y = origin_01[1] + disp_y
    else:
        # First image in the other strips
        # Find bearing
        gps_01 = geopy.Point(lat_01, lon_01)
        gps_02 = geopy.Point(lat_02, lon_02)
        dist = gd.vincenty(gps_01, gps_02).meters
        angle = bearing(lat_01, lon_01, lat_02, lon_02)
        disp_x = math.cos(math.radians(angle)) * dist
        disp_y = math.sin(math.radians(angle)) * dist
        cam_x = origin_01[0] + disp_x
        cam_y = origin_01[1] + disp_y

    # Corner 1 of the ground projection:
    # Calculate the direction, the ground intersection, then the distance and angle from the origin
    direction_1 = np.array([1, th_fov, tv_fov]).reshape((3, 1))
    rot_dir_1 = np.dot(rm, direction_1)
    rot_ray_1 = Ray3D(Point3D(0, 0, 0), Point3D(rot_dir_1))
    grnd_int_1 = np.array(rot_ray_1.intersection(ground_plane))
    corner_1x = round(grnd_int_1[0, 0] + cam_x, 4)
    corner_1y = round(grnd_int_1[0, 1] + cam_y, 4)

    # Corner 2 of the ground projection:
    # Calculate the direction, the ground intersection, then the distance and angle from the origin
    direction_2 = np.array([1, -th_fov, tv_fov]).reshape((3, 1))
    rot_dir_2 = np.dot(rm, direction_2)
    rot_ray_2 = Ray3D(Point3D(0, 0, 0), Point3D(rot_dir_2))
    grnd_int_2 = np.array(rot_ray_2.intersection(ground_plane))
    corner_2x = round(grnd_int_2[0, 0] + cam_x, 4)
    corner_2y = round(grnd_int_2[0, 1] + cam_y, 4)

    # Corner 3 of the ground projection:
    # Calculate the direction, the ground intersection, then the distance and angle from the origin
    direction_3 = np.array([1, -th_fov, -tv_fov]).reshape((3, 1))
    rot_dir_3 = np.dot(rm, direction_3)
    rot_ray_3 = Ray3D(Point3D(0, 0, 0), Point3D(rot_dir_3))
    grnd_int_3 = np.array(rot_ray_3.intersection(ground_plane))
    corner_3x = round(grnd_int_3[0, 0] + cam_x, 4)
    corner_3y = round(grnd_int_3[0, 1] + cam_y, 4)

    # Corner 4 of the ground projection:
    # Calculate the direction, the ground intersection, then the distance and angle from the origin
    direction_4 = np.array([1, th_fov, -tv_fov]).reshape((3, 1))
    rot_dir_4 = np.dot(rm, direction_4)
    rot_ray_4 = Ray3D(Point3D(0, 0, 0), Point3D(rot_dir_4))
    grnd_int_4 = np.array(rot_ray_4.intersection(ground_plane))
    corner_4x = round(grnd_int_4[0, 0] + cam_x, 4); corner_4y = round(grnd_int_4[0, 1] + cam_y, 4)

    # Create ground polygon
    poly = Polygon([(corner_1x, corner_1y), (corner_2x, corner_2y), (corner_3x, corner_3y), (corner_4x, corner_4y),
                    (corner_1x, corner_1y)])

    return poly, (cam_x, cam_y)


def overlap(poly_1, poly_2):
    # OVERLAP: Calculates the area of overlap of polygon 2 on polygon 1 and returns a percentage
    area_1 = shape(poly_1).area
    lap = poly_1.intersection(poly_2)
    if lap:
        lap_area = shape(lap).area
        over = (1 - (area_1 - lap_area)/area_1)*100
        return over
    else:
        return 0

def main():
    args = parse_args()

    # Store inputs
    images = args.Image_Directory
    if args.Print_Metadata == 'True':
        # Read metadata of one of the images
        count = 0
        for image_name in os.listdir(images):
            image_path = images + '/' + image_name
            if count == 0:
                break

        with exiftool.ExifTool() as et:
            info = str(et.get_metadata(image_path))
        # Write the metadata to a file
        name = images + '/' + 'Metadata.txt'
        met = open(name, "w")
        met.write(info)
        met.close()

    else:
        sens_w = float(args.Sensor_Width)
        sens_h = float(args.Sensor_Height)

        req_end = args.End_Overlap
        req_side = args.Side_Overlap

        # Metadata Tags
        met_image_width = args.Metadata_Image_Width
        met_image_height = args.Metadata_Image_Height
        met_focal_length = args.Metadata_Focal_Length
        met_lat = args.Metadata_GPS_Latitude
        met_lon = args.Metadata_GPS_Longitude
        met_rel_alt = args.Metadata_Relative_Altitude
        met_flight_yaw = args.Metadata_Flight_Yaw
        met_gimbal_yaw = args.Metadata_Gimbal_Yaw
        met_gimbal_pitch = args.Metadata_Gimbal_Pitch
        met_gimbal_roll = args.Metadata_Gimbal_Roll

        print('Sorting images by capture time...')
        # Create empty lists for sorting the files
        tot_names = []
        tot_cap_time = []

        # Loop through and sort the images based on capture time
        for image_name in os.listdir(images):
            file_path = images + '/' + image_name
            with exiftool.ExifTool() as et:
                try:
                    cap_time = str(et.get_tag('EXIF:DateTimeOriginal', file_path))
                    date = dt.strptime(cap_time, "%Y:%m:%d %H:%M:%S")

                    tot_names.append(image_name)
                    tot_cap_time.append(date)

                except:
                    print('Error reading capture time EXIF of {}.\n'.format(image_name))

        tot_par = zip(tot_names, tot_cap_time)
        tot_par.sort(key=operator.itemgetter(1))

        # Initial lists to store calculated values
        tot_rel_alt = []
        tot_foc_len = []
        tot_latt = []
        tot_long = []

        tot_poly = defaultdict(list)
        tot_poly['names'] = []
        tot_poly['polygons'] = []
        tot_poly['origin'] = []; tot_poly['origin'].append([0, 0])

        tot_flight_yaw = []
        strip = 0; im_num = 0
        tot_overlap = defaultdict(dict); tot_overlap[strip] = defaultdict(list)
        tot_overlap[strip]['image'] = []; tot_overlap[strip]['end_label'] = []; tot_overlap[strip]['side_label'] = []
        tot_overlap[strip]['poly'] = []; tot_overlap[strip]['orient'] = []
        tot_overlap[strip]['end_lap'] = []; tot_overlap[strip]['end_name'] = []
        tot_overlap[strip]['side_lap'] = []; tot_overlap[strip]['side_name'] = []; tot_overlap[strip]['side_strip'] = []

        print('Reading metadata and sorting images into strips:')
        # Loop through the sorted images
        for i, (img_title, time) in enumerate(tot_par):
            tot_path = images + '/' + img_title

            img_width, img_height, cam_yaw, cam_pitch, cam_roll, flight_yaw, rel_alt, foc_len, latt, lon = exifread(
                tot_path, met_image_width, met_image_height, met_focal_length, met_lat, met_lon, met_gimbal_yaw,
                met_gimbal_pitch, met_gimbal_roll, met_flight_yaw, met_rel_alt)

            if img_width != 0:
                tot_rel_alt.append(rel_alt)
                tot_foc_len.append(foc_len)
                tot_latt.append(latt)
                tot_long.append(lon)
                tot_flight_yaw.append(flight_yaw)

                # Calculate Ground Geometry of photo
                print img_title
                # Separate the flights into strips using the flight yaw
                # ASSUMPTION: Flight is in strips
                if i == 0:
                    polygon, origin = grndpolygon(rel_alt, flight_yaw, cam_yaw, cam_pitch, cam_roll, sens_w, sens_h,
                                    foc_len, tot_poly['origin'][i], latt, lon, latt, lon, 0)
                    tot_overlap[strip]['image'].append(img_title); tot_overlap[strip]['poly'].append(polygon)
                    tot_overlap[strip]['orient'].append('Horizontal'); im_num += 1

                elif abs(tot_flight_yaw[i - 1] - tot_flight_yaw[i]) <= 20:
                    polygon, origin = grndpolygon(rel_alt, flight_yaw, cam_yaw, cam_pitch, cam_roll, sens_w, sens_h,
                                    foc_len, tot_poly['origin'][i - 1], tot_latt[i - 1], tot_long[i - 1], latt, lon, 1)
                    tot_poly['origin'].append(origin)
                    tot_overlap[strip]['image'].append(img_title)
                    tot_overlap[strip]['poly'].append(polygon)
                    # Calculate Endlap between images and store in the dictionary for each strip
                    tot_overlap[strip]['end_lap'].append(round(overlap(tot_overlap[strip]['poly'][im_num - 1],
                                                                           tot_overlap[strip]['poly'][im_num]), 2))
                    tot_overlap[strip]['end_name'].append(img_title)
                    im_num += 1

                else:
                    polygon, origin = grndpolygon(rel_alt, flight_yaw, cam_yaw, cam_pitch, cam_roll, sens_w, sens_h,
                                    foc_len, tot_poly['origin'][i - 1], tot_latt[i - 1], tot_long[i - 1], latt, lon, 2)
                    tot_poly['origin'].append(origin)
                    tot_overlap[strip]['end_lap'].append(0.00); tot_overlap[strip]['end_name'].append(None)
                    im_num = 0; strip += 1
                    tot_overlap[strip]['image'] = []; tot_overlap[strip]['poly'] = []
                    tot_overlap[strip]['end_label'] = []; tot_overlap[strip]['side_label'] = []
                    tot_overlap[strip]['end_lap'] = []; tot_overlap[strip]['end_name'] = []
                    tot_overlap[strip]['side_lap'] = []; tot_overlap[strip]['side_name'] = []
                    tot_overlap[strip]['orient'] = []; tot_overlap[strip]['side_strip'] = []
                    tot_overlap[strip]['image'].append(img_title); tot_overlap[strip]['poly'].append(polygon)
                    if 20 < abs(tot_flight_yaw[i - 1] - tot_flight_yaw[i]) < 110:
                        tot_overlap[strip]['orient'].append('Vertical')
                    else:
                        tot_overlap[strip]['orient'].append('Horizontal')

            else:
                (tot_rel_alt, tot_foc_len, tot_latt, tot_long, tot_flight_yaw).append(None)
                print 'Error reading EXIF values of {}.\n'.format(img_title)

        # To account for the last image in the set
        tot_overlap[strip]['end_lap'].append(0.00)
        tot_overlap[strip]['end_name'].append(None)

        # Calculate Sidelap for each strip
        # ASSUMPTION: Only a slight elevation change in the terrain, but not drastic
        for row in tot_overlap:
            if len(tot_overlap[row]['image']) > 1 and row != strip:
                count = row + 1
                while tot_overlap[row]['orient'][0] != tot_overlap[count]['orient'][0]:
                    count += 1
                tot_overlap[row]['side_strip'].append(count)
                rows_lap = []
                rows_index = []
                for j, gon_1 in enumerate(tot_overlap[row]['poly']):
                    # Find the sidelap of the adjacent images and store in the local list
                    loc_side_lap = []
                    loc_index = []
                    for spot, gon_2 in enumerate(tot_overlap[count]['poly']):
                        if gon_1.intersection(gon_2):
                            lapp = overlap(gon_1, gon_2)
                            loc_side_lap.append(lapp)
                            loc_index.append(spot)
                    if len(loc_side_lap):
                        max_side = (max(loc_side_lap))
                        rows_lap.append(max_side)  # Max sidelap
                        max_index = loc_index[loc_side_lap.index(max_side)]
                        rows_index.append(max_index)
                        tot_overlap[row]['side_lap'].append(max_side)
                        tot_overlap[row]['side_name'].append(tot_overlap[count]['image'][max_index])
                    else:
                        rows_lap.append(0.00)
                        rows_index.append('None')

            else:
                tot_overlap[row]['side_lap'].append(0.00)
                tot_overlap[row]['side_name'].append(None)

        for i in tot_overlap[0]['side_lap']:
            print i


        filtered = []

        # Filter images based on end overlap
        for strip in tot_overlap:
            j = 0
            for i, end in enumerate(tot_overlap[strip]['end_lap']):
                if j == i:
                    continue
                else:
                    if end > req_end - 5:
                        if overlap(tot_overlap[strip]['poly'][i-1], tot_overlap[strip]['poly'][i+1]) > req_end - 5:
                            j = i+1
                            filtered.append(tot_overlap[strip]['image'][i])

        # Filter images based on side overlap
        for row in tot_overlap:
            check = True
            for side in tot_overlap[row]['side_lap']:
                if side < req_end - 5 and side != 0.00:
                    check = False
            if check:
                if len(tot_overlap[row]['image']) > 1 and row != strip:
                    count = tot_overlap[row]['side_strip'] + 1
                    while tot_overlap[row]['orient'][0] != tot_overlap[count]['orient'][0]:
                        count += 1
                    rows_lap = []
                    rows_index = []
                    for j, gon_1 in enumerate(tot_overlap[row]['poly']):
                        # Find the sidelap of the next row and store in the local list
                        loc_side_lap = []
                        loc_index = []
                        for spot, gon_2 in enumerate(tot_overlap[count]['poly']):
                            if gon_1.intersection(gon_2):
                                lapp = overlap(gon_1, gon_2)
                                loc_side_lap.append(lapp)
                                loc_index.append(spot)
                        if len(loc_side_lap):
                            max_side = (max(loc_side_lap))
                            rows_lap.append(max_side)  # Max sidelap
                            max_index = loc_index[loc_side_lap.index(max_side)]
                            rows_index.append(max_index)
                        else:
                            rows_lap.append(0.00)
                            rows_index.append('None')
                        check = True
                        for lap in rows_lap:
                            if lap < req_side - 5 and lap != 0.00:
                                check = False
                        if check:
                            s = tot_overlap[row]['side_strip'][0]
                            for n in tot_overlap[s]['image']:
                                if n not in filtered:
                                    filtered.append(n)

        print ("Filtered: {}".format(filtered))

        # Move filtered images to new folder
        new_folder = os.path.join(images, 'Filtered')
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for i in filtered:
            shutil.move(os.path.join(images, i), new_folder)

if __name__ == "__main__":
        main()
