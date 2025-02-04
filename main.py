from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

#this is a very simple function that will calculate the line passing through p1 and p2 
def find_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    return a, b, c

#we will find the intersection of the lines defined by (p1, p2) and (p3, p4)
def find_foe(p1, p2, p3, p4):
    a1, b1, c1 = find_line(p1, p2)
    a2, b2, c2 = find_line(p3, p4)
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return np.array([x, y])

#find the distance between p3 and the line defined by p1 and p2
def find_dist(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v = p2 - p1
    w = p3 - p1
    projection = np.dot(w, v) / np.dot(v, v) * v
    return np.sqrt(np.dot(w - projection, w - projection))

#given old and new points, determine the FOE point. 
def ransac(good_new, good_old, img, attempts=100):
    best_foe = None
    best_count = 0
    found_outliers = set()
    found_outliers = None
    for attempt_number in range(attempts):
        outliers_indices = set()
        sample = np.random.randint(0, len(good_new), 2)
        ind1 = sample[0]
        ind2 = sample[1]
        if sample[0] == sample[1]:
            continue
        vec11 = good_old[ind1]
        vec12 = good_new[ind1]
        vec21 = good_old[ind2]
        vec22 = good_new[ind2]
        fpt = find_foe(vec11, vec12, vec21, vec22)
        h, w, _ = img.shape
        #if the points found are outside the image, don't intersect, or intersect at infinity, we don't consider foe
        if fpt is None:
            continue
        if np.isnan(fpt[0]) or np.isnan(fpt[1]):
            continue
        if fpt[0] < 0 or fpt[0] >= w or fpt[1] < 0 or fpt[1] >= h:
            continue
        threshold = 10
        count = 0
        for i in range(len(good_new)):
            dist = find_dist(good_new[i], good_old[i], fpt)
            if dist < threshold:
                count += 1
            else:
                outliers_indices.add(i)

        if count > best_count:
            best_foe = fpt
            best_count = count
            found_outliers = outliers_indices

    return best_foe, (len(good_new) / 5 < best_count), found_outliers

def group_nearby_vectors(good_new, good_old, outliers, num_clusters):
    good_oldcp = good_old[list(outliers)]
    good_newcp = good_new[list(outliers)]
    all_points = np.concatenate((good_newcp, good_oldcp), axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_points)
    clustered_indices = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(cluster_labels):
        clustered_indices[label].append(i)
    grouped_indices = [np.array(clustered_indices[label])
                       for label in range(num_clusters)]
    return grouped_indices, all_points


def draw_bounding_boxes(image, grouped_indices, all_points):
    for indices in grouped_indices:
        if len(indices) == 0:
            continue
        endpoints = all_points[indices]
        mean_endpoint = np.mean(endpoints, axis=0).astype(int)
        min_x = np.min(endpoints[:, [0, 2]])
        max_x = np.max(endpoints[:, [0, 2]])
        min_y = np.min(endpoints[:, [1, 3]])
        max_y = np.max(endpoints[:, [1, 3]])
        color = np.random.randint(0, 256, 3).tolist()
        cv2.rectangle(image, (int(min_x), int(min_y)),
                      (int(max_x), int(max_y)), color, 2)
        cv2.arrowedLine(image, (mean_endpoint[0], mean_endpoint[1]),
                        (mean_endpoint[2], mean_endpoint[3]), color, 2)

def draw_all(name1, name2):
	image1 = cv2.imread(name1)
	image2 = cv2.imread(name2)
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT_create()
	keypoints1 = sift.detect(gray1, None)
	p0 = np.array([kp.pt for kp in keypoints1], dtype=np.float32).reshape(-1, 1, 2)
	p1, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
 
	good_new = p1[status == 1]
	good_old = p0[status == 1]
	motion_vectors = good_new - good_old
	motion_magnitudes = np.linalg.norm(motion_vectors, axis=1).flatten()

	movement_threshold_lower = np.percentile(motion_magnitudes, 90)
	movement_threshold_upper = np.percentile(motion_magnitudes, 99)
	indices_to_draw = np.where((motion_magnitudes > movement_threshold_lower) &
							(motion_magnitudes < movement_threshold_upper))[0]
	good_new = good_new[indices_to_draw]
	good_old = good_old[indices_to_draw]

	foe, is_moving, outliers = ransac(good_new, good_old, image1)
	mask = np.zeros_like(image2)
	for i, new, old in zip(range(len(good_new)), good_new, good_old):
		if i in outliers:
			a, b = new.ravel().astype(int)
			c, d = old.ravel().astype(int)
			mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
			mask = cv2.circle(mask, (c, d), 5, (0, 255, 0), 5)
	if is_moving:
		print("Camera moving")
		foe_x, foe_y = foe.astype(int)
		foe_size = 25
		cv2.line(image2, (foe_x - foe_size, foe_y - foe_size), (foe_x + foe_size, foe_y + foe_size), (0, 0, 255), 2)
		cv2.line(image2, (foe_x - foe_size, foe_y + foe_size), (foe_x + foe_size, foe_y - foe_size), (0, 0, 255), 2)
	else:
		print("Camera not moving")
	output_image = cv2.add(image2, mask)
	plt.figure(figsize=(18, 14))


	plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()

	plt.figure(figsize=(18, 14))
	temp, all_pts = group_nearby_vectors(good_new, good_old, outliers, 4)
	draw_bounding_boxes(image2, temp, all_pts)
	plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()
