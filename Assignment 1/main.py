import cv2
import math
import numpy as np
import os
import sys

ip_folder = sys.argv[1]

def preprocess(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame

def params():
	k = 3 #gaussians
	alpha = 0.03 #alpha
	return k, alpha

def match(val, mu, sigm):
	dist = abs(val-mu)
	if dist<=(2.5)*sigm:
		return True, dist
	else:
		return False, dist

image_files = [f for f in os.listdir(ip_folder) if f.endswith(".jpg") or f.endswith(".png")]
image_files.sort()


frame = cv2.imread(os.path.join(ip_folder, image_files[0]))

ht, wt, dim = frame.shape
k, alpha = params()
wThresh = 0.6
bgTol = 30

N = 3
n_frames = []
for i in range(N):
	frame = cv2.imread(os.path.join(ip_folder, image_files[i]))
	frame = preprocess(frame)
	n_frames.append(frame)

for i in range(N):
	image_files.pop(0)



sigma_val = 10.0

mu = np.zeros((wt, ht, k))

sigma = np.ones((wt, ht, k)) * sigma_val

w = np.ones((wt, ht, k)) * float(1/k)

fore = np.zeros((ht, wt), np.uint8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writeFore = cv2.VideoWriter("Foreground.mp4", fourcc, 5, (wt, ht), False)
writeProc = cv2.VideoWriter("Foreground_Processed.mp4", fourcc, 5, (wt, ht), False)
writeBox = cv2.VideoWriter("Foreground_BoundingBox.mp4", fourcc, 5, (wt, ht), False)


def updatew(x, y, val):
	sum_v = 0
	for i in range(k):
		if i == val:
			w[x][y][i] = (1-alpha)*w[x][y][i] + alpha
		else:
			w[x][y][i] = (1-alpha)*w[x][y][i]
		sum_v += w[x][y][i]

	for i in range(k):
		w[x][y][i] /= sum_v

def sortweights(x, y):
	for i in range(1, k):
		for j in range(k-i): 
			if (w[x][y][j]/sigma[x][y][j]) < (w[x][y][j+1]/sigma[x][y][j+1]):
				w[x][y][j], w[x][y][j+1] = w[x][y][j+1], w[x][y][j]
				sigma[x][y][j], sigma[x][y][j+1] = sigma[x][y][j+1], sigma[x][y][j]
				mu[x][y][j], mu[x][y][j+1] = mu[x][y][j+1], mu[x][y][j]

def rhocalc(sigm, v, m):
	return alpha*(1/((2*np.pi*sigm*sigm)**0.5))*np.exp((-0.5*((v-m)**2 ))/(sigm**2))

def updatems(x, y, i, p, avg, n_fr):

	rho = rhocalc(p, mu[x][y][i], sigma[x][y][i])

	mu[x][y][i] = (1-rho)*avg[y][x] + rho*p

	s = 0
	for z in range(N):
		s += (n_fr[z][y][x] - mu[x][y][i])**2
	sig_v = (1-rho)*s + rho*((p-mu[x][y][i])**2)
	sigma[x][y][i] = math.sqrt(sig_v/N)


def fitgaussian(x, y, pix, avg, n_fr):
	matching = False
	gaussian_num = -1

	v = pix
	ind = -1
	dist_measure = float('inf')

	for i in range(k):
		match_bool, dist = match(v, mu[x][y][i], sigma[x][y][i])
		if(match_bool):
			matching = True
			
			if dist < dist_measure:
				dist_measure = dist 
				ind = i

	if matching==True:
		gaussian_num = ind
		updatew(x, y ,ind)
		updatems(x, y, ind, pix, avg, n_fr)
		sortweights(x, y)
		return matching
	if not matching:
		sortweights(x, y)
		w[x][y][k-1] = 0.33/k
		mu[x][y][k-1] = pix
		sigma[x][y][k-1] = sigma_val
		updatew(x, y, k-1)

	return matching

def upgradebackfore(y, x, isFit):
	wSum = 0
	i, v = 0, 0
	while(wSum<wThresh):
		v += w[x][y][i]*mu[x][y][i]

		wSum += w[x][y][i]
		i+=1

	v /= wSum 

	if(abs(frame[y][x] - v) > bgTol or not(isFit)):
		fore[y][x] = 255
	else:
		fore[y][x] = 0

	
def post_process(fr):
    _, binary_frame = cv2.threshold(fr, 128, 255, cv2.THRESH_BINARY)

    # function is used to find connectd components
    op = cv2.connectedComponentsWithStats(binary_frame, 4, cv2.CV_32S)
    num_labels, labels, stats, cen = op

    # define the desired size range for objects
    min_size = 50
    max_size = 100000

    # Create an empty black image to store the aggregated pixels
    result_image = np.zeros_like(fr)
    

    # iterate through each connected component
    for k in range(1, num_labels):
        size = stats[k, cv2.CC_STAT_AREA]
        if min_size <= size <= max_size:
            # get the coordinates of the rectangular region for the component
            xi, yi, wi, hi = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            # integrate the pixels within the region
            object_pixels = fr[yi:yi+hi, xi:xi+wi]
            # Aggregate the pixels and store them in the result image
            result_image[yi:yi+hi, xi:xi+wi] = object_pixels

    return result_image

### Noise removal using Integral images ###
def remove_noise(fr, threshold, kernel_size):
    h, w = fr.shape[:2]
    integral = cv2.integral(fr)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for r in range(kernel_size//2, h-(kernel_size//2), kernel_size):
        for c in range(kernel_size//2, w-(kernel_size//2), kernel_size):
            x1, y1 = max(0, r-kernel_size//2), max(0, c-kernel_size//2)
            x2, y2 = min(h-1, r+kernel_size//2), min(w-1, c+kernel_size//2)
            region = integral[x2, y2] - integral[x2, y1-1] - integral[x1-1, y2] + integral[x1-1, y1-1]
            if region >= threshold:
                fr[x1:x2+1, y1:y2+1] = 255
    return fr


### Pixel aggregation ###

def merge(r1, r2):
    # Merge rectangles r1 and r2 into a single rectangle
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)
    x2 = max(x2_1, x2_2)
    y2 = max(y2_1, y2_2)
    return x1, y1, x2, y2

def overlap(r1, r2):
    # Check if rectangles r1 and r2 overlap
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    return (x1_1 < x2_2) and (x2_1 > x1_2) and (y1_1 < y2_2) and (y2_1 > y1_2)

def close(r1, r2, min_distance=20):
    # Check if rectangles r1 and r2 are close to each other
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    center_1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
    center_2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
    distance = np.sqrt((center_1[0] - center_2[0]) ** 2 + (center_1[1] - center_2[1]) ** 2)
    return distance < min_distance

def filter_noise(stats, centroids, min_size=100, max_size=10000):
    # Filter out small and large regions
    valid_regions = []
    for i in range(len(stats)):
        size = stats[i, cv2.CC_STAT_AREA]
        if min_size <= size <= max_size:
            valid_regions.append(i)
    return valid_regions

def merge_rectangles(rectangles, min_distance=20):
    # Merge rectangles that overlap or are too close to each other
    merged_rectangles = []
    while rectangles:
        r1 = rectangles.pop()
        merged = False
        for i, r2 in enumerate(merged_rectangles):
            if overlap(r1, r2) or close(r1, r2, min_distance):
                merged_rectangles[i] = merge(r1, r2)
                merged = True
                break
        if not merged:
            merged_rectangles.append(r1)
    return merged_rectangles

def aggregate_foreground_pixels(image, threshold=127, min_size=100, max_size=10000, min_distance=20):
    # Threshold the image to segment foreground from background
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    
    # Filter out small and large regions
    valid_regions = filter_noise(stats, centroids, min_size, max_size)
    
    # Draw rectangles around the remaining regions
    rectangles = []
    for i in valid_regions:
        x, y, w, h, _ = stats[i]
        rectangles.append((x, y, x + w, y + h))
        
    # Merge rectangles that overlap or are too close to each other
    merged_rectangles = merge_rectangles(rectangles, min_distance)
    
    # Create final image
    final_image = np.zeros_like(image)
    for x1, y1, x2, y2 in merged_rectangles:
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
    return final_image

###

for image_file in image_files:


	frame = cv2.imread(os.path.join(ip_folder, image_file))
	cv2.imshow("Original Frame", frame)

	if frame is None:
		break

	ifFit = False

	frame = preprocess(frame)
	

	init_avg = np.mean(np.array(n_frames), axis = 0)

	for x in range(wt):
		for y in range(ht):
			pix = frame[y][x]
			isFit= fitgaussian(x, y, pix, init_avg, n_frames)

			upgradebackfore(y, x, isFit)

	cv2.medianBlur(fore, 5, fore)

	cv2.imshow('PreProcessed', fore)
	# Preprocessing
	writeFore.write(fore)

	kernel_size = 5
	processed_frame = remove_noise(fore, threshold=(kernel_size**2//2), kernel_size=3)

	cv2.imshow('Processed', processed_frame)
	
	processed_frame = post_process(processed_frame)
	cv2.imshow('PostProcessed', processed_frame)
	writeProc.write(processed_frame)

	processed_frame = aggregate_foreground_pixels(processed_frame)
	cv2.imshow('Bounding box', processed_frame)
	writeBox.write(processed_frame)


	n_frames.append(frame)

	if len(n_frames) > N:
		n_frames.pop(0)

	if cv2.waitKey(1) and 0xFF == ord('q'):
		break


writeFore.release()
writeProc.release()
cv2.destroyAllWindows()