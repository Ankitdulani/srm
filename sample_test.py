import srm
if __name__ == "__main__":
	import sys
	import cv2
	
	im = cv2.imread(sys.argv[1])
	q = int(sys.argv[2]) 

	algo = srm.srm()
	segmented = algo.execute(im,q)
	print(segmented.shape)

	cv2.imwrite("srm_result.jpg",segmented)