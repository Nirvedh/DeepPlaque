import SimpleITK as sitk
import numpy as np
def setup2d (array,img3d):
	""" This is to make the image 2d for resizing """
	img = sitk.GetImageFromArray(array)
	img.SetSpacing(img3d.GetSpacing()[0:-1])
	img.SetOrigin(img3d.GetOrigin()[0:-1])
	#img.SetDirection(img3d.GetDirection()[0:-1]) cant do this so hoping directions are fine by default
	return img
def setup3d (img,img3d):
	""" This is to setup the thrid dimension of 2d image"""
	array = sitk.GetArrayFromImage(img)
	#print (array.shape)
	array3d = np.expand_dims(array,axis=0)
	#print (array3d.shape)
	returnImg = sitk.GetImageFromArray(array3d)
	returnImg.SetSpacing([img.GetSpacing()[0],img.GetSpacing()[1],img3d.GetSpacing()[2]])
	returnImg.SetOrigin([img.GetOrigin()[0],img.GetOrigin()[1],img3d.GetOrigin()[2]])
	return returnImg

def resizeimage (sizes,imageType,img3d,reference_img3d=0):
	if sizes==0:
		#print(img3d.GetSize(),reference_img3d.GetSize())
		imgArray = np.squeeze(sitk.GetArrayFromImage(img3d))
		dimension=imgArray.ndim
		referenceArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
		img = setup2d(imgArray,img3d)
		reference_img = setup2d(referenceArray,reference_img3d)
		reference_origin = reference_img.GetOrigin()
		reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
		transform = sitk.AffineTransform(dimension)
		transform.SetMatrix(img.GetDirection())
		transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
		# Modify the transformation to align the centers of the original and reference image instead of their origins.
		#centering_transform = sitk.TranslationTransform(dimension)
		#img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
		#centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
		centered_transform = sitk.Transform(transform)
		#centered_transform.AddTransform(centering_transform)
		# Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth 
		# segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that 
		# no new labels are introduced.
		if(imageType==0):
			img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
		else:
			img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

		return setup3d(img3,img3d)	
	else:
		imgArray = np.squeeze(sitk.GetArrayFromImage(img3d))
		referenceimgArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
		dimension=imgArray.ndim
		dim=imgArray.ndim-1
		spacing=reference_img3d.GetSpacing()
		referencespacing=[]
		referenceArray=np.zeros([sizes[1],sizes[0]])
		for i in range(dimension):
			referencespacing.append(spacing[i]*referenceimgArray.shape[dim-i]/sizes[i])
		#print(referencespacing)
		if(referencespacing[1]*sizes[1]<img3d.GetSpacing()[1]*imgArray.shape[0]):
			#print('input image has larger FOV')

			img = setup2d(imgArray,img3d)
			reference_img = sitk.GetImageFromArray(referenceArray)
			reference_img.SetSpacing(referencespacing)
			reference_img.SetOrigin(img.GetOrigin())
			reference_origin = reference_img.GetOrigin()
			index=np.array(reference_img.GetSize())-1.0
			#print(index,list(index))
			#print(reference_img.TransformContinuousIndexToPhysicalPoint(list(index)))
			reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(index))
			transform = sitk.AffineTransform(dimension)
			transform.SetMatrix(img.GetDirection())
			transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
			# Modify the transformation to align the centers of the original and reference image instead of their origins.
			centering_transform = sitk.TranslationTransform(dimension)
			img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())-1.0))
			centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
			centered_transform = sitk.Transform(transform)
			centered_transform.AddTransform(centering_transform)
			# Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
			# segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
			# no new labels are introduced.
			if(imageType==0):
				img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
			else:
				img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

			return setup3d(img3,img3d)
		else:
			#print('input image has smaller FOV')			
			#referenceArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
			img = setup2d(imgArray,img3d)
			reference_img = sitk.GetImageFromArray(referenceArray)
			reference_img.SetSpacing(referencespacing)
			reference_img.SetOrigin(img.GetOrigin())
			reference_origin = reference_img.GetOrigin()
			reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
			transform = sitk.AffineTransform(dimension)
			transform.SetMatrix(img.GetDirection())
			transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
			# Modify the transformation to align the centers of the original and reference image instead of their origins.
			#centering_transform = sitk.TranslationTransform(dimension)
			#img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
			#centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
			centered_transform = sitk.Transform(transform)
			#centered_transform.AddTransform(centering_transform)
			# Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
			# segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
			# no new labels are introduced.
		if(imageType==0):
			img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
		else:
                	img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

		return setup3d(img3,img3d)
		

def resizeimageactual (sizes,imageType,img3d,reference_img3d,x,y,angle,scalex,scaley,shearx,sheary):
        if sizes==0:
                #print(img3d.GetSize(),reference_img3d.GetSize())
                imgArray = np.squeeze(sitk.GetArrayFromImage(img3d))
                dimension=imgArray.ndim
                referenceArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
                img = setup2d(imgArray,img3d)
                reference_img = setup2d(referenceArray,reference_img3d)
                reference_origin = reference_img.GetOrigin()
                reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
                transform = sitk.AffineTransform(dimension)
                transform.SetMatrix(img.GetDirection())
                transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin+(x,y))
                matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
                matrix[0,0] = scalex
                matrix[1,1] = scaley
                transform.SetMatrix(matrix.ravel())
                # Modify the transformation to align the centers of the original and reference image instead of their origins.
                #centering_transform = sitk.TranslationTransform(dimension)
                #img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
                #centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
                centered_transform = sitk.Transform(transform)
                #centered_transform.AddTransform(centering_transform)
                # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
                # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
                # no new labels are introduced.
                if(imageType==0):
                        img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
                else:
                        img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

                return setup3d(img3,img3d)
        else:
                imgArray = np.squeeze(sitk.GetArrayFromImage(img3d))
                referenceimgArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
                dimension=imgArray.ndim
                dim=imgArray.ndim-1
                spacing=reference_img3d.GetSpacing()
                referencespacing=[]
                referenceArray=np.zeros([sizes[1],sizes[0]])
                for i in range(dimension):
                        referencespacing.append(spacing[i]*referenceimgArray.shape[dim-i]/sizes[i])
                #print(referencespacing)
                if(referencespacing[1]*sizes[1]<img3d.GetSpacing()[1]*imgArray.shape[0]):
                        #print('input image has larger FOV')

                        img = setup2d(imgArray,img3d)
                        reference_img = sitk.GetImageFromArray(referenceArray)
                        reference_img.SetSpacing(referencespacing)
                        reference_img.SetOrigin(img.GetOrigin())
                        reference_origin = reference_img.GetOrigin()
                        index=np.array(reference_img.GetSize())-1.0
                        #print(index,list(index))
                        #print(reference_img.TransformContinuousIndexToPhysicalPoint(list(index)))
                        index=index-(x,y)
                        reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(index))
                        transform = sitk.AffineTransform(dimension)
                        transform.SetMatrix(img.GetDirection())
                        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
                        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
                        matrix[0,0] = scalex
                        matrix[1,1] = scaley
                        matrix[0,1] = shearx
                        matrix[1,0] = sheary

                        radians = -np.pi * angle / 180.
                        rotation = np.array([[np.cos(radians), -np.sin(radians)],[np.sin(radians), np.cos(radians)]])
                        matrix = np.dot(rotation, matrix)                     
                        transform.SetMatrix(matrix.ravel())

                        # Modify the transformation to align the centers of the original and reference image instead of their origins.
                        centering_transform = sitk.TranslationTransform(dimension)
                        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())-1.0))
                        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
                        centered_transform = sitk.Transform(transform)
                        centered_transform.AddTransform(centering_transform)
                        # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
                        # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
                        # no new labels are introduced.
                        if(imageType==0):
                                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
                        else:
                                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

                        return setup3d(img3,img3d)
                else:
                        #print('input image has smaller FOV')
                        #referenceArray = np.squeeze(sitk.GetArrayFromImage(reference_img3d))
                        img = setup2d(imgArray,img3d)
                        reference_img = sitk.GetImageFromArray(referenceArray)
                        reference_img.SetSpacing(referencespacing)
                        reference_img.SetOrigin(img.GetOrigin())
                        reference_origin = reference_img.GetOrigin()
                        reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
                        index=(x,y)
                        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
                        transform = sitk.AffineTransform(dimension)
                        transform.SetMatrix(img.GetDirection())
                        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin-trans)
                        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
                        matrix[0,0] = scalex
                        matrix[1,1] = scaley
                        matrix[0,1] = shearx
                        matrix[1,0] = sheary

                        radians = -np.pi * angle / 180.
                        rotation = np.array([[np.cos(radians), -np.sin(radians)],[np.sin(radians), np.cos(radians)]])
                        matrix = np.dot(rotation, matrix)
                        transform.SetMatrix(matrix.ravel())

                        # Modify the transformation to align the centers of the original and reference image instead of their origins.
                        #centering_transform = sitk.TranslationTransform(dimension)
                        #img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
                        #centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
                        centered_transform = sitk.Transform(transform)
                        #centered_transform.AddTransform(centering_transform)
                        # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
                        # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
                        # no new labels are introduced.
                if(imageType==0):
                        img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
                else:
                        img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)

                return setup3d(img3,img3d)
	






#img2=sitk.ReadImage('../train/pat20050_L_BULB_LONG_cont_20130910085802_bmode_33.mha')
##img1=sitk.ReadImage('referenceImages/pat20002leftcont_20110309114255_bmode_67.mha')
#img1=sitk.ReadImage('../train_segmentation/pat20050_L_BULB_LONG_cont_20130910085802_Segmentation_33.mha')
#img3=resizeimage([512,512],1,img1,img2)
#sitk.WriteImage(img3,'../outscriptaffine_bmodebase.mha')
#img3=resizeimageactual([512,512],1,img3,img2,x,y,angle,scalex,scaley,shearx,sheary)
#sitk.WriteImage(img3,'../outscriptaffine_bmode4.mha')
#resizeimage(0,0,img2)
	
