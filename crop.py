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

def getbound(a):
        min_x=10000
        min_y=10000
        max_x=-1
        max_y=-1
        rows=a.shape[0]
        cols=a.shape[1]		
        for y in range(0, rows):
                for x in range(0, cols):
                        if(a[y,x]>0):
                                if(x>max_x):
                                        max_x=x
                                if(y>max_y):
                                        max_y=y
                                if(x<min_x):
                                        min_x=x
                                if(y<min_y):
                                        min_y=y
        return[min_x,min_y,max_x,max_y]

def getguidance(a,padding,xind=0,yind=0):
        min_x=10000
        min_y=10000
        max_x=-1
        max_y=-1
        #print(a.shape)
        rows=a.shape[1]
        cols=a.shape[2]
        for y in range(0, rows):
                for x in range(0, cols):
                        if(a[0,y,x]>0):
                                if(x>max_x):
                                        max_x=x
                                if(y>max_y):
                                        max_y=y
                                if(x<min_x):
                                        min_x=x
                                if(y<min_y):
                                        min_y=y
        b = np.array(a,copy=True)
        max_x+=(padding+xind)
        max_y+=(padding+yind)
        min_x-=(padding-xind)
        min_y-=(padding-yind)
        #print(xind,yind,max_x,max_y)
        for y in range(0, rows):
                for x in range(0, cols):
                         if(x>max_x):
                                b[0,y,x]=0
                         elif(y>max_y):
                                b[0,y,x]=0
                         elif(x<min_x):
                                b[0,y,x]=0
                         elif(y<min_y):
                                b[0,y,x]=0
                         else:
                                b[0,y,x]=255

        return b


def getwindowdef(a,x_offset,y_offset,desired_padding,sizes):
        a=np.squeeze(sitk.GetArrayFromImage(a))
        [min_x,min_y,max_x,max_y]=getbound(a)
        center_x=(min_x+max_x)/2
        center_y=(min_y+max_y)/2
        #center_x=min_x-10
        #center_y=min_y-10
        if(max_x-min_x>max_y-min_y):
                radius=(max_x-min_x)/2
        else:
                radius=(max_y-min_y)/2
        if(desired_padding==0):
                padding=0
        else:    
                padding=radius/(sizes[0]/(2*desired_padding)-1)
        paddingx=padding-x_offset*(2*(radius+padding))/sizes[0]
        paddingy=padding-y_offset*(2*(radius+padding))/sizes[1]
        return[center_x-radius-paddingx,center_y-radius-paddingy,radius+padding]


def getwindowdefv2(a,x_offset,y_offset,desired_padding,sizes):
        a=np.squeeze(sitk.GetArrayFromImage(a))
        [min_x,min_y,max_x,max_y]=getbound(a)
        center_x=(min_x+max_x)/2
        center_y=(min_y+max_y)/2
        #center_x=min_x-10
        #center_y=min_y-10
        radiusx=(max_x-min_x)/2
        radiusy=(max_y-min_y)/2
        if(desired_padding==0):
                paddingx=0
                paddingy=0
        else:
                paddingx=radiusx/(sizes[0]/(2*desired_padding)-1)
                paddingy=radiusy/(sizes[0]/(2*desired_padding)-1)
        paddingx1=paddingx-x_offset*(2*(radiusx+paddingx))/sizes[0]
        paddingy1=paddingy-y_offset*(2*(radiusy+paddingy))/sizes[1]
        return[center_x-radiusx-paddingx1,center_y-radiusy-paddingy1,radiusx+paddingx,radiusy+paddingy]


def getwindow (sizes,imageType,img3d,reference_img3d=0,windowinfo=0):
        [center_x,center_y,radius]=windowinfo
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
        img = setup2d(imgArray,img3d)
        #print(imgArray.shape)
        [min_x,min_y,max_x,max_y]=getbound(imgArray)
        #print([min_x,min_y,max_x,max_y])
        reference_img = sitk.GetImageFromArray(referenceArray)
        reference_img.SetSpacing(referencespacing)
        #img.SetOrigin(((center_x,center_y)))
        reference_img.SetOrigin(img.GetOrigin())
        #img.SetOrigin(((center_x,center_y)))
        reference_origin = reference_img.GetOrigin()
        reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
        #transform = sitk.AffineTransform(dimension)
        #transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        #centering_transform = sitk.TranslationTransform(dimension)
        #img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        #centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        scalex=(2*radius)/sizes[0]
        scaley=(2*radius)/sizes[1]
        index=(center_x,center_y)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=list(trans)
        trans[0]=-trans[0]
        trans[1]=-trans[1]
        img.SetOrigin(trans)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=reference_center-trans
        #print(np.array(img.GetOrigin()) - reference_origin)
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin-trans)
        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
        #print((2*radius)/sizes[0])
        matrix[0,0] = scalex#(2*radius)/sizes[0]#x
        matrix[1,1] = scaley#(2*radius)/sizes[1]#y
        transform.SetMatrix(matrix.ravel())

        centered_transform = sitk.Transform(transform)
        if(imageType==0):
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
        else:
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)
        return setup3d(img3,img3d)


def getwindowv2 (sizes,imageType,img3d,reference_img3d=0,windowinfo=0):
        [center_x,center_y,radiusx,radiusy]=windowinfo
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
        img = setup2d(imgArray,img3d)
        #print(imgArray.shape)
        [min_x,min_y,max_x,max_y]=getbound(imgArray)
        #print([min_x,min_y,max_x,max_y])
        reference_img = sitk.GetImageFromArray(referenceArray)
        reference_img.SetSpacing(referencespacing)
        #img.SetOrigin(((center_x,center_y)))
        reference_img.SetOrigin(img.GetOrigin())
        #img.SetOrigin(((center_x,center_y)))
        reference_origin = reference_img.GetOrigin()
        reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
        #transform = sitk.AffineTransform(dimension)
        #transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        #centering_transform = sitk.TranslationTransform(dimension)
        #img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        #centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        scalex=(2*radiusx)/sizes[0]
        scaley=(2*radiusy)/sizes[1]
        index=(center_x,center_y)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=list(trans)
        trans[0]=-trans[0]
        trans[1]=-trans[1]
        img.SetOrigin(trans)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=reference_center-trans
        #print(np.array(img.GetOrigin()) - reference_origin)
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin-trans)
        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
        #print((2*radius)/sizes[0])
        matrix[0,0] = scalex#(2*radius)/sizes[0]#x
        matrix[1,1] = scaley#(2*radius)/sizes[1]#y
        transform.SetMatrix(matrix.ravel())

        centered_transform = sitk.Transform(transform)
        if(imageType==0):
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
        else:
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)
        return setup3d(img3,img3d)


def getreversewindowv2 (sizes,imageType,img3d,reference_img3d=0,windowinfo=0):
        [center_x,center_y,radiusx,radiusy]=windowinfo
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
        img = setup2d(imgArray,img3d)
        #print(imgArray.shape)
        [min_x,min_y,max_x,max_y]=getbound(imgArray)
        #print([min_x,min_y,max_x,max_y])
        reference_img = sitk.GetImageFromArray(referenceArray)
        reference_img.SetSpacing(referencespacing)
        #img.SetOrigin(((center_x,center_y)))
        reference_img.SetOrigin(img.GetOrigin())
        #img.SetOrigin(((center_x,center_y)))
        reference_origin = reference_img.GetOrigin()
        reference_center = np.array(reference_img.TransformContinuousIndexToPhysicalPoint(np.array(reference_img.GetSize())/2.0))
        #transform = sitk.AffineTransform(dimension)
        #transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        #centering_transform = sitk.TranslationTransform(dimension)
        #img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        #centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        scalex=(2*radiusx)/sizes[0]
        scaley=(2*radiusy)/sizes[1]
        index=(center_x,center_y)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=list(trans)
        trans[0]=trans[0]/scalex
        trans[1]=trans[1]/scaley
        img.SetOrigin(trans)
        trans=reference_img.TransformContinuousIndexToPhysicalPoint(index)
        trans=reference_center-trans
        #print(np.array(img.GetOrigin()) - reference_origin)
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        #transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin-trans)
        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
        #print((2*radius)/sizes[0])
        matrix[0,0] = 1/scalex#(2*radius)/sizes[0]#x
        matrix[1,1] = 1/scaley#(2*radius)/sizes[1]#y
        transform.SetMatrix(matrix.ravel())

        centered_transform = sitk.Transform(transform)
        if(imageType==0):
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkLinear, 0.0)
        else:
                img3=sitk.Resample(img, reference_img, centered_transform, sitk.sitkNearestNeighbor, 0.0)
        return setup3d(img3,img3d)







def backregister(img,ref,params,imagetype=1):
    img2=img
    img=resizeimage([512,512],1,img2,img)
    referenceimg = sitk.ReadImage('../referenceImages/pat20050_L_BULB_LONG_cont_20130910085802_bmode_33.mha')
    ref = resizeimage([512, 512],0,ref,referenceimg)
    maskv2 = getwindowv2([512, 512],1,ref,referenceimg,params)
    img=setup2d(np.squeeze(sitk.GetArrayFromImage(img)),maskv2)
    maskv2=setup3d(img,maskv2)
    return getreversewindowv2([512,512],imagetype,maskv2,ref,params)



#img2=sitk.ReadImage('../train/pat20050_L_BULB_LONG_cont_20130910085802_bmode_33.mha')
##img1=sitk.ReadImage('referenceImages/pat20002leftcont_20110309114255_bmode_67.mha')
#img1=sitk.ReadImage('../train_segmentation/pat20050_L_BULB_LONG_cont_20130910085802_Segmentation_33.mha')
#img3=resizeimage([512,512],1,img1,img2)
##sitk.WriteImage(img3,'outscriptaffine_bmodebase.mha')
#img4=getwindow([512,512],1,img3,img2,getwindowdef( img3,0,10,41,[512,512]))
#sitk.WriteImage(img3,'outscriptaffine_bmodebase.mha')
#sitk.WriteImage(img4,'outscriptaffine_window_small.mha')
#img5=resizeimage([256,256],1,img4,img3)
#sitk.WriteImage(img5,'outscriptaffine_window_small_256.mha')
#print(getbound( np.squeeze(sitk.GetArrayFromImage(img5))))
#print(getwindowdef( np.squeeze(sitk.GetArrayFromImage(img4)),0,0,11,[512,512]))

#img3=resizeimageactual([512,512],1,img3,img2,x,y,angle,scalex,scaley,shearx,sheary)
#sitk.WriteImage(img3,'../outscriptaffine_bmode4.mha')
#resizeimage(0,0,img2)
	
