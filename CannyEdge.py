import matplotlib as mp #used at end for displaying image
from warnings import simplefilter as sf #Used to ignore FutureWarning from try except structure in imagefile() 
import numpy as np
sf(action='ignore', category=FutureWarning) #ignore future warning. No problems are caused by doing this. It just comes from comparing string and numpy array in error handling later. Is just used because warning text is annoying

def getname():
    print('This program detects edges of an image.')
    filename=input('Enter full file path to png image: ')#input filename for any file, not just in folder of program. Not case sensitive
    filetype = filename[-3:]
    if filetype == 'png': #checks if file is png
        return filename
    else: 
        return 'ERROR: Input file is not .png'

def imagefile(filename): #returns image array from file returned from getname
    import matplotlib as mp
    import numpy as np 
    try: #Try/except makes sure file exists
        with open(filename,"rb") as f:
            image = mp.image.imread(f)#using matplot to create array from image pixel data
    except FileNotFoundError: #if file doesn't exist, say that
        return 'ERROR: Input file does not exist'
    if len(np.shape(image)) == 2 : #This if statement converts 1 channel png images to 3 channel to avoid errors
        image = np.dstack((image,image,image)) #makes it so image is 3d array with R,G,B = Y,Y,Y of initial image
    return image

def grey(image): #edited version of original img_to_grey() for compatibility. Input must be array, not file
    R = np.array(image[:, :, 0]) #makes an array of each color RGB
    G = np.array(image[:, :, 1])
    B = np.array(image[:, :, 2])
    #Y = 0.2126R + 0.7152G + 0.0722B is the conversion factor as specified by course standard
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B #apply conversion to all 3 color arrays
    for i in range(3): #replaces all image color channels RGB with Y
        image[:,:,i] = Y
    return image

def save(image,file): #this saves an image array(image) to file as png 
    import matplotlib.image as mpimage
    mpimage.imsave(file,image)
    
def img_th(image): #Thresholding function- works by subtracting from all values to make values <threshold 0(black), multiplies to raise all others to >1(white)
    import numpy as np    
    image -=.075 #threshold is 0.35. Determined by trial and error so the example images looked good in image correction
    image *=100 #keeps zeroes at 0, raises all nonzero values to 1 or greater(reads as 1/white)
    image = -image #inverting so lines are black on white bg
    image = image[:,:,0]
    for x in image:
        for y in x:
            if y<=0:
                y=0
            else:
                y = 1
    image = np.dstack((image,image,image))  #put the image back together as rgb not Y
    return image

def smooth(inputimg): #really long function that blurs the image. takes forever to run but thats ok for this project
    import numpy
    from PIL import Image
    img = Image.open(inputimg) #opens an image saved from earlier function
    
    imgArr = numpy.asarray(img) #opens PIL image as array
    # blur radius in pixels
    radius = 2 # 5x5 blur window
    # blur window length in pixels
    windowLen = radius*2+1
    # columns (x) image width in pixels
    imgWidth = imgArr.shape[1]
    # rows (y) image height in pixels
    imgHeight = imgArr.shape[0]
    #simple box/window blur
    def doblur(imgArr):
        # create array for processed image based on input image dimensions
        imgB = numpy.zeros((imgHeight,imgWidth,3),numpy.uint8) #PIL uses 0-255 for images so we do that here
        imgC = numpy.zeros((imgHeight,imgWidth,3),numpy.uint8)
    
        # blur horizontal row by row
        for row in range(imgHeight):
            # RGB color values
            totalR = 0
            totalG = 0
            totalB = 0
    
            # calculate blurred value of first pixel in each row
            for rads in range(-radius, radius+1):
                if (rads) >= 0 and (rads) <= imgWidth-1:
                    totalR += imgArr[row,rads][0]/windowLen
                    totalG += imgArr[row,rads][1]/windowLen
                    totalB += imgArr[row,rads][2]/windowLen
    
            imgB[row,0] = [totalR,totalG,totalB]
    
            # calculate blurred value of the rest of the row based on
            # unweighted average of surrounding pixels within blur radius
            # using sliding window totals (add incoming, subtract outgoing pixels)
            for col in range(1,imgWidth):
                if (col-radius-1) >= 0:
                    totalR -= imgArr[row,col-radius-1][0]/windowLen
                    totalG -= imgArr[row,col-radius-1][1]/windowLen
                    totalB -= imgArr[row,col-radius-1][2]/windowLen
                if (col+radius) <= imgWidth-1:
                    totalR += imgArr[row,col+radius][0]/windowLen
                    totalG += imgArr[row,col+radius][1]/windowLen
                    totalB += imgArr[row,col+radius][2]/windowLen
    
                # put average color value into imgB pixel
    
                imgB[row,col] = [totalR,totalG,totalB]
    
        # blur vertical because otherwise it looks weird
    
        for col in range(imgWidth):
            totalR = 0
            totalG = 0
            totalB = 0
    
            for rads in range(-radius, radius+1):
                if (rads) >= 0 and (rads) <= imgHeight-1:
                    totalR += imgB[rads,col][0]/windowLen
                    totalG += imgB[rads,col][1]/windowLen
                    totalB += imgB[rads,col][2]/windowLen
    
            imgC[0,col] = [totalR,totalG,totalB]
    
            for row in range(1,imgHeight):
                if (row-radius-1) >= 0:
                    totalR -= imgB[row-radius-1,col][0]/windowLen
                    totalG -= imgB[row-radius-1,col][1]/windowLen
                    totalB -= imgB[row-radius-1,col][2]/windowLen
                if (row+radius) <= imgHeight-1:
                    totalR += imgB[row+radius,col][0]/windowLen
                    totalG += imgB[row+radius,col][1]/windowLen
                    totalB += imgB[row+radius,col][2]/windowLen
    
                imgC[row,col] = [totalR,totalG,totalB]
    
        return imgC
    
    # number of times to run blur because 1 isnt enough to get rid of noise around edges of lines. 2 preserves detail in the hat thing while removing noise
    blurPasses = 2
    
    # temporary image array for multiple passes
    imgTmp = imgArr
    
    for k in range(blurPasses): 
        imgTmp = doblur(imgTmp)
        print("blur pass #",k+1,"of",blurPasses,"done") #works like a progress bar because this process is slow on low power pcs
    
    
    imgOut = Image.fromarray(numpy.uint8(imgTmp))
    
    imgOut.save('testimage-processed.png', 'PNG') #saves image so we stop using PIL image and can go back to numpy

def edges(image):
    
    #vertical filter
    fy = [[-1,-2,-1], [0,0,0], [1,2,1]]
    
    #horizontal filter
    fx = [[-1,0,1], [-2,0,2], [-1,0,1]]
    
    #get the dimensions of the image
    n,m,d = image.shape
    
    #make copy for output
    edges_img = image.copy()
    
    #loop over all pixels in the image
    for row in range(3, n-2): #does this for not the edges of the image to avoid errors
        for col in range(3, m-2):
            
            #window of image
            window = image[row-1:row+2, col-1:col+2, 0]
            
            #apply the vertical filter
            vertical_transformed_pixels = fy*window
            #remap the vertical score
            vertical_score = vertical_transformed_pixels.sum()/4
            
            #apply the horizontal filter
            horizontal_transformed_pixels = fx*window
            #remap the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum()/4
            
            #using sqrt(x^2+y^2) for combination
            combined = (vertical_score**2 + horizontal_score**2)**.5
            
            #put this into the output
            edges_img[row, col] = [combined]*3
    
    #this makes the image output better with more detail for the threshold function
    edges_img /= edges_img.max()
    print('edge detection done')
    return edges_img    



#executions
image = getname()
if image == 'ERROR: Input file is not .png': #error handling 1- checks for error in getname()
    print(image) #prints normally if it isnt
else: #this structure of if else makes there not be errors for any type of input
    image = imagefile(image)
    if image == 'ERROR: Input file does not exist': #error handling 2- checks for error from imagefile()
        print(image) #prints normally if module is not installed
    else: #If the above functions produce an image, all following functions work on the image. No errors should exist after this point
        image = grey(image) 
        save(image,'temp.png')
        smooth('temp.png') #this function uses a file, not an array so we use temp saved above
        image = imagefile('testimage-processed.png') #smooth outputs a file, not an array so we reopen the image temp for array numpy
        image = edges(image)
        image = img_th(image)
        mp.pyplot.imshow(image) #final image is printed to console
        