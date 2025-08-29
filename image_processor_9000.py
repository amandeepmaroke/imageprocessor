import numpy as np
from PIL import Image as im, ImageChops, UnidentifiedImageError

def printHeader(projectName):
# This function will print a header with a string ProjectName as the input. 
# The function determines the longest string in the header, adds padding on
# either side, and then pads all the other lines so that they are all the same
# length.

    def paddedLine(text, bannerLength):
        # This function will output a string that is a line of text with the 
        # proper padding given the desired width of the banner.
        toReturn = "#"
        paddingSize = int((bannerLength - len(text))/2)
        for i in range (paddingSize):
            toReturn = toReturn + " "
        toReturn = toReturn + text
        
        while len(toReturn) < bannerLength:
            toReturn = toReturn + " "
        
        toReturn = toReturn + "#"

        return toReturn
    
    # Lines that won't change between projects
    nameLine = "By: Amandeep Maroke"
    studendNumberLine = "By: 031596455"

    # Determine the length of each of the lines
    projectNameLength = len(projectName)
    nameLength = len(nameLine)
    studentNumberLength = len(studendNumberLine)

    # Determine the longest line
    longestLine = max(projectNameLength, nameLength, studentNumberLength)

    # Add 5 spaces of padding on each side of the longest line, and that will 
    # determine the width of the title banner. Added 1 as well so that loops
    # create the correct number of characters before stopping.

    bannerWidth = longestLine + 5 + 5 + 1

    # Top and bottom of banner
    topAndBottom = ""
    for i in range (bannerWidth + 1):
        topAndBottom = topAndBottom + "#"
    
    # Array to hold all the lines to print
    linesToPrint = []

    # Add the first line to the list of lines to print
    linesToPrint.append(topAndBottom)

    # Pad the middle lines and append them to the list of lines to print
    linesToPrint.append(paddedLine(projectName, bannerWidth))
    linesToPrint.append(paddedLine(nameLine, bannerWidth))
    linesToPrint.append(paddedLine(studendNumberLine, bannerWidth))

    # Add the last line to the list of lines to print
    linesToPrint.append(topAndBottom)
    
    # Print all the elements from the list
    for i in linesToPrint:
        print(i)

def exit_program():
    print("\nGoodbye!\n")
    raise SystemExit()

def applyKernel(image, kernel, denom=1):
    # This function applies a convolution kernel (kernel) to the image array
    # (image) and returns an image with the applied kernel/filter. It requires:
    # 1) 'image' - A 3-D numpy array that was obtained using the asArray method
    # 2) 'kernel' - a numpy array object of shape axa, where a >= 3, 
    # 3) 'denom' - an 'int' value that indicates the denominator of the kernel.
    #     Default value is 1 if a denominator is not needed.

    # This method also prints out a progress message indicating what percentage
    # of the kernel convolution has been completed.

    # Make sure the matrix denominator is an integer
    denom = int(denom)

    # convert the image to a 3D array of shape (height, width, 3).  The 3
    # layers correspond to the RGB values of each pixel
    arrayIn = np.asarray(image)

    # Determine the width/height of convolution kernel (should be the same)
    kernelSize = int(kernel.shape[0])
    
    # Determine the "radius" of the convolution kernel. 
    radius = int((kernelSize - 1)/2)

    # determine the width and height of the input image
    width = np.size(arrayIn, 0)
    height = np.size(arrayIn, 1)

    # Create an copy of the input image.  This is the copy we will apply the 
    # convolution kernel to.
    outputImage = image.copy()

    # Count the number of pixels that need to be processed.  This is for 
    # progress bar purposes
     
    totalPixels = height*width
    
    #tracks how many pixels have been processed
    processedPixels = 0
    
    #Tracks the previous percentage of pixels that were processed
    oldPercentage = 0

    for x in range(width):
        for y in range(height):
            # determine the RGB values for the current pixel using the Box Blur
            # calculation.

            # Initialize the RGB pixel values
            rValue = 0
            gValue = 0
            bValue = 0

            # determine the x and y values of the top-left corner of the image
            # where the convolution kernel will start 

            xStart = x - radius
            yStart = y - radius
            
            for i in range(0,kernelSize):
                for j in range(0,kernelSize):
                    
                    # Edge handling using the "extend" method .  If either of 
                    # the X or Y values of where we need to look in the 
                    # image array are outside the height or width of the
                    # array, this calculation ensures that we are looking
                    # at the closest edge, effectively "extending" the edges
                    # of the original image to provide values for the
                    # convolution calculation.
                    arrayInX = min(max(0,(xStart + i)), width-1)
                    arrayInY = min(max(0,(yStart + j)), height-1)

                    
                    # Perform convolution calculation on each of the RGB values
                    rValue = rValue + (arrayIn[arrayInX,arrayInY][0])   \
                        * kernel[i][j]
                    gValue = gValue + (arrayIn[arrayInX,arrayInY][1])   \
                        * kernel[i][j]
                    bValue = bValue + (arrayIn[arrayInX,arrayInY][2])   \
                        * kernel[i][j]

            # Divide the RBG values by the convolution kernel denonimator
            rValue = round(rValue/denom)
            gValue = round(gValue/denom)
            bValue = round(bValue/denom)

            # Apply RGB values to the corresponding pixel in the outputImage
            outputImage.putpixel((y,x), (rValue, gValue, bValue))
            
            #Determine what percentage of pixels have been processed
            currentPercentage =  round(100*processedPixels/totalPixels) 

            # once current percentage is greater than old percentage, we update 
            # old percentage and print it out with a carriage return at the end
            # so that the next statement will overwrite the current one 
            
            if currentPercentage > oldPercentage:
                oldPercentage = currentPercentage
                
                # Print the next percentage message
                print(f" {currentPercentage}% complete", end="\r")
           
            #update the count of processed pixels
            processedPixels +=1

    
    print("\033[K" + "100% complete")
    return outputImage
    
def sobel(image):
    # Apply the horizontal and vertical Sobel kernels to the input image and 
    # return the image

    # The Horizontal Sobel Convolution Kernel
    sobelHorizontalKernel = np.array([[-1, 0, 1], [-2, 0, 2],[-1,0,1]])
    sobelHorizontalDenom = 1

    # The Vertical Sobel Convolution Kernel
    sobelVerticalKernel = np.array([[-1, -2, -1], [0, 0, 0],[1,2,1]])
    sobelVerticalDenom = 1
    
    print("\nApplying \"Sobel\" filter to the image:")
    
    # Apply the horizontal and vertical Sobel kernels to the image and then
    # convert them into a 3D array. 
    print("Step 1 of 3: Applying the Horizontal Sobel kernel to the image")
    horizontalSobelArray = np.asarray    \
        (applyKernel(image, sobelHorizontalKernel, sobelHorizontalDenom))
    print("Step 2 of 3: Applying the Vertical Sobel kernel to the image")
    verticalSobelArray = np.asarray    \
        (applyKernel(image, sobelVerticalKernel, sobelVerticalDenom))
    outputImage = image.copy()


    # convert the image to a 3D array of shape (height, width, 3).  The 3
    # layers store the RGB values of each pixel
    arrayIn = np.asarray(outputImage)

    #determine the width and height of the image
    width = np.size(arrayIn, 0)
    height = np.size(arrayIn, 1)

    print("Step 3 of 3: Combining \"X\" and \"Y\" kernels")

    # Count the number of pixels that need to be processed.  This is for 
    # progress bar purposes
     
    totalPixels = height*width

    #tracks how many pixels have been processed
    processedPixels = 0
    
    #Tracks the previous percentage of pixels that were processed
    oldPercentage = 0

    for x in range(width):
        for y in range(height):
            # Perform the magnitude calculation for each of the RGB values

            rValue = np.hypot(horizontalSobelArray[x,y][0], \
                              verticalSobelArray[x,y][0])
            gValue = np.hypot(horizontalSobelArray[x,y][1], \
                              verticalSobelArray[x,y][1]) 
            bValue = np.hypot(horizontalSobelArray[x,y][2], \
                              verticalSobelArray[x,y][2])  

            rValue = round(rValue)
            gValue = round(gValue)
            bValue = round(bValue)

            # Apply RGB values to the corresponding pixel in the outputImage
            outputImage.putpixel((y,x), (rValue, gValue, bValue))

            # Determine what percentage of pixels have been processed
            currentPercentage =  round(100*processedPixels/totalPixels) 

            # once current percentage is greater than old percentage, we update 
            # old percentage and print it out with a carriage return at the end
            # so that the next statement will overwrite the current one 
            
            if currentPercentage > oldPercentage:
                oldPercentage = currentPercentage
                
                # Print the next percentage message
                print(f" {currentPercentage}% complete", end="\r")
           
            #update the count of processed pixels
            processedPixels +=1

    print("\033[K" + "100% complete")
    return outputImage

def boxBlur(image):
    # Apply the "Box Blur" filter to the passed image
    
    # The Box Blur Convolution Kernel
    boxBlurKernel = np.array([[1, 1, 1], [1, 1, 1],[1,1,1]])
    # Denominator of the Convolution Matrix
    boxBlurDenom = 9
    print("\nApplying \"Box Blur\" filter to the image")
    imageToReturn = applyKernel(image, boxBlurKernel, boxBlurDenom)

    return imageToReturn

def sharpen(image):
    # Apply the "Sharpen" filter to the passed image
    
    # The sharpen Convolution Kernel
    sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1],[0,-1,0]])
    # Denominator of the Convolution Matrix
    sharpenDenom = 1
    print("\nApplying \"Sharpen\" filter to the image")
    imageToReturn = applyKernel(image, sharpenKernel, sharpenDenom)

    return imageToReturn

def unsharpMasking5x5(image):
    # Apply the "Unsharp Masking 5x5" filter to the passed image
    
    # The Unsharp masking 5x5 Convolution Kernel
    unsharpMasking5x5Kernel = np.array([[1, 4, 6, 4, 1], \
                                        [4, 16, 24, 16, 4], \
                                        [6, 24, -476, 24, 6],   \
                                        [4, 16, 24, 16, 4], \
                                        [1, 4, 6, 4, 1]])
    # Denominator of the Convolution Matrix
    unsharpMasking5x5Denom = -256
    print("\nApplying \"Unsharp Masking 5x5\" filter to the image")
    imageToReturn = applyKernel(image, unsharpMasking5x5Kernel, 
                                unsharpMasking5x5Denom)

    return imageToReturn

def gaussianBlur5x5(image):
    # Apply the "Gaussian Blur 5 × 5" filter to the passed image
    
    # The Gaussian Blur 5 × 5 Convolution Kernel
    gaussianBlur5x5Kernel = np.array([[1, 4, 6, 4, 1], \
                                      [4, 16, 24, 16, 4],   \
                                      [6, 24, 36, 24, 6],   \
                                      [4, 16, 24, 16, 4],   \
                                      [1, 4, 6, 4, 1]])
    # Denominator of the Convolution Matrix
    gaussianBlur5x5Denom = 256
    print("\nApplying \"Gaussian Blur 5 × 5\" filter to the image")
    imageToReturn = applyKernel\
                    (image, gaussianBlur5x5Kernel, gaussianBlur5x5Denom)

    return imageToReturn

def emboss(image):
    # Apply the "Emboss" filter to the passed image
    
    # The Box Blur Convolution Kernel
    embossKernel = np.array([[-2, -1, 0], [-1, 1, 1],[0,1,2]])
    # Denominator of the Convolution Matrix
    embossDenom = 1
    print("\nApplying \"Emboss\" filter to the image")
    imageToReturn = applyKernel(image, embossKernel, embossDenom)

    return imageToReturn

def edgeEnhance(image):
    # Apply the "Edge Enhance" filter to the passed image
    
    # The Edge Enhance Convolution Kernel
    edgeEnhanceKernel = np.array([[0, 0, 0], [-1, 1, 0],[0, 0, 0]])
    # Denominator of the Convolution Matrix
    edgeEnhancedenom = 1
    print("\nApplying \"Edge Enhance\" filter to the image")
    imageToReturn = applyKernel(image, edgeEnhanceKernel, edgeEnhancedenom)

    return imageToReturn


# Print the header of the lab
printHeader("Image-A-Tron 9000")



# Check to see if "image1.jpg" exists in the folder, and give the user an 
# error message if it's not.
while True:
    try:
        img = im.open("image1.jpg")
        print("\n######################")
        print("#                    #")
        print("#  image1.jpg found  #")
        print("#                    #")
        print("######################")
        break
    except FileNotFoundError:
        fileNotFoundText = "\nimage1.jpg WAS NOT FOUND!\nWhat would you like " \
        "to do?\n\nQ - Quit\nT - Try again (be sure to check the file name " \
        "and location first)\nMake a selection: "
        tryAgain = input(fileNotFoundText)
 
        # We give the user the option to either exit the program, or move/rename
        # the file and try again
        while True:
            if tryAgain.lower() == "t":
                break
            elif tryAgain.lower() == "q":
                exit_program()
            #If the user's input was not a valid option
            else:
                tryAgain = input("\nThat's not a valid option.  Type \"q\" to" \
                " quit, or check the file location and name and type \"t\" to" \
                " try again: ")

    except UnidentifiedImageError:
        fileNotFoundText = "\nERROR: image1.jpg is either corrupt or not a " \
        "valid .jpg image\nWhat would you like to do?\n\nQ - Quit\nT - Try " \
        "again (be sure to check the file name and location first)\nMake a " \
        "selection: "
        tryAgain = input(fileNotFoundText)
 
        # Give the user the option to either exit the program, or move/rename
        # the file and try again
        while True:
            if tryAgain.lower() == "t":
                break
            elif tryAgain.lower() == "q":
                exit_program()
            #If the user's input was not a valid option
            else:
                tryAgain = input("\nThat's not a valid option.  \nType \"q\" " \
                "to quit, or check the file location and name and type \"t\" " \
                "to try again: ")

# A list of filters the user can pick from
filterSelections = {"1": boxBlur, "2": sharpen, "3":unsharpMasking5x5, 
                    "4":gaussianBlur5x5, "5":sobel, "6":emboss, "7":edgeEnhance}

whatToDo = input("\nWhat would you like to do?:\n1) Apply Box Blur filter\n2) "\
"Apply Sharpen filter\n3) Apply Unsharp masking 5x5 filter\n4) Apply " \
"Gaussian Blur 5x5 filter\n5) Apply Sobel filter\n6) Apply Emboss filter\n7) " \
"Apply Edge Enhance filter\n8) Exit\nMake a selection: ")



# perform the function corresponding to the user input
while True:
    try:
        # Quit the program if requested by the user
        if whatToDo == "8":
            exit_program()
        # Apply the requested convolution kernel to the image matrix
        outputImage = filterSelections[whatToDo](img)
        
        # Show the user a preview of the image
        outputImage.show()

        while True:
            # Ask the user what they want to do with this image
            whatToDoNext = input("\nWhat would you like to do with this " \
                                "image?\n1) Save the image and exit\n2) " \
                                "Apply another filter\n3) Discard changes " \
                                "and exit\nMake a selection: ")

            if whatToDoNext == "1":
                outputImage.save("processed_image.jpg")
                print("\nImage saved as \"processed_image.jpg\"")
                exit_program()
            elif whatToDoNext == "2":
                whatToDo = input("\nWhich filter would you like to apply?:" \
                                 "\n1) Apply Box Blur filter\n2) Apply " \
                                 "Sharpen filter\n3) Apply Unsharp masking " \
                                 "5x5 filter\n4) Apply Gaussian Blur 5x5 " \
                                 "filter\n5) Apply Sobel filter\n6) Apply " \
                                 "Emboss filter\n7) Apply Edge Enhance " \
                                 "filter\nMake a selection: ")
                # Apply the requested convolution kernel to the image matrix
                outputImage = filterSelections[whatToDo](outputImage)
            
                # Show the user a preview of the image
                outputImage.show()
            elif whatToDoNext == "3":
                # Make sure the user actually WANTS to exit
                while True:
                    confirmExit = input("\nAre you sure you want to exit?" \
                                        " ALL CHANGES WILL BE DISCARDED"
                                        " (Y/N): ")
                    if confirmExit.lower() == "y" \
                        or confirmExit.lower() == "yes":
                        exit_program()
                    elif confirmExit.lower() == "n" \
                        or confirmExit.lower() == "no":
                        break
                    else:
                        print("\nThat's not a valid selection")
            else:
                print("\nThat's not a valid selection")
        break
    except KeyError:
        print("\nThat's not a valid selection")
        whatToDo = input("\nWhat would you like to do?:\n1) Apply Box Blur " \
        "filter\n2) Apply Sharpen filter\n3) Apply Unsharp masking 5x5 " \
        "filter\n4) Apply Gaussian Blur 5x5 filter\n5) Apply Sobel " \
        "filter\n6) Apply Emboss filter\n7) Apply Edge Enhance filter\n8) " \
        "Exit\nMake a selection: ")