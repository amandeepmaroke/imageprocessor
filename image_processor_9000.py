import numpy as np, tkinter as tk # For array operations
from PIL import ImageTk as itk, Image as im # For image file handling
from tkinter import ttk, filedialog # To create the GUI and file dialog popups

def applyKernel(image, kernel, denom=1):
    denom = int(denom)
    
    # Ensure RGB input (avoids surprises if image is "L" or "RGBA")
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert the kernel and the input image into numpy arrays for 
    # faster processing
    kernel = np.asarray(kernel, dtype=np.float32)
    srcImage = np.asarray(image, dtype=np.float32)   # (H, W, 3)

    
    # Make sure the convolution kernel is a proper shape and size
    kernelHgt, kernelWid = kernel.shape
    if kernelHgt != kernelWid or kernelHgt % 2 == 0 or kernelHgt < 3:
        raise ValueError("kernel must be square, odd-sized, and >= 3x3")

    r = kernelHgt // 2

    # "extend" edges of the array in order to apply kernel to edge/corner pixels
    padded = np.pad(srcImage, ((r, r), (r, r), (0, 0)), mode="edge")

    # windows: (H, W, 3, kernelHgt, kernelWid)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernelHgt, kernelWid), axis=(0, 1)
    )

    # Apply kernel over the LAST TWO axes (kernelHgt, kernelWid), 
    # preserving channels
    out = (windows * kernel[None, None, None, :, :]).sum(axis=(3, 4))

    if denom != 1:
        out = out / denom

    outputImage = np.rint(out)
    outputImage = np.clip(out, 0, 255).astype(np.uint8)

    return im.fromarray(outputImage, mode="RGB")

def boxBlur(image):
    # Apply the "Box Blur" filter to the passed image
    
    # The Box Blur Convolution Kernel
    boxBlurKernel = np.array([[1, 1, 1], [1, 1, 1],[1,1,1]])
    # Denominator of the Convolution Matrix
    boxBlurDenom = 9
    return applyKernel(image, boxBlurKernel, boxBlurDenom)

def emboss(image):
    # Apply the "Emboss" filter to the passed image
    
    # The Box Blur Convolution Kernel
    embossKernel = np.array([[-2, -1, 0], [-1, 1, 1],[0,1,2]])
    # Denominator of the Convolution Matrix
    embossDenom = 1
    return applyKernel(image, embossKernel, embossDenom)

def sharpen(image):
# Apply the "Sharpen" filter to the passed image
    
    # The sharpen Convolution Kernel
    sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1],[0,-1,0]])
    # Denominator of the Convolution Matrix
    sharpenDenom = 1
    return applyKernel(image, sharpenKernel, sharpenDenom)

def unsharp(image):
    # Apply the "Unsharp Masking 5x5" filter to the passed image
    
    # The Unsharp masking 5x5 Convolution Kernel
    unsharpMasking5x5Kernel = np.array([[1, 4, 6, 4, 1], \
                                        [4, 16, 24, 16, 4], \
                                        [6, 24, -476, 24, 6],   \
                                        [4, 16, 24, 16, 4], \
                                        [1, 4, 6, 4, 1]])
    # Denominator of the Convolution Matrix
    unsharpMasking5x5Denom = -256

    return applyKernel(image, unsharpMasking5x5Kernel, unsharpMasking5x5Denom)

def sobel(image):
    # Apply the horizontal and vertical Sobel kernels to the input image and 
    # return the image

    # The Horizontal Sobel Convolution Kernel
    sobelHorizontalKernel = np.array([[-1, 0, 1], [-2, 0, 2],[-1,0,1]])
    sobelHorizontalDenom = 1

    # The Vertical Sobel Convolution Kernel
    sobelVerticalKernel = np.array([[-1, -2, -1], [0, 0, 0],[1,2,1]])
    sobelVerticalDenom = 1
    
    # Apply the horizontal and vertical Sobel kernels to the image and then
    # convert them into a 3D array. 
    horizontalSobelArray = np.asarray    \
        (applyKernel(image, sobelHorizontalKernel, sobelHorizontalDenom))
    verticalSobelArray = np.asarray    \
        (applyKernel(image, sobelVerticalKernel, sobelVerticalDenom))

    # Compute magnitude image directly (vectorized)
    magnitude = np.hypot(horizontalSobelArray, verticalSobelArray)

    # Round, clip, and convert to uint8
    outputImage = np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)

    return im.fromarray(outputImage, mode="RGB")

def gaussian(image):
    # Apply the "Gaussian Blur 5 × 5" filter to the passed image
    
    # The Gaussian Blur 5 × 5 Convolution Kernel
    gaussianBlur5x5Kernel = np.array([[1, 4, 6, 4, 1], \
                                      [4, 16, 24, 16, 4],   \
                                      [6, 24, 36, 24, 6],   \
                                      [4, 16, 24, 16, 4],   \
                                      [1, 4, 6, 4, 1]])
    # Denominator of the Convolution Matrix
    gaussianBlur5x5Denom = 256
    return applyKernel(image, gaussianBlur5x5Kernel, gaussianBlur5x5Denom)

def edgeEnhance(image):
    # Apply the "Edge Enhance" filter to the passed image
    
    # The Edge Enhance Convolution Kernel
    edgeEnhanceKernel = np.array([[0, 0, 0], [-1, 1, 0],[0, 0, 0]])
    # Denominator of the Convolution Matrix
    edgeEnhancedenom = 1
    return applyKernel(image, edgeEnhanceKernel, edgeEnhancedenom)

# Allow for the disabling and enabling of buttons.  
# Used to disable filter buttons until
# the user has selected and opened an image.

def enableButton(button):
    button.config(state=tk.NORMAL)

def disableButton(button):
    button.config(state=tk.DISABLED)

filterCount = 0 #Tracks how many filters have been applied to the image
# Tracks the size of the header in the ListBox so that Image Appllication 
# Messages can properly be placed underneath the header
headerSize = 2 


# Create root window
root = tk.Tk()
root.title("Image Processor 9000")
root.resizable(True, True)


filePath = "" # Stores path of the user-selected file

#Define the dimensions of the display images 
# (doesn't affect the output file dimensions)
displayImageWidth, displayImageHeight = 320, 240

#placeholder image (it's just a black square)
placeholder_pil = im.new("RGB", (displayImageWidth, displayImageHeight))
placeholder = itk.PhotoImage(placeholder_pil)

#input image
input_image = None
input_image_itk = None

# Used to contain scaled-down versions of the input images for 
# display purposes
input_image_display = None
input_image_display_itk = None

#output image
output_image = None
output_image_itk = None

# Used to contain scaled-down versions of the output images for 
# display purposes
output_image_display = None
output_image_display_itk = None

# Create left and right frames
left_frame  =  tk.Frame(root,  width=200,  height=  400,  bg='grey')
left_frame.grid(row=0,  column=0,  padx=10,  pady=5)

right_frame  =  tk.Frame(root,  width=200,  height=  400,  bg='grey')
right_frame.grid(row=0,  column=1,  padx=10,  pady=5)

# Frame Titles
leftFrameTitle = tk.Label(left_frame, 
         text="Original Image", 
         bg="grey", 
         font=("Arial", 12, "bold"))
leftFrameTitle.grid(row=0, column=0, pady=(5,2))

rightFrameTitle = tk.Label(right_frame, 
                           text="Filtered Image", 
                           bg="grey", 
                           font=("Arial", 12, "bold"))
rightFrameTitle.grid(row=0, column=0, pady=(5,2))

# Image panels
leftImagePanel = tk.Label(left_frame, image=placeholder, bg="grey")
leftImagePanel.grid(row=1, column=0, padx=5, pady=5)

rightImagePanel = tk.Label(right_frame, image=placeholder, bg="grey")
rightImagePanel.grid(row=1, column=0, padx=5, pady=5)


left_tool_bar  =  tk.Frame(left_frame,  width=180,  height=185,  bg='grey')
left_tool_bar.grid(row=2,  column=0,  padx=5,  pady=5)

right_tool_bar  =  tk.Frame(right_frame,  width=180,  height=185,  bg='grey')
right_tool_bar.grid(row=2,  column=0,  padx=5,  pady=5)


listBoxFrame = tk.Frame(root,  width=180,  height=185,  bg='grey')
listBoxFrame.grid(row=1,  column=1,  padx=5,  pady=5)
# Create a ListBox to track which filters have been applied to the image
listbox = tk.Listbox(listBoxFrame, height = 13, 
                  width = 42, 
                  bg = "white",
                  activestyle = 'dotbox', 
                  font=("Arial", 10, "bold"),
                  fg = "green")
listbox.grid(row=1,  column=1,  padx=5,  pady=5)

scrollbar = tk.Scrollbar(listBoxFrame, orient=tk.VERTICAL)
scrollbar.grid(row=1,  column=2,  padx=5,  pady=5, sticky="ns")

scrollbar.config(command=listbox.yview)
listbox.config(yscrollcommand=scrollbar.set)

listbox.insert(0, "Filters applied to image will be listed below")
listbox.insert(1, "----------------------------------------------------------------------")


#label = tk.Label(root, text = "Filters Applied") 
#listbox.grid(row=0,  column=0,  padx=5,  pady=5)

# Once an image has had a kernel applied to it, this function is used to 
# replace output_image with the newly filtered image and place it on the right 
# side of the window for viewing.
def place_processed_image(newImage):
    global rightImagePanel, output_image, output_image_itk
    global input_image_display, input_image_display_itk
    global output_image_display, output_image_display_itk

    output_image = newImage.copy()
    output_image_itk = itk.PhotoImage(output_image)
    
    output_image_display = newImage.resize((displayImageWidth, displayImageHeight), 
                                   im.Resampling.LANCZOS)
    output_image_display_itk = itk.PhotoImage(output_image_display)

    rightImagePanel.config(image=output_image_display_itk)
    rightImagePanel.image = output_image_display_itk

# define the behaviour of the "Open Image File" button
def openFileClick():
    global leftImagePanel, rightImagePanel, input_image, input_image_itk
    global output_image, output_image_itk
    global input_image_display, input_image_display_itk
    global output_image_display,output_image_display_itk
    # Open file picker dialog.  The file_path string will contain the absolute
    # path of the image file selected by the user.
    filePath = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[('Jpg Files', '*.jpg'),('PNG Files','*.png')], 
    )
    input_image = im.open(filePath)
    input_image_itk = itk.PhotoImage(input_image)

    input_image_display = input_image.resize((displayImageWidth, displayImageHeight), 
                                             im.Resampling.LANCZOS)
    input_image_display_itk = itk.PhotoImage(input_image_display)

    output_image = input_image.copy()
    output_image_itk = itk.PhotoImage(output_image)

    output_image_display = output_image.resize((displayImageWidth, displayImageHeight), 
                                               im.Resampling.LANCZOS)
    output_image_display_itk = itk.PhotoImage(output_image_display)

    leftImagePanel.config(image=input_image_display_itk)
    leftImagePanel.image = input_image_display_itk

    rightImagePanel.config(image=output_image_display_itk)
    rightImagePanel.image = output_image_display_itk

    # Now that the user has selected an opened an image, 
    # we can go ahead and enable the Filter buttons as well
    # as the "Save Image" and "Clear All Filters" buttons

    enableButton(boxBlurButton)
    enableButton(sharpenButton)
    enableButton(unsharpenButton)
    enableButton(gaussianBlurButton)
    enableButton(sobelButton)
    enableButton(embossButton)
    enableButton(edgeEnhanceButton)

    enableButton(clearImageButton)
    enableButton(saveImageButton)

# define the behaviour of the "Save Image File" button

def saveFileClick():
    output_image_pil = itk.getimage(output_image_itk)
    output_path = filedialog.asksaveasfilename(
        defaultextension=".png", # Sets a default extension
        filetypes=[('PNG Files','*.png')],
        title="Save Image As" 
        )
    output_image_pil.save(output_path)

# Define the behaviour of the "Clear All Filters" button
def clearImageClick():
    global filterCount
    place_processed_image(input_image)
    filterCount = 0
    listbox.delete(0, tk.END)
    listbox.insert(0, "Filters applied to image will be listed below")
    listbox.insert(1, "----------------------------------------------------------------------")
        

# Create and place the "Open Image File" button
openImageButton = ttk.Button(
        left_tool_bar,  
        text="Open Image File...", 
        command = openFileClick
        )
openImageButton.grid(row=0,  column=0,  padx=5,  pady=3,  ipadx=10)

# Create and place the "Save File" button
saveImageButton = ttk.Button(
        right_tool_bar,  
        text="Save Image File...", 
        command = saveFileClick
        )
disableButton(saveImageButton)
saveImageButton.grid(row=0,  column=1,  padx=5,  pady=3,  ipadx=10)

# Create and place the "Clear All Filters" button
clearImageButton = ttk.Button(
        right_tool_bar,  
        text="Clear All Filters", 
        command = clearImageClick
        )
disableButton(clearImageButton)
clearImageButton.grid(row=0,  column=2,  padx=5,  pady=3,  ipadx=10)

###############################
## Convolution Button Events ##
###############################

def boxBlurClick():
    global output_image, filterCount
    newImage = boxBlur(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Box Blur Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def sharpenClick():
    global output_image, filterCount
    newImage = sharpen(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Sharpen Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def embossClick():
    global output_image, filterCount
    newImage = emboss(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Emboss Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def unsharpClick():
    global output_image, filterCount
    newImage = unsharp(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Unsharp masking (5x5) Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def sobelClick():
    global output_image, filterCount
    newImage = sobel(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Sobel Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def gaussianClick():
    global output_image, filterCount
    newImage = gaussian(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Gaussian Blur (5x5) Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

def edgeEnhanceClick():
    global output_image, filterCount
    newImage = edgeEnhance(output_image)
    filterCount +=1
    listbox.insert(filterCount + headerSize, f"{filterCount}: Edge Enhance Filter Applied")
    listbox.see(tk.END)
    place_processed_image(newImage)

#########################
## Convolution Buttons ##
#########################

# Button Frame
frame1 = tk.Frame(root)
frame1.grid(row=1,  column=0,  padx=5,  pady=5,  sticky=tk.E)

# boxBlur button
boxBlurButton = ttk.Button(
    frame1,  
    text="Apply Box Blur filter",  
    command=boxBlurClick)
disableButton(boxBlurButton)
boxBlurButton.grid(row=1,  column=0,  padx=5,  pady=5,  sticky=tk.E)

# Sharpen button
sharpenButton = ttk.Button(
    frame1,
    text="Apply Sharpen filter",
    command=sharpenClick,
)
disableButton(sharpenButton)
sharpenButton.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)

# Unsharp masking 5x5 button
unsharpenButton = ttk.Button(
    frame1,
    text="Apply Unsharp masking (5x5) filter",
    command=unsharpClick,
)
disableButton(unsharpenButton)
unsharpenButton.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)

# Gaussian Blur 5x5 button
gaussianBlurButton = ttk.Button(
    frame1,
    text="Apply Gaussian Blur (5x5) filter",
    command=gaussianClick,
)
disableButton(gaussianBlurButton)
gaussianBlurButton.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E)

# Sobel button
sobelButton = ttk.Button(
    frame1,
    text="Apply Sobel filter",
    command=sobelClick,
)
disableButton(sobelButton)
sobelButton.grid(row=5, column=0, padx=5, pady=5, sticky=tk.E)

# emboss button
embossButton = ttk.Button(
    frame1,
    text="Apply Emboss filter",
    command=embossClick,
)
disableButton(embossButton)
embossButton.grid(row=6, column=0, padx=5, pady=5, sticky=tk.E)

# Edge Enhance button
edgeEnhanceButton = ttk.Button(
    frame1,
    text="Apply Edge Enhance filter",
    command=edgeEnhanceClick,
)
disableButton(edgeEnhanceButton)
edgeEnhanceButton.grid(row=7, column=0, padx=5, pady=5, sticky=tk.E)

root.mainloop()